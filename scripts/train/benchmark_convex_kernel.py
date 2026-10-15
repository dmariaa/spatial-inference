"""Compare complete local training steps, including batch preparation."""
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as GraphLoader

from metraq_gnn.data.cache import AQSensorCache, load_aq_sensor_cache
from metraq_gnn.data.dataset import SensorGraphWindowDataset
from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.data.kernel_window import (
    CachedKernelDataset, KernelWindowDataset, collate_kernel_snapshots,
)
from metraq_gnn.data.splits import compute_training_normalization, split_sensor_nodes
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel


def synthetic_cache():
    times = pd.date_range("2015-01-01", periods=1047, freq="h").as_unit("ns")
    rng = np.random.default_rng(123)
    nodes = np.sort(rng.choice(24*23, size=24, replace=False))
    values = rng.normal(30., 10., (len(times), 24, 1)).astype(np.float32)
    return AQSensorCache(
        path=Path("."), metadata={"grid_shape": [24, 23], "cell_size_m": 1000},
        timestamps_ns=times.asi8, sensor_ids=np.arange(24),
        sensor_node_indices=nodes, magnitude_ids=np.array([8]),
        values=values, availability=np.ones_like(values, dtype=bool),
        graph=build_grid_graph({"grid": np.zeros((24, 23))}))


@click.command()
@click.option("--cache", type=click.Path(exists=True, path_type=Path),
              help="Use a real local AQ cache instead of synthetic data.")
@click.option("--windows", default=256, type=click.IntRange(min=1))
@click.option("--batch-size", default=64, type=click.IntRange(min=1))
@click.option("--repeats", default=3, type=click.IntRange(min=1))
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
def main(cache, windows, batch_size, repeats, device):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    aq = synthetic_cache() if cache is None else load_aq_sensor_cache(cache)
    train, validation, test = split_sensor_nodes(
        aq, validation_nodes=4, test_nodes=4, seed=42)
    mean, std = compute_training_normalization(
        aq, train_node_mask=train, start="2015-01-01", end="2022-12-31 23:00")
    common = dict(cache=aq, hours=24, context_sensor_mask=train,
                  forbidden_context_mask=validation | test, target_fraction=.25,
                  normalization_mean=mean, normalization_std=std, seed=42,
                  split_start="2015-01-01", split_end="2022-12-31 23:00")
    begin = perf_counter()
    old = KernelWindowDataset(SensorGraphWindowDataset(**common))
    old_setup = perf_counter() - begin
    begin = perf_counter()
    new = CachedKernelDataset(**common)
    new_setup = perf_counter() - begin
    if old.dataset.end_indices != new.end_indices:
        raise ValueError("window selections differ")
    count = min(windows, len(new))
    old_loader = GraphLoader(Subset(old, range(count)), batch_size=batch_size)
    new_loader = DataLoader(Subset(new, range(count)), batch_size=batch_size,
                            collate_fn=collate_kernel_snapshots)
    legacy = ConvexGeometricKernel().to(device)
    vectorized = deepcopy(legacy).prepare_geometry(
        aq.graph.pos, np.flatnonzero(train))
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    def step(model, batch):
        model.zero_grad(set_to_none=True)
        batch = batch.to(device)
        prediction = model(batch)
        loss = (prediction[batch.target_mask] - batch.y[batch.target_mask]).square().mean()
        loss.backward()
    # Validate the measured data and predictions before timing either path.
    old_batch = next(iter(old_loader)).to(device)
    new_batch = next(iter(new_loader)).to(device)
    torch.testing.assert_close(old_batch.kernel_values, new_batch.kernel_values.flatten(0, 1))
    torch.testing.assert_close(old_batch.target_mask, new_batch.target_mask.flatten(0, 1))
    with torch.no_grad():
        torch.testing.assert_close(
            legacy(old_batch), vectorized(new_batch).flatten(0, 1),
            rtol=3e-5, atol=3e-6)
    # Warm up both execution paths before timing, with identical data.
    step(legacy, next(iter(old_loader)))
    step(vectorized, next(iter(new_loader)))
    measurements = {}
    for name, model, loader, setup in (
        ("original", legacy, old_loader, old_setup),
        ("optimized", vectorized, new_loader, new_setup),
    ):
        times = []
        for _ in range(repeats):
            sync()
            begin = perf_counter()
            for batch in loader:
                step(model, batch)
            sync()
            times.append(perf_counter()-begin)
        measurements[name] = dict(
            setup_seconds=setup, median_seconds=float(np.median(times)),
            windows_per_second=count/float(np.median(times)), seconds=times)
    measurements["speedup"] = (
        measurements["original"]["median_seconds"] / measurements["optimized"]["median_seconds"])
    measurements.update(device=device, windows=count, batch_size=batch_size,
                        synthetic=cache is None, repeats=repeats)
    click.echo(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
