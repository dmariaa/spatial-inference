"""Evaluate a frozen convex-kernel checkpoint without training or selection."""
import hashlib
import json
from pathlib import Path

import click
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data.cache import load_aq_sensor_cache
from metraq_gnn.data.dataset import SensorGraphWindowDataset
from metraq_gnn.data.kernel_window import KernelWindowDataset
from metraq_gnn.data.splits import split_sensor_nodes
from metraq_gnn.evaluation import (
    collect_sensor_predictions, summarize_all_predictions, summarize_sensor_predictions,
)
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel


@click.command(context_settings={"show_default": True})
@click.option("--run-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--reference-summary", type=click.Path(exists=True, path_type=Path))
@click.option("--device", default="cpu")
@click.option("--batch-size", default=64, type=click.IntRange(min=1))
def main(run_dir, cache, output, reference_summary, device, batch_size):
    """Evaluate all eligible 2024 windows using saved TRAIN+validation context."""
    if output.exists():
        raise click.ClickException("output already exists; choose a new directory")
    torch.set_num_threads(1)
    config = json.loads((run_dir / "config.json").read_text())
    training = json.loads((run_dir / "summary.json").read_text())
    checkpoint_path = run_dir / "best-model.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != training["best_epoch"]:
        raise click.ClickException("checkpoint is not the validation-selected epoch")
    aq = load_aq_sensor_cache(cache)
    with np.load(run_dir / "partition.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sensor_ids"], aq.sensor_ids)
        np.testing.assert_array_equal(saved["sensor_node_indices"], aq.sensor_node_indices)
        train, validation, test = (
            np.asarray(saved[name], dtype=bool) for name in
            ("train_mask", "validation_mask", "test_mask"))
    for mask in (train, validation, test):
        if mask.shape != aq.grid_shape:
            raise click.ClickException("partition shape differs from cache grid")
    if (train & validation).any() or (train & test).any() or (validation & test).any():
        raise click.ClickException("saved partitions overlap")
    np.testing.assert_array_equal(np.flatnonzero(train), config["train_nodes"])
    np.testing.assert_array_equal(np.flatnonzero(validation), config["validation_nodes"])
    test_ids = aq.sensor_ids[test.reshape(-1)[aq.sensor_node_indices]].tolist()
    if sorted(test_ids) != sorted(config["test_sensor_ids"]):
        raise click.ClickException("TEST sensor identities differ from training config")
    reference = None
    if reference_summary is not None:
        reference = json.loads(reference_summary.read_text())
        masks = split_sensor_nodes(
            aq, validation_nodes=reference["validation_nodes"],
            test_nodes=reference["test_nodes"], seed=reference["seed"],
            test_sensor_ids=reference.get("test_sensor_ids"))
        for actual, expected in zip((train, validation, test), masks, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for name in ("normalization_mean", "normalization_std"):
            np.testing.assert_allclose(config[name], reference[name], rtol=1e-6)
        if reference.get("test_windows_index") is not None or reference.get("stride", 1) != 1:
            raise click.ClickException("reference requires a selected-window evaluator")
    mean = np.asarray(config["normalization_mean"], dtype=np.float32)
    std = np.asarray(config["normalization_std"], dtype=np.float32)
    dataset = KernelWindowDataset(SensorGraphWindowDataset(
        cache=aq, hours=config["hours"], context_sensor_mask=train | validation,
        forbidden_context_mask=test, target_sensor_mask=test,
        split_start="2024-01-01", split_end="2024-12-31 23:00",
        normalization_mean=mean, normalization_std=std, seed=config["seed"]))
    if reference is not None and len(dataset) != reference["test_windows"]:
        raise click.ClickException("TEST window count differs from reference")
    model = ConvexGeometricKernel(mode=config["mode"], cell_size_m=config["cell_size_m"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    click.echo(f"Evaluating {len(dataset)} TEST windows; checkpoint epoch {checkpoint['epoch']+1}.")
    predictions = collect_sensor_predictions(
        model, DataLoader(dataset, batch_size=batch_size, shuffle=False),
        mean=mean, std=std, device=device, sensor_ids=aq.sensor_ids,
        sensor_node_indices=aq.sensor_node_indices,
        node_positions=aq.graph.pos.numpy(), magnitude_ids=aq.magnitude_ids)
    keys = ["window_end", "sensor_id", "magnitude_id"]
    if predictions.duplicated(keys).any():
        raise click.ClickException("duplicate TEST prediction keys")
    if not np.isfinite(predictions[["target", "prediction"]].to_numpy()).all():
        raise click.ClickException("nonfinite TEST targets/predictions")
    if reference is not None and len(predictions) != reference["test_target_count"]:
        raise click.ClickException("TEST target count differs from reference")
    metrics = summarize_all_predictions(predictions)
    summary = dict(
        **metrics, test_windows=len(dataset), test_sensor_ids=test_ids,
        checkpoint_epoch=checkpoint["epoch"], checkpoint_sha256=checkpoint_hash,
        context="TRAIN+VALIDATION", period="2024", test_evaluated=True,
        reference_summary=str(reference_summary) if reference_summary else None)
    if reference is not None:
        summary["reference_metrics"] = {
            key: reference[key] for key in
            ("test_mae_physical", "test_rmse_physical", "test_wape")}
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise click.ClickException("checkpoint changed during evaluation")
    output.mkdir(parents=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    summarize_sensor_predictions(predictions).to_csv(output / "by-sensor.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    click.echo(json.dumps(summary, indent=2))
