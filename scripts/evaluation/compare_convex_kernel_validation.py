"""Compare two frozen convex-kernel pilots on identical 2023 validation targets."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data.cache import load_aq_sensor_cache
from metraq_gnn.data.kernel_window import CachedKernelDataset
from metraq_gnn.data.wind_cache import file_sha256, load_wind_source_cache
from metraq_gnn.evaluation import (
    collect_sensor_predictions, summarize_all_predictions, summarize_sensor_predictions,
)
from metraq_gnn.model import ConvexGeometricKernel

KEYS = ["window_end", "sensor_id", "magnitude_id"]
SHARED = ("mode", "seed", "hours", "cell_size_m", "epsilon_m", "distance_scale_m",
          "source_policy", "implementation", "train_start", "train_end", "train_period",
          "validation_period", "normalization_period", "normalization_mean", "normalization_std",
          "train_nodes", "validation_nodes", "test_sensor_ids", "batch_size", "epochs_requested",
          "learning_rate", "early_stopping_patience", "optimization_loss",
          "train_windows", "validation_windows")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_partition(folder, aq):
    with np.load(folder / "partition.npz") as saved:
        result = {key: saved[key].copy() for key in saved.files}
    np.testing.assert_array_equal(result["sensor_ids"], aq.sensor_ids)
    np.testing.assert_array_equal(result["sensor_node_indices"], aq.sensor_node_indices)
    masks = [np.asarray(result[key], dtype=bool) for key in
             ("train_mask", "validation_mask", "test_mask")]
    if any(mask.shape != aq.grid_shape for mask in masks):
        raise ValueError("partition grid differs from AQ")
    if any((masks[a] & masks[b]).any() for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("partitions overlap")
    return result, masks


def evaluate(folder, aq, config, masks, *, wind_cache, device, batch_size):
    started = perf_counter()
    saved = read_json(folder / "summary.json")
    checkpoint = torch.load(folder / "best-model.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != saved["best_epoch"]:
        raise ValueError("checkpoint differs from validation selection")
    if checkpoint["validation_loss"] != saved["best_validation_mae"]:
        raise ValueError("checkpoint validation loss differs from summary")
    history = saved["history"]
    best = min(history, key=lambda row: row["validation_mae"])
    if int(best["epoch"]) != saved["best_epoch"]:
        raise ValueError("saved checkpoint is not the minimum validation MAE")
    train, validation, test = masks
    np.testing.assert_array_equal(np.flatnonzero(train), config["train_nodes"])
    np.testing.assert_array_equal(np.flatnonzero(validation), config["validation_nodes"])
    actual_test_ids = aq.sensor_ids[test.reshape(-1)[aq.sensor_node_indices]].tolist()
    if sorted(actual_test_ids) != sorted(config["test_sensor_ids"]):
        raise ValueError("TEST identities differ from config")
    active = config.get("wind_mode", "none") == "directional"
    wind = load_wind_source_cache(wind_cache) if active else None
    if active and wind.metadata["fingerprint"] != config["wind_cache_fingerprint"]:
        raise ValueError("wind cache identity differs from training")
    dataset = CachedKernelDataset(
        cache=aq, wind_cache=wind, hours=int(config["hours"]), seed=int(config["seed"]),
        context_sensor_mask=train, forbidden_context_mask=validation | test,
        target_sensor_mask=validation, split_start="2023-01-01", split_end="2023-12-31 23:00",
        normalization_mean=config["normalization_mean"], normalization_std=config["normalization_std"])
    if len(dataset) != config["validation_windows"]:
        raise ValueError("validation window count differs from training")
    model = ConvexGeometricKernel(
        mode=config["mode"], wind_mode=config.get("wind_mode", "none"),
        cell_size_m=config["cell_size_m"], epsilon_m=config["epsilon_m"],
        distance_scale_m=config["distance_scale_m"],
        wind_speed_scale_m_s=config.get("wind_speed_scale_m_s", 1.),
        initial_wind_strength=config.get("initial_wind_strength", .01))
    model.prepare_geometry(aq.graph.pos, np.unique(aq.sensor_node_indices[
        train.reshape(-1)[aq.sensor_node_indices]]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    predictions = collect_sensor_predictions(
        model, DataLoader(dataset, batch_size=batch_size, shuffle=False),
        mean=np.asarray(config["normalization_mean"], dtype=np.float32),
        std=np.asarray(config["normalization_std"], dtype=np.float32), device=device,
        sensor_ids=aq.sensor_ids, sensor_node_indices=aq.sensor_node_indices,
        node_positions=aq.graph.pos.numpy(), magnitude_ids=aq.magnitude_ids)
    if (predictions.empty or predictions.duplicated(KEYS).any()
            or not predictions.window_end.dt.year.eq(2023).all()
            or not np.isfinite(predictions[["target", "prediction"]].to_numpy()).all()):
        raise ValueError("invalid or non-validation predictions")
    metrics = {key.replace("test_", "validation_", 1): value
               for key, value in summarize_all_predictions(predictions).items()}
    if not np.isfinite(metrics["validation_wape"]):
        metrics["validation_wape"] = None
    std = float(config["normalization_std"][0])
    metrics["validation_mae_normalized"] = metrics["validation_mae_physical"] / std
    metrics["validation_rmse_normalized"] = metrics["validation_rmse_physical"] / std
    np.testing.assert_allclose(metrics["validation_mae_normalized"], saved["best_validation_mae"],
                               rtol=2e-6, atol=2e-7)
    report = dict(
        run=str(folder), epochs_completed=saved["epochs_completed"],
        best_epoch_one_based=saved["best_epoch"] + 1,
        power=float(model.power.detach().cpu()), beta=float(model.wind_strength.detach().cpu()),
        training_seconds=saved["training_seconds"], evaluation_seconds=perf_counter() - started,
        wind_available_fraction=dataset.wind_coverage,
        wind_train_available_fraction=config.get("wind_train_available_fraction"), **metrics)
    return report, predictions, summarize_sensor_predictions(predictions)


@click.command(context_settings={"show_default": True})
@click.option("--without-wind-run", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--with-wind-run", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--wind-cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--batch-size", default=64, type=click.IntRange(min=1))
@click.option("--device", default="cpu")
def main(without_wind_run, with_wind_run, cache, wind_cache, output, batch_size, device):
    """Only validation 2023, TRAIN-only context, and already selected checkpoints."""
    if output.exists():
        raise click.ClickException("comparison output exists")
    started = perf_counter()
    config = read_json(without_wind_run / "config.json")
    other = read_json(with_wind_run / "config.json")
    for name in SHARED:
        if config[name] != other[name]:
            raise click.ClickException(f"pilot protocols differ: {name}")
    if (config["mode"] != "power" or config.get("wind_mode", "none") != "none"
            or other.get("wind_mode") != "directional"):
        raise click.ClickException("expected learned-power pilots without and with directional wind")
    aq = load_aq_sensor_cache(cache)
    if len(aq.magnitude_ids) != 1:
        raise click.ClickException("this comparison requires one pollutant")
    first, masks = read_partition(without_wind_run, aq)
    second, other_masks = read_partition(with_wind_run, aq)
    if set(first) != set(second):
        raise click.ClickException("partition files differ")
    for name, values in first.items():
        np.testing.assert_array_equal(values, second[name])
    paths = [folder / name for folder in (without_wind_run, with_wind_run)
             for name in ("config.json", "summary.json", "partition.npz", "best-model.pt")]
    hashes = {str(path.resolve()): file_sha256(path) for path in paths}
    results, frames, sensor_tables = [], [], []
    for label, folder, current, current_masks in (
        ("without-wind", without_wind_run, config, masks),
        ("with-wind", with_wind_run, other, other_masks),
    ):
        result, frame, sensors = evaluate(folder, aq, current, current_masks,
            wind_cache=wind_cache, device=device, batch_size=batch_size)
        result["variant"] = label
        sensors.insert(0, "variant", label)
        results.append(result)
        frames.append(frame)
        sensor_tables.append(sensors)
    a, b = (frame.sort_values(KEYS).reset_index(drop=True) for frame in frames)
    pd.testing.assert_frame_equal(a[KEYS], b[KEYS])
    np.testing.assert_array_equal(a.target.to_numpy(), b.target.to_numpy())
    reference = results[0]["validation_mae_physical"]
    for row in results:
        row["mae_change_percent"] = 100 * (row["validation_mae_physical"] / reference - 1) if reference else None
    for path in paths:
        if file_sha256(path) != hashes[str(path.resolve())]:
            raise click.ClickException("a saved training input changed during comparison")
    summary = dict(
        protocol="single-pilot-matched-wind-ablation-validation-only",
        train_start=config["train_start"], train_end=config["train_end"],
        validation_period="2023", validation_windows=config["validation_windows"],
        validation_target_count=len(a), source_policy="TRAIN AQ only",
        partition_matches=True, normalization_matches=True, keys_and_targets_match=True,
        normalization_mean=config["normalization_mean"], normalization_std=config["normalization_std"],
        wind_cache_fingerprint=other["wind_cache_fingerprint"], results=results,
        input_sha256=hashes, test_evaluated=False, retrained=False,
        elapsed_seconds=perf_counter() - started,
        note="Same code, partition, seed, dates, normalization and settings. One pilot; validation used for selection.")
    output.mkdir(parents=True)
    pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)
    pd.concat(sensor_tables, ignore_index=True).to_csv(output / "by-sensor.csv", index=False)
    for label, frame in zip(("without-wind", "with-wind"), frames):
        frame.to_csv(output / f"{label}-validation-predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    click.echo(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
