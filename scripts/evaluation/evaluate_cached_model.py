from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data import (
    SensorGraphWindowDataset,
    load_aq_sensor_cache,
    load_traffic_grid_cache,
    split_sensor_nodes,
)
from metraq_gnn.data.kernel_window import CachedKernelDataset
from metraq_gnn.data.wind_cache import load_wind_source_cache
from metraq_gnn.evaluation import (
    collect_sensor_predictions, summarize_all_predictions, summarize_sensor_predictions,
)
from metraq_gnn.model import ConvexGeometricKernel, SensorToGridGNN, SpatioTemporalGNN


def selected_test_windows(dataset, *, index_path: Path):
    with np.load(index_path, allow_pickle=True) as index:
        if "time_windows" not in index.files:
            raise ValueError("test windows index must contain 'time_windows'")
        requested = pd.DatetimeIndex(pd.to_datetime(index["time_windows"].tolist()))
    if requested.has_duplicates:
        raise ValueError("test windows index must not contain duplicate timestamps")
    available = {
        int(timestamp.value): position
        for position, timestamp in enumerate(dataset.window_end_times)
    }
    missing = [timestamp for timestamp in requested if int(timestamp.value) not in available]
    if missing:
        raise ValueError(f"{len(missing)} requested test timestamps are unavailable")
    return torch.utils.data.Subset(
        dataset,
        [available[int(timestamp.value)] for timestamp in requested],
    )


def load_partition(run_dir, aq, config):
    """Read saved masks or reconstruct legacy GNN masks, then validate both alike."""
    partition_path = run_dir / "partition.npz"
    if partition_path.exists():
        with np.load(partition_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["sensor_ids"], aq.sensor_ids)
            np.testing.assert_array_equal(saved["sensor_node_indices"], aq.sensor_node_indices)
            masks = tuple(np.asarray(saved[name], dtype=bool) for name in
                          ("train_mask", "validation_mask", "test_mask"))
    else:
        masks = split_sensor_nodes(
            aq, validation_nodes=int(config["validation_nodes"]),
            test_nodes=int(config["test_nodes"]), seed=int(config["seed"]),
            test_sensor_ids=config.get("test_sensor_ids"))
    train, validation, test = masks
    if any(mask.shape != aq.grid_shape for mask in masks):
        raise click.ClickException("partition shape differs from cache grid")
    if (train & validation).any() or (train & test).any() or (validation & test).any():
        raise click.ClickException("saved partitions overlap")
    for name, mask in (("train_nodes", train), ("validation_nodes", validation)):
        if isinstance(config.get(name), list):
            np.testing.assert_array_equal(np.flatnonzero(mask), config[name])
    test_ids = aq.sensor_ids[test.reshape(-1)[aq.sensor_node_indices]].tolist()
    if config.get("test_sensor_ids") is not None and sorted(test_ids) != sorted(config["test_sensor_ids"]):
        raise click.ClickException("TEST sensor identities differ from training config")
    return masks, test_ids


def prepare_dataset(model_kind, aq, config, masks, traffic_cache, wind_cache=None):
    """Prepare architecture-specific features with the same evaluation masks."""
    train, validation, test = masks
    options = dict(
        cache=aq, hours=int(config["hours"]), seed=int(config["seed"]),
        normalization_mean=np.asarray(config["normalization_mean"], dtype=np.float32),
        normalization_std=np.asarray(config["normalization_std"], dtype=np.float32),
        context_sensor_mask=train | validation, forbidden_context_mask=test,
        target_sensor_mask=test, split_start="2024-01-01", split_end="2024-12-31 23:00")
    if model_kind == "convex-kernel":
        if traffic_cache is not None:
            raise click.ClickException("convex kernel has no traffic features; omit --traffic-cache")
        active = config.get("wind_mode", "none") == "directional"
        path = wind_cache or config.get("wind_cache")
        if not active and path is not None:
            raise click.ClickException("this kernel has no wind features; omit --wind-cache")
        wind = None
        if active:
            if path is None or not Path(path).is_dir():
                raise click.ClickException("this kernel requires its wind cache; provide --wind-cache")
            wind = load_wind_source_cache(path)
            if wind.metadata["fingerprint"] != config.get("wind_cache_fingerprint"):
                raise click.ClickException("wind cache identity differs from training")
        return CachedKernelDataset(**options, wind_cache=wind)
    if wind_cache is not None:
        raise click.ClickException("GNN wind features are not implemented; omit --wind-cache")
    if traffic_cache is None and config.get("traffic_cache"):
        raise click.ClickException("this GNN was trained with traffic; provide --traffic-cache")
    return SensorGraphWindowDataset(
        **options, traffic_cache=load_traffic_grid_cache(traffic_cache) if traffic_cache else None,
        traffic_normalization_mean=config.get("traffic_normalization_mean"),
        traffic_normalization_std=config.get("traffic_normalization_std"),
        include_time_features=bool(config.get("time_features", False)))


def build_model(model_kind, aq, config, masks, sample):
    """Only architecture construction differs; checkpoint loading is shared."""
    if model_kind == "convex-kernel":
        model = ConvexGeometricKernel(
            mode=config["mode"], cell_size_m=config["cell_size_m"],
            epsilon_m=config.get("epsilon_m", 1.), distance_scale_m=config.get("distance_scale_m", 1000.),
            wind_mode=config.get("wind_mode", "none"),
            wind_speed_scale_m_s=config.get("wind_speed_scale_m_s", 1.),
            initial_wind_strength=config.get("initial_wind_strength", .01))
        context = masks[0] | masks[1]
        model.prepare_geometry(aq.graph.pos, np.unique(aq.sensor_node_indices[
            context.reshape(-1)[aq.sensor_node_indices]]))
        return model
    options = dict(input_channels=int(sample.x.shape[-1]),
                   output_channels=len(aq.magnitude_ids),
                   edge_channels=int(aq.graph.edge_attr.shape[-1]),
                   temporal_hidden_channels=32, graph_hidden_channels=32,
                   attention_heads=4, dropout=0.1)
    if config["architecture"] == "sensor-to-grid":
        return SensorToGridGNN(**options, local_refinement_layers=int(config["local_refinement_layers"]))
    if config["architecture"] == "local":
        return SpatioTemporalGNN(**options, graph_layers=2)
    raise click.ClickException(f"unknown architecture: {config['architecture']}")


def validate_reference(reference, config, aq, masks, test_windows_index):
    expected = split_sensor_nodes(
        aq, validation_nodes=reference["validation_nodes"], test_nodes=reference["test_nodes"],
        seed=reference["seed"], test_sensor_ids=reference.get("test_sensor_ids"))
    for actual, other in zip(masks, expected, strict=True):
        np.testing.assert_array_equal(actual, other)
    for name in ("normalization_mean", "normalization_std"):
        np.testing.assert_allclose(config[name], reference[name], rtol=1e-6)
    reference_index = reference.get("test_windows_index")
    if reference_index and (test_windows_index is None or Path(reference_index).resolve() != test_windows_index.resolve()):
        raise click.ClickException("reference uses a different selected-window index")
    if reference.get("stride", 1) != 1:
        raise click.ClickException("reference stride is not supported")


@click.command(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option("--model", "model_kind", type=click.Choice(["auto", "gnn", "convex-kernel"]), default="auto", help="Auto detects kernel runs from config.json.")
@click.option("--reference-summary", type=click.Path(exists=True, path_type=Path), help="Optional reference partition, normalization and TEST counts.")
@click.option("--run-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--cache", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--traffic-cache", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--wind-cache", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--test-windows-index", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--output", type=click.Path(path_type=Path))
@click.option("--batch-size", type=click.IntRange(min=1), default=64)
@click.option("--num-workers", type=click.IntRange(min=0), default=0)
@click.option("--pin-memory/--no-pin-memory", default=None)
@click.option("--device")
def main(run_dir, cache, traffic_cache, test_windows_index, output, batch_size,
         num_workers, pin_memory, device, model_kind, reference_summary, wind_cache):
    """Evaluate a frozen GNN or kernel on all or selected 2024 windows.

    Both models write predictions.csv, by-sensor.csv and summary.json in a new
    output directory. An explicit .csv output preserves the legacy GNN path;
    its summaries use <stem>.summary.json and <stem>.by-sensor.csv alongside it.
    """
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config_path = run_dir / "config.json"
    config = {**training, **(json.loads(config_path.read_text()) if config_path.exists() else {})}
    if model_kind == "auto":
        model_kind = "convex-kernel" if config.get("mode") in {"fixed", "power", "neural"} else "gnn"
    destination = output or run_dir / "test-evaluation"
    csv_output = destination.suffix.lower() == ".csv"
    predictions_path = destination if csv_output else destination / "predictions.csv"
    summary_path = destination.with_suffix(".summary.json") if csv_output else destination / "summary.json"
    sensor_path = destination.with_suffix(".by-sensor.csv") if csv_output else destination / "by-sensor.csv"
    if (not csv_output and destination.exists()) or any(path.exists() for path in (predictions_path, summary_path, sensor_path)):
        raise click.ClickException("output already exists; choose a new destination")
    checkpoint_path = run_dir / "best-model.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != training["best_epoch"]:
        raise click.ClickException("checkpoint is not the validation-selected epoch")
    aq = load_aq_sensor_cache(cache)
    masks, test_ids = load_partition(run_dir, aq, config)
    reference = json.loads(reference_summary.read_text()) if reference_summary else None
    if reference is not None:
        validate_reference(reference, config, aq, masks, test_windows_index)
    dataset = prepare_dataset(model_kind, aq, config, masks, traffic_cache, wind_cache)
    test_data = selected_test_windows(dataset, index_path=test_windows_index) if test_windows_index else dataset
    if not len(test_data):
        raise click.ClickException("no TEST windows selected")
    if reference is not None and len(test_data) != reference["test_windows"]:
        raise click.ClickException("TEST window count differs from reference")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_kind, aq, config, masks, test_data[0])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=resolved_device.type == "cuda" if pin_memory is None else pin_memory,
                        persistent_workers=num_workers > 0)
    click.echo(f"Evaluating {len(test_data)} TEST windows; checkpoint epoch {checkpoint['epoch']+1}.")
    predictions = collect_sensor_predictions(
        model, loader, mean=np.asarray(config["normalization_mean"], dtype=np.float32),
        std=np.asarray(config["normalization_std"], dtype=np.float32), device=resolved_device,
        sensor_ids=aq.sensor_ids, sensor_node_indices=aq.sensor_node_indices,
        node_positions=aq.graph.pos.numpy(), magnitude_ids=aq.magnitude_ids)
    if predictions.empty or predictions.duplicated(["window_end", "sensor_id", "magnitude_id"]).any():
        raise click.ClickException("empty or duplicate TEST predictions")
    if not np.isfinite(predictions[["target", "prediction"]].to_numpy()).all():
        raise click.ClickException("nonfinite TEST targets/predictions")
    if reference is not None and len(predictions) != reference["test_target_count"]:
        raise click.ClickException("TEST target count differs from reference")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise click.ClickException("checkpoint changed during evaluation")
    method = (f"CONVEX_KERNEL_{config['mode'].upper()}" if model_kind == "convex-kernel"
              else "GNN_TRAFFIC" if traffic_cache else "GNN")
    if model_kind == "convex-kernel" and config.get("wind_mode", "none") == "directional":
        method += "_WIND_DIRECTIONAL"
    predictions.insert(0, "sensor_group", "-".join(str(value) for value in sorted(test_ids)))
    predictions.insert(4, "method", method)
    metrics = summarize_all_predictions(predictions)
    summary = dict(**metrics, model=model_kind, method=method, test_windows=len(test_data),
                   test_sensor_ids=test_ids, checkpoint_epoch=checkpoint["epoch"],
                   checkpoint_sha256=checkpoint_hash, context="TRAIN+VALIDATION", period="2024",
                   test_evaluated=True, test_windows_index=str(test_windows_index) if test_windows_index else None,
                   reference_summary=str(reference_summary) if reference_summary else None)
    if model_kind == "convex-kernel":
        summary.update(wind_mode=config.get("wind_mode", "none"),
                       wind_cache_fingerprint=config.get("wind_cache_fingerprint"),
                       wind_available_fraction=dataset.wind_coverage,
                       wind_strength=float(model.wind_strength.detach().cpu()))
    if reference is not None:
        summary["reference_metrics"] = {key: reference[key] for key in
            ("test_mae_physical", "test_rmse_physical", "test_wape")}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    summarize_sensor_predictions(predictions).to_csv(sensor_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    click.echo(json.dumps({"output": str(predictions_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
