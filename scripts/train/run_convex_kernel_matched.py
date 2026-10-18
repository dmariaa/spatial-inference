"""Train ten wind kernels and extend the historical frozen TEST comparison."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import click
import numpy as np
import pandas as pd

from metraq_gnn.comparison import append_matched_candidate, load_matched
from metraq_gnn.data import load_aq_sensor_cache, split_sensor_nodes
from metraq_gnn.data.splits import compute_training_normalization
from metraq_gnn.data.wind_cache import load_wind_source_cache


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_campaign(gnn_root, reference, index, cache, wind_cache):
    """Reject mismatched groups, dates, cache or normalization before training."""
    folders = sorted(gnn_root.glob("group-*"))
    if [p.name for p in folders] != [f"group-{i:02d}" for i in range(1, 11)]:
        raise click.ClickException("expected the ten original GNN group directories")
    baseline = load_matched(gnn_root, reference)
    dip = baseline[baseline.method == "DIP"]
    if len(dip) != 9600 or dip.sensor_group.nunique() != 10:
        raise click.ClickException("expected 9600 historical targets across ten groups")
    with np.load(index, allow_pickle=True) as saved:
        times = pd.DatetimeIndex(pd.to_datetime(saved["time_windows"].tolist()))
        index_groups = {tuple(sorted(map(int, row))) for row in saved["test_sensors"]}
    if len(times) != 240 or times.has_duplicates or not (times.year == 2024).all():
        raise click.ClickException("expected 240 unique TEST timestamps in 2024")
    aq = load_aq_sensor_cache(cache)
    wind = load_wind_source_cache(wind_cache)
    if wind.metadata.get("mapping") != "station":
        raise click.ClickException("expected the direct mapped wind cache")
    plans, inputs = [], [reference, index]
    seen = set()
    for folder in folders:
        path = folder / "summary.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        if (pd.Timestamp(config["train_start"]) != pd.Timestamp("2015-01-01")
                or config["hours"] != 24 or config.get("stride", 1) != 1
                or config["seed"] != 42 or config["train_nodes"] != 16
                or config["validation_nodes"] != 4 or config["test_nodes"] != 4
                or config["test_windows"] != 240):
            raise click.ClickException(f"unexpected original protocol: {folder.name}")
        ids = tuple(sorted(config["test_sensor_ids"]))
        if ids in seen or ids not in index_groups:
            raise click.ClickException("duplicate or unknown TEST sensor group")
        seen.add(ids)
        group = "-".join(map(str, ids))
        rows = dip[dip.sensor_group == group]
        if len(rows) != 960 or set(rows.sensor_id) != set(ids):
            raise click.ClickException("historical group targets differ")
        if set(rows.window_end) != set(times) or not rows.groupby("window_end").size().eq(4).all():
            raise click.ClickException("historical group timestamps differ")
        masks = split_sensor_nodes(aq, validation_nodes=4, test_nodes=4,
                                   seed=config["seed"], test_sensor_ids=list(ids))
        mean, std = compute_training_normalization(
            aq, train_node_mask=masks[0], start="2015-01-01", end="2022-12-31 23:00")
        np.testing.assert_allclose(mean, config["normalization_mean"], rtol=1e-6)
        np.testing.assert_allclose(std, config["normalization_std"], rtol=1e-6)
        plans.append(dict(group=folder.name, sensor_group=group, reference=str(path),
                          seed=config["seed"], normalization_mean=mean.tolist(),
                          normalization_std=std.tolist()))
        inputs.extend([path, folder / "predictions.csv"])
    if seen != index_groups:
        raise click.ClickException("original index and GNN groups differ")
    return baseline, plans, {str(p.resolve()): file_hash(p) for p in inputs}, wind.metadata["fingerprint"]


@click.command(context_settings={"show_default": True})
@click.option("--gnn-root", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--reference", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--test-windows-index", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--wind-cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--device", default="cuda")
@click.option("--epochs", default=100, type=click.IntRange(min=1))
@click.option("--batch-size", default=64, type=click.IntRange(min=1))
@click.option("--check-only", is_flag=True, help="Validate inputs without training or writing output.")
def main(gnn_root, reference, test_windows_index, cache, wind_cache, output,
         device, epochs, batch_size, check_only):
    if output.exists():
        raise click.ClickException("campaign output already exists")
    started = perf_counter()
    baseline, plans, hashes, fingerprint = prepare_campaign(
        gnn_root, reference, test_windows_index, cache, wind_cache)
    manifest = dict(protocol="ten-group-matched-wind-kernel", train_period="2015-2022",
                    validation_period="2023", test_period="2024", groups=plans,
                    test_windows_per_group=240, test_targets_total=9600,
                    epochs=epochs, batch_size=batch_size, patience=20,
                    mode="power", wind_mode="directional", initial_beta=.01,
                    wind_cache_fingerprint=fingerprint, input_sha256=hashes,
                    wandb_mode="offline", test_used_for_selection=False)
    if check_only:
        click.echo(json.dumps(manifest, indent=2))
        return
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, WANDB_MODE="offline", PYTHONUNBUFFERED="1")
    def run(stage, group, command):
        click.echo(json.dumps(dict(stage=stage, group=group, event="started",
                                   elapsed_seconds=perf_counter()-started)))
        log = output / f"{group}.{stage}.log"
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, *map(str, command)],
                                    cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        (output / f"{group}.{stage}.exit").write_text(str(result.returncode)+"\n")
        click.echo(json.dumps(dict(stage=stage, group=group, event="finished",
                                   exit=result.returncode, elapsed_seconds=perf_counter()-started)))
        if result.returncode:
            raise click.ClickException(f"{group} {stage} failed; see {log}")
    for plan in plans:
        name = plan["group"]
        folder = output / name
        run("train", name, ["scripts/train/train_convex_kernel.py", "--cache", cache,
            "--wind-cache", wind_cache, "--output", folder, "--mode", "power",
            "--wind-mode", "directional", "--initial-wind-strength", ".01",
            "--partition-summary", plan["reference"], "--seed", plan["seed"],
            "--train-start", "2015-01-01", "--train-end", "2022-12-31 23:00",
            "--epochs", epochs, "--batch-size", batch_size, "--device", device,
            "--wandb", "--wandb-mode", "offline",
            "--wandb-name", f"convex-power-wind-matched-{output.name}-{name}"])
        run("test", name, ["scripts/evaluation/evaluate_cached_model.py",
            "--model", "convex-kernel", "--run-dir", folder, "--cache", cache,
            "--wind-cache", wind_cache, "--reference-summary", plan["reference"],
            "--test-windows-index", test_windows_index, "--output", folder / "test",
            "--batch-size", batch_size, "--device", device])
        # Fail early if a group's TEST does not match, without changing any hyperparameters.
        candidate = pd.read_csv(folder / "test/predictions.csv")
        append_matched_candidate(baseline[baseline.sensor_group == plan["sensor_group"]],
                                 candidate, "CONVEX_KERNEL_POWER_WIND_DIRECTIONAL")
        for path, digest in hashes.items():
            if file_hash(Path(path)) != digest:
                raise click.ClickException("an original benchmark input changed")
    run("comparison", "all", ["scripts/evaluation/compare_gnn_matched_predictions.py",
        "--gnn-root", gnn_root, "--reference", reference, "--kernel-root", output,
        "--output-dir", output / "comparison"])
    (output / "summary.json").write_text(json.dumps(dict(
        completed_groups=10, test_targets_per_method=9600, matched_cases=2400,
        elapsed_seconds=perf_counter()-started, input_hashes_unchanged=True,
        test_used_for_selection=False, wandb_mode="offline"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
