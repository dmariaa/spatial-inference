"""Frozen matched AQ-station removal: kernel, existing Kriging and existing IDW."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data import load_aq_sensor_cache
from metraq_gnn.data.kernel_window import CachedKernelDataset
from metraq_gnn.data.wind_cache import load_wind_source_cache
from metraq_gnn.evaluation import collect_sensor_predictions
from metraq_gnn.model import ConvexGeometricKernel
from metraq_gnn.sparsity import (
    KEYS, LEVELS, RestrictedContext, canonicalize, classical_predictions,
    subset_plans, summarize_sparse,
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_metrics(frame):
    cases = frame.groupby(["group", "repeat", "n_sources", "window_end", "method"])["absolute_error"].mean()
    pivot = cases.unstack("method")
    rows = []
    for comparator in ("KRG", "IDW"):
        d = (pivot.KERNEL_WIND - pivot[comparator]).rename("mae_delta").reset_index()
        d["comparator"] = comparator
        rows.append(d)
    paired = pd.concat(rows, ignore_index=True)
    summaries = []
    for (count, comparator), group in paired.groupby(["n_sources", "comparator"]):
        delta = group.mae_delta
        summaries.append(dict(n_sources=int(count), comparator=comparator, cases=len(delta),
            kernel_wins=int((delta < 0).sum()), ties=int((delta == 0).sum()),
            kernel_losses=int((delta > 0).sum()), mean_mae_delta=float(delta.mean()),
            median_mae_delta=float(delta.median())))
    return paired, pd.DataFrame(summaries)


@click.command(context_settings={"show_default": True})
@click.option("--runs-root", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--wind-cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--reference", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--seed", default=20261019, type=click.IntRange(min=0))
@click.option("--repeats", default=5, type=click.IntRange(min=1))
@click.option("--group", "selected_groups", multiple=True, type=click.IntRange(min=1, max=10),
              help="Limit groups only for a local smoke check; default is all ten.")
@click.option("--max-windows", type=click.IntRange(min=1, max=240), help="Smoke check only.")
@click.option("--batch-size", default=64, type=click.IntRange(min=1))
@click.option("--device", default="cpu")
def main(runs_root, cache, wind_cache, reference, output, seed, repeats,
         selected_groups, max_windows, batch_size, device):
    started = perf_counter()
    if output.exists():
        raise click.ClickException("output already exists")
    selected_groups = sorted(set(selected_groups)) if selected_groups else list(range(1, 11))
    aq, wind = load_aq_sensor_cache(cache), load_wind_source_cache(wind_cache)
    if list(aq.magnitude_ids) != [8] or len(np.unique(aq.sensor_node_indices)) != len(aq.sensor_ids):
        raise click.ClickException("expected NO2 and one station per AQ cell")
    ref = pd.read_csv(reference, parse_dates=["window_end"])
    canonical = ref[ref.method == "DIP"]
    if canonical.duplicated(KEYS).any() or not canonical.window_end.dt.year.eq(2024).all():
        raise click.ClickException("invalid historical targets")
    inputs = {str(reference.resolve()): sha(reference),
              str((cache/"metadata.json").resolve()): sha(cache/"metadata.json"),
              str((wind_cache/"metadata.json").resolve()): sha(wind_cache/"metadata.json")}
    plans = []
    for index in selected_groups:
        name = f"group-{index:02d}"
        folder = runs_root/name
        c, training, previous = (read(folder/f) for f in ("config.json", "summary.json", "test/summary.json"))
        if c["mode"] != "power" or c["wind_mode"] != "directional":
            raise click.ClickException("expected a learned-power directional wind checkpoint")
        if c["wind_cache_fingerprint"] != wind.metadata["fingerprint"]:
            raise click.ClickException("wind cache differs from training")
        for f in ("config.json", "summary.json", "partition.npz", "best-model.pt", "test/predictions.csv"):
            inputs[str((folder/f).resolve())] = sha(folder/f)
        if previous["checkpoint_sha256"] != sha(folder/"best-model.pt"):
            raise click.ClickException("checkpoint differs from the original TEST evaluation")
        with np.load(folder/"partition.npz") as z:
            np.testing.assert_array_equal(z["sensor_ids"], aq.sensor_ids)
            np.testing.assert_array_equal(z["sensor_node_indices"], aq.sensor_node_indices)
            train, val, test = (z[k].copy() for k in ("train_mask", "validation_mask", "test_mask"))
        if (train & val).any() or ((train | val) & test).any() or tuple(map(np.sum, (train, val, test))) != (16, 4, 4):
            raise click.ClickException("unexpected original partition")
        np.testing.assert_array_equal(np.flatnonzero(train), c["train_nodes"])
        np.testing.assert_array_equal(np.flatnonzero(val), c["validation_nodes"])
        group = "-".join(map(str, sorted(c["test_sensor_ids"])))
        truth = canonical[canonical.sensor_group == group].copy()
        times = sorted(truth.window_end.unique())
        if len(truth) != 960 or len(times) != 240 or not truth.groupby("window_end").size().eq(4).all():
            raise click.ClickException("expected all 240 windows and 960 targets per group")
        times = times[:max_windows] if max_windows else times
        truth = truth[truth.window_end.isin(times)]
        subsets = subset_plans(train | val, seed=seed, group_index=index, repeats=repeats)
        for plan in subsets:
            plan["sensor_ids"] = aq.sensor_ids[np.isin(aq.sensor_node_indices, plan["nodes"])].astype(int).tolist()
        plans.append(dict(group=name, sensor_group=group, config=c, train=train, val=val,
            test=test, truth=truth, times=times, subsets=subsets, best_epoch=training["best_epoch"]))
    manifest = dict(protocol="frozen-AQ-source-removal", seed=seed, repeats=repeats, levels=LEVELS,
        groups=[dict(group=p["group"], sensor_group=p["sensor_group"], subsets=p["subsets"]) for p in plans],
        windows_per_group=len(plans[0]["times"]), exploratory=True, retrained=False,
        weather_policy="existing cached wind; only retained AQ context sources use wind",
        normalization_policy="unchanged TRAIN statistics from each selected checkpoint",
        baseline_policy="20 sources evaluated once; five nested removals at each smaller level",
        selection_policy="geometry-only; masks fixed before errors; no failed targets dropped",
        smoke_check=bool(max_windows or len(selected_groups) != 10),
        wind_cache_fingerprint=wind.metadata["fingerprint"], input_sha256=inputs)
    output.mkdir(parents=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    case_counts = []
    timings = []
    dense_checks = []
    completed = 0
    total = sum(len(p["subsets"]) for p in plans)
    for plan in plans:
        name, c, truth = plan["group"], plan["config"], plan["truth"]
        destination = output/name
        destination.mkdir()
        dataset = CachedKernelDataset(cache=aq, wind_cache=wind, hours=c["hours"], seed=c["seed"],
            context_sensor_mask=plan["train"] | plan["val"], forbidden_context_mask=plan["test"],
            target_sensor_mask=plan["test"], split_start="2024-01-01", split_end="2024-12-31 23:00",
            normalization_mean=c["normalization_mean"], normalization_std=c["normalization_std"])
        available = {pd.Timestamp(t): i for i, t in enumerate(dataset.window_end_times)}
        samples = [dataset[available[pd.Timestamp(t)]] for t in plan["times"]]
        for subset in plan["subsets"]:
            counts = [int(s.context_mask[subset["nodes"], -1, 0].sum()) for s in samples]
            if min(counts) < 2:
                raise click.ClickException("a planned case has fewer than two sources; preserve the plan and investigate")
        model = ConvexGeometricKernel(mode=c["mode"], wind_mode=c["wind_mode"], cell_size_m=c["cell_size_m"],
            epsilon_m=c["epsilon_m"], distance_scale_m=c["distance_scale_m"],
            wind_speed_scale_m_s=c["wind_speed_scale_m_s"], initial_wind_strength=c["initial_wind_strength"])
        model.prepare_geometry(aq.graph.pos, np.flatnonzero(plan["train"] | plan["val"]))
        cp = torch.load(runs_root/name/"best-model.pt", map_location="cpu", weights_only=True)
        if cp["epoch"] != plan["best_epoch"]:
            raise click.ClickException("checkpoint is not the original validation selection")
        model.load_state_dict(cp["model_state_dict"], strict=True)
        model.to(device).eval().requires_grad_(False)
        for scenario_index, subset in enumerate(plan["subsets"]):
            stage_start = perf_counter()
            status = dict(group=name, source_count=subset["source_count"], repeat=subset["repeat"],
                          completed_scenarios=completed, total_scenarios=total, elapsed_seconds=perf_counter()-started)
            (output/"status.json").write_text(json.dumps(status), encoding="utf-8")
            click.echo(json.dumps(dict(status, event="started")))
            restricted = RestrictedContext(samples, subset["nodes"])
            kernel_start = perf_counter()
            predicted = collect_sensor_predictions(model, DataLoader(restricted, batch_size=batch_size, shuffle=False),
                mean=np.asarray(c["normalization_mean"], np.float32), std=np.asarray(c["normalization_std"], np.float32),
                device=device, sensor_ids=aq.sensor_ids, sensor_node_indices=aq.sensor_node_indices,
                node_positions=aq.graph.pos.numpy(), magnitude_ids=aq.magnitude_ids)
            kernel_seconds = perf_counter()-kernel_start
            predicted["sensor_group"] = plan["sensor_group"]
            predicted["method"] = "KERNEL_WIND"
            kernel = canonicalize(predicted, truth)
            if subset["source_count"] == LEVELS[0]:
                old = pd.read_csv(runs_root/name/"test/predictions.csv", parse_dates=["window_end"])
                old = old[old.window_end.isin(plan["times"])]
                paired = kernel.merge(old[KEYS+["prediction"]], on=KEYS, validate="one_to_one", suffixes=("", "_old"))
                np.testing.assert_allclose(paired.prediction, paired.prediction_old, rtol=5e-7, atol=2e-5)
                dense_checks.append(dict(group=name, max_prediction_delta=float((paired.prediction-paired.prediction_old).abs().max())))
            classic_start = perf_counter()
            rows = []
            counts = {}
            node_to_sensor = dict(zip(aq.sensor_node_indices, aq.sensor_ids))
            for sample in restricted:
                nodes, estimates, actual_count = classical_predictions(
                    sample, aq.grid_shape, c["normalization_mean"], c["normalization_std"])
                timestamp = pd.Timestamp(int(sample.window_end_ns.item()))
                counts[timestamp] = actual_count
                case_counts.append(dict(group=name, repeat=subset["repeat"], n_sources=subset["source_count"],
                                        window_end=timestamp, actual_sources=actual_count, targets=len(nodes)))
                for method, values in estimates.items():
                    for node, estimate in zip(nodes, values, strict=True):
                        actual = float(sample.y[node, 0])*c["normalization_std"][0]+c["normalization_mean"][0]
                        rows.append(dict(sensor_group=plan["sensor_group"], window_end=timestamp,
                            sensor_id=int(node_to_sensor[node]), method=method, target=actual, prediction=float(estimate)))
            classical_seconds = perf_counter()-classic_start
            parts = [kernel]+[canonicalize(pd.DataFrame(rows).query("method == @m"), truth) for m in ("KRG", "IDW")]
            frames = []
            for frame in parts:
                frame = frame[KEYS+["method", "target", "prediction", "error", "absolute_error", "squared_error"]].copy()
                frame["group"] = name
                frame["n_sources"] = subset["source_count"]
                frame["repeat"] = subset["repeat"]
                frame["actual_sources"] = frame.window_end.map(counts)
                if len(frame) != len(truth) or frame.actual_sources.isna().any():
                    raise click.ClickException("a method lost original targets")
                frames.append(frame)
            scored = pd.concat(frames, ignore_index=True)
            scored.to_csv(destination/"predictions.csv", index=False, mode="x" if scenario_index == 0 else "a", header=scenario_index == 0)
            completed += 1
            timings.append(dict(group=name, n_sources=subset["source_count"], repeat=subset["repeat"],
                kernel_seconds=kernel_seconds, classical_seconds=classical_seconds, elapsed_seconds=perf_counter()-stage_start))
            click.echo(json.dumps(dict(status, event="finished", completed_scenarios=completed,
                                       scenario_seconds=timings[-1]["elapsed_seconds"])))
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise click.ClickException("an input or frozen checkpoint changed")
    full = pd.concat((pd.read_csv(output/p["group"]/"predictions.csv", parse_dates=["window_end"]) for p in plans), ignore_index=True)
    if full.duplicated(["group", "repeat", "n_sources", "window_end", "sensor_id", "method"]).any():
        raise click.ClickException("duplicate scenario target keys")
    summarize_sparse(full, ["n_sources", "method"]).to_csv(output/"metrics_by_level.csv", index=False)
    summarize_sparse(full, ["group", "n_sources", "method"]).to_csv(output/"metrics_by_group.csv", index=False)
    summarize_sparse(full, ["repeat", "n_sources", "method"]).to_csv(output/"metrics_by_repeat.csv", index=False)
    summarize_sparse(full, ["group", "repeat", "n_sources", "method"]).to_csv(output/"metrics_by_scenario.csv", index=False)
    cases, paired = paired_metrics(full)
    cases.to_csv(output/"paired_by_case.csv", index=False)
    paired.to_csv(output/"paired_by_level.csv", index=False)
    pd.DataFrame(case_counts).to_csv(output/"case_source_counts.csv", index=False)
    pd.DataFrame(timings).to_csv(output/"timings.csv", index=False)
    summary = dict(completed_groups=len(plans), completed_scenarios=completed,
        targets_per_method=int(len(full)//3), prediction_rows=len(full), dense_checkpoint_checks=dense_checks,
        levels=LEVELS, repeats=repeats, seed=seed, retrained=False, inputs_unchanged=True,
        exploratory=True, failed_cases=0, all_targets_retained=True, elapsed_seconds=perf_counter()-started)
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output/"status.json").write_text(json.dumps(dict(summary, phase="complete")), encoding="utf-8")
    click.echo(json.dumps(summary))


if __name__ == "__main__":
    main()
