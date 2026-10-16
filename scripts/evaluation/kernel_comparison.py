"""Matched pilot ablation: fixed IDW, learned exponent and neural correction."""
import hashlib
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

KEYS = ["window_end", "sensor_id", "magnitude_id"]


def align_predictions(reference, candidate):
    for frame in (reference, candidate):
        if frame.duplicated(KEYS).any():
            raise ValueError("duplicate prediction keys")
    aligned = reference.merge(candidate, on=KEYS, how="outer",
                              suffixes=("_ref", "_candidate"), validate="one_to_one", indicator=True)
    if not (aligned["_merge"] == "both").all():
        raise ValueError("prediction keys differ")
    if not np.allclose(aligned["target_ref"], aligned["target_candidate"], rtol=0, atol=1e-5):
        raise ValueError("targets differ")
    return aligned


@click.command()
@click.option("--fixed-run", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--fixed-test", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--power-run", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--power-test", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--neural-run", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--neural-test", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def main(fixed_run, fixed_test, power_run, power_test, neural_run, neural_test, output):
    if output.exists():
        raise click.ClickException("comparison output already exists")
    def read(path):
        return json.loads(path.read_text())
    base_config = read(neural_run / "config.json")
    base_predictions = pd.read_csv(neural_test / "predictions.csv", parse_dates=["window_end"])
    rows, per_sensor, per_window = [], [], []
    for mode, run, evaluation in (
        ("fixed", fixed_run, fixed_test), ("power", power_run, power_test),
        ("neural", neural_run, neural_test),
    ):
        config, training = read(run/"config.json"), read(run/"summary.json")
        test = read(evaluation/"summary.json")
        if config["mode"] != mode:
            raise ValueError("unexpected model mode")
        for name in ("hours", "train_period", "validation_period", "seed",
                     "cell_size_m", "train_nodes", "validation_nodes", "test_sensor_ids",
                     "normalization_mean", "normalization_std"):
            if config[name] != base_config[name]:
                raise ValueError(f"configuration differs: {name}")
        for name, default in (("epsilon_m", 1.), ("distance_scale_m", 1000.),
                              ("source_policy", "all-current-visible")):
            if config.get(name, default) != base_config.get(name, default):
                raise ValueError(f"kernel geometry differs: {name}")
        with np.load(neural_run/"partition.npz") as reference, np.load(run/"partition.npz") as candidate:
            for key in reference.files:
                np.testing.assert_array_equal(reference[key], candidate[key])
        digest = hashlib.sha256((run/"best-model.pt").read_bytes()).hexdigest()
        if test["checkpoint_sha256"] != digest or test["checkpoint_epoch"] != training["best_epoch"]:
            raise ValueError("evaluation checkpoint differs from validation selection")
        predictions = pd.read_csv(evaluation/"predictions.csv", parse_dates=["window_end"])
        aligned = align_predictions(base_predictions, predictions)
        # Use the same canonical targets across all three methods.
        error = aligned["prediction_candidate"].to_numpy()-aligned["target_ref"].to_numpy()
        target = aligned["target_ref"].to_numpy()
        if not np.isfinite(error).all() or not len(error):
            raise ValueError("invalid or empty predictions")
        metrics = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).mean())),
                       wape=float(np.abs(error).sum()/np.abs(target).sum()))
        for key, stored in (("mae","test_mae_physical"),("rmse","test_rmse_physical"),("wape","test_wape")):
            np.testing.assert_allclose(metrics[key], test[stored], atol=1e-6, rtol=1e-6)
        rows.append(dict(mode=mode, power=training["power"],
                         best_epoch=training["best_epoch"]+1 if mode != "fixed" else None,
                         epochs_completed=training["epochs_completed"],
                         validation_mae=training["best_validation_mae"],
                         validation_mae_physical=training["best_validation_mae_physical"],
                         test_count=len(error), test_windows=predictions.window_end.nunique(), **metrics))
        scored = aligned[KEYS].copy()
        scored["mode"], scored["absolute_error"], scored["squared_error"] = mode, np.abs(error), error**2
        per_window.append(scored.groupby(["mode","window_end"], as_index=False).agg(
            mae=("absolute_error","mean"), mse=("squared_error","mean")))
        sensor = pd.read_csv(evaluation/"by-sensor.csv")
        sensor.insert(0,"mode",mode)
        per_sensor.append(sensor)
    frame = pd.DataFrame(rows)
    fixed_mae = float(frame.loc[frame["mode"]=="fixed","mae"].iloc[0])
    frame["mae_change_vs_fixed_percent"] = 100*(frame["mae"]/fixed_mae-1)
    output.mkdir(parents=True)
    frame.to_csv(output/"comparison.csv", index=False)
    pd.concat(per_window).to_csv(output/"by-window.csv", index=False)
    pd.concat(per_sensor).to_csv(output/"by-sensor.csv", index=False)
    summary = dict(protocol="single-pilot-matched-ablation", window_count=len(base_predictions.window_end.unique()),
                   target_count=len(base_predictions), results=rows,
                   note="Same sources, geometry, seed, masks and normalization. Selection uses validation only; this is not the ten-group benchmark.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    click.echo(frame.to_string(index=False))


if __name__ == "__main__":
    main()
