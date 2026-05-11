from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ADD24_EXPERIMENTS = (
    "airparif_no2_add24h",
    "airparif_nox_add24h",
    "metraq_no2_add24h",
    "metraq_nox_add24h",
)
ADDONS_EXPERIMENTS = (
    "airparif_no2_addons",
    "airparif_nox_addons",
    "metraq_no2_addons",
    "metraq_nox_addons",
)
EXPERIMENT_LABELS = {
    "airparif_no2_add24h": "AIRPARIF_NO2",
    "airparif_nox_add24h": "AIRPARIF_NOX",
    "metraq_no2_add24h": "METRAQ_NO2",
    "metraq_nox_add24h": "METRAQ_NOX",
    "airparif_no2_addons": "AIRPARIF_NO2_ADDONS",
    "airparif_nox_addons": "AIRPARIF_NOX_ADDONS",
    "metraq_no2_addons": "METRAQ_NO2_ADDONS",
    "metraq_nox_addons": "METRAQ_NOX_ADDONS",
}


def _clean_setup_name(folder_name: str) -> str:
    setup = folder_name
    if setup.startswith("experiment_"):
        setup = setup[len("experiment_") :]
    if setup.endswith("_newnorm"):
        setup = setup[: -len("_newnorm")]
    return setup


def _generic_experiment_label(folder: Path, *, dataset: str, pollutant: str) -> str:
    setup = _clean_setup_name(folder.name).upper()
    return f"{dataset.upper()}_{pollutant.upper()}_{setup}"


def _load_config(root: Path) -> dict[str, Any]:
    with (root / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _experiment_file(root: Path, sensor_group: str, time_window: str | pd.Timestamp) -> Path:
    ts = pd.Timestamp(time_window)
    return root / f"exp_{sensor_group}_{ts.strftime('%Y%m%dT%H%M%S')}.npz"


def _denormalized_test_values(exp_path: Path, *, config: dict[str, Any]) -> np.ndarray:
    pollutant = int(config["pollutants"][0])
    with np.load(exp_path, allow_pickle=True) as experiment:
        test_data = np.asarray(experiment["test_data"][0, 0, -1], dtype=np.float32)
        test_mask = np.asarray(experiment["test_mask"][0, 0, -1], dtype=bool)
        values = test_data[test_mask]

        if bool(config.get("normalize", False)):
            normalization_stats = experiment["normalization_stats"].item()
            mean, std = normalization_stats[pollutant]
            values = values * (std + 1e-6) + mean

    if values.size == 0:
        raise ValueError(f"Empty test mask in {exp_path}")
    return np.asarray(values, dtype=np.float32)


def _row_metrics_for_experiment(root: Path, *, label: str) -> pd.DataFrame:
    config = _load_config(root)
    results = pd.read_csv(root / "results.csv", dtype={"sensor_group": "string"}, parse_dates=["time_window"])
    if "processed" in results:
        results = results[results["processed"].astype(str).str.lower().isin(["true", "1", "yes"])]

    rows: list[dict[str, Any]] = []
    for _, result in results.iterrows():
        exp_path = _experiment_file(root, str(result["sensor_group"]), result["time_window"])
        values = _denormalized_test_values(exp_path, config=config)
        rows.append(
            {
                "pollutant": label,
                "root": str(root),
                "time_window": result["time_window"],
                "sensor_group": result["sensor_group"],
                "DIP_L1": float(result["DIP_L1Loss"]),
                "KRG_L1": float(result["KRG_L1Loss"]),
                "IDW_L1": float(result["IDW_L1Loss"]),
                "DIP_MSE": float(result["DIP_MSELoss"]),
                "KRG_MSE": float(result["KRG_MSELoss"]),
                "IDW_MSE": float(result["IDW_MSELoss"]),
                "target_mean_abs": float(np.abs(values).mean()),
                "target_max": float(np.max(values)),
                "target_std": float(np.std(values)),
                "target_p90_p10": float(np.percentile(values, 90) - np.percentile(values, 10)),
                "gap_l1_dip_minus_krg": float(result["DIP_L1Loss"] - result["KRG_L1Loss"]),
                "gap_mse_dip_minus_krg": float(result["DIP_MSELoss"] - result["KRG_MSELoss"]),
            }
        )
    return pd.DataFrame(rows)


def _exemplary_windows(row_metrics: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pollutant, pollutant_rows in row_metrics.groupby("pollutant"):
        window_metrics = (
            pollutant_rows.groupby("time_window", as_index=False)
            .agg(
                target_mean_abs=("target_mean_abs", "mean"),
                target_std=("target_std", "mean"),
                target_p90_p10=("target_p90_p10", "mean"),
                DIP_L1=("DIP_L1", "mean"),
                KRG_L1=("KRG_L1", "mean"),
                IDW_L1=("IDW_L1", "mean"),
                gap_l1_dip_minus_krg=("gap_l1_dip_minus_krg", "mean"),
                DIP_MSE=("DIP_MSE", "mean"),
                KRG_MSE=("KRG_MSE", "mean"),
                gap_mse_dip_minus_krg=("gap_mse_dip_minus_krg", "mean"),
            )
        )
        for case_type, ascending in (("best_l1_vs_krg", True), ("worst_l1_vs_krg", False)):
            for _, row in window_metrics.sort_values("gap_l1_dip_minus_krg", ascending=ascending).head(top_n).iterrows():
                rows.append({"pollutant": pollutant, "case_type": case_type, **row.to_dict()})
    return pd.DataFrame(rows)


def _write_summary(row_metrics: pd.DataFrame, outdir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for pollutant, df in row_metrics.groupby("pollutant"):
        rows.append(
            {
                "Experiment": pollutant,
                "DIP L1": df["DIP_L1"].mean(),
                "KRG L1": df["KRG_L1"].mean(),
                "IDW L1": df["IDW_L1"].mean(),
                "DIP<KRG L1 %": (df["DIP_L1"] < df["KRG_L1"]).mean() * 100,
                "DIP MSE": df["DIP_MSE"].mean(),
                "KRG MSE": df["KRG_MSE"].mean(),
                "IDW MSE": df["IDW_MSE"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    headers = list(summary.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for _, row in summary.iterrows():
        values = [str(row["Experiment"])]
        values.extend(f"{float(row[column]):.3f}" for column in headers[1:])
        lines.append("| " + " | ".join(values) + " |")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    *,
    root: Path = Path("output/experiments/cross_dataset_no2_nox_autoencoder"),
    outdir: Path | None = None,
    include_addons: bool = False,
    top_n: int = 5,
    experiment_folders: tuple[Path, ...] = (),
    dataset: str = "METRAQ",
    pollutant: str = "NO",
) -> None:
    root = root.resolve()
    outdir = (outdir or root / "window_diagnostics").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    if experiment_folders:
        for experiment_root in experiment_folders:
            experiment_root = experiment_root.resolve()
            frames.append(
                _row_metrics_for_experiment(
                    experiment_root,
                    label=_generic_experiment_label(experiment_root, dataset=dataset, pollutant=pollutant),
                )
            )
    else:
        experiments = list(ADD24_EXPERIMENTS)
        if include_addons:
            experiments.extend(ADDONS_EXPERIMENTS)

        for experiment in experiments:
            experiment_root = root / experiment
            if not experiment_root.exists():
                raise FileNotFoundError(experiment_root)
            frames.append(_row_metrics_for_experiment(experiment_root, label=EXPERIMENT_LABELS[experiment]))

    row_metrics = pd.concat(frames, ignore_index=True)
    row_metrics.to_csv(outdir / "row_level_metrics.csv", index=False)
    _exemplary_windows(row_metrics, top_n=top_n).to_csv(outdir / "exemplary_windows.csv", index=False)
    _write_summary(row_metrics, outdir)

    print(f"Wrote diagnostics to {outdir}")


if __name__ == "__main__":
    import click

    @click.command(context_settings={"show_default": True})
    @click.option(
        "--root",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=Path("output/experiments/cross_dataset_no2_nox_autoencoder"),
    )
    @click.option(
        "--outdir",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=None,
    )
    @click.option("--include-addons", is_flag=True)
    @click.option("--top-n", type=int, default=5)
    @click.option(
        "--experiment-folder",
        "experiment_folders",
        multiple=True,
        type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
        help="Experiment folder to summarize. Repeat to compare multiple folders. When omitted, ROOT uses the cross-dataset layout.",
    )
    @click.option("--dataset", default="METRAQ", help="Dataset label used with --experiment-folder.")
    @click.option("--pollutant", default="NO", help="Pollutant label used with --experiment-folder.")
    def cli(
        root: Path,
        outdir: Path | None,
        include_addons: bool,
        top_n: int,
        experiment_folders: tuple[Path, ...],
        dataset: str,
        pollutant: str,
    ) -> None:
        main(
            root=root,
            outdir=outdir,
            include_addons=include_addons,
            top_n=top_n,
            experiment_folders=experiment_folders,
            dataset=dataset,
            pollutant=pollutant,
        )

    cli()
