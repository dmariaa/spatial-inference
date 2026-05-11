from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import yaml

from metraq_dip.experiments import _denormalize_masked_channels


EXPERIMENTS = (
    "airparif_no2_add24h",
    "airparif_no2_addons",
    "airparif_nox_add24h",
    "airparif_nox_addons",
    "metraq_no2_add24h",
    "metraq_no2_addons",
    "metraq_nox_add24h",
    "metraq_nox_addons",
)
METHODS = ("DIP", "KRG", "IDW")


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    folder: Path
    dataset: str
    pollutant: str
    setup: str
    pollutants: list[int]
    normalize: bool

POLLUTANT_META: dict[str, dict[str, object]] = {
    "airparif_no2_add24h": {"pollutants": [1], "normalize": True},
    "airparif_no2_addons": {"pollutants": [1], "normalize": True},
    "airparif_nox_add24h": {"pollutants": [2], "normalize": True},
    "airparif_nox_addons": {"pollutants": [2], "normalize": True},
    "metraq_no2_add24h": {"pollutants": [8], "normalize": True},
    "metraq_no2_addons": {"pollutants": [8], "normalize": True},
    "metraq_nox_add24h": {"pollutants": [12], "normalize": True},
    "metraq_nox_addons": {"pollutants": [12], "normalize": True},
}


def _exp_file(folder: Path, sensor_group: str, time_window: str) -> Path:
    ts = pd.Timestamp(time_window)
    return folder / f"exp_{sensor_group}_{ts.strftime('%Y%m%dT%H%M%S')}.npz"


def _target_mean_abs(
    experiment_npz: np.lib.npyio.NpzFile,
    *,
    pollutants: list[int],
    normalize: bool,
) -> float:
    test_data = np.asarray(experiment_npz["test_data"][0], dtype=np.float32)
    test_mask = np.asarray(experiment_npz["test_mask"][0], dtype=bool)
    if normalize:
        y_data = _denormalize_masked_channels(
            test_data,
            test_mask,
            pollutants=pollutants,
            normalization_stats=experiment_npz["normalization_stats"].item(),
        )
    else:
        y_data = test_data

    target = np.asarray(y_data[:, -1, ...], dtype=np.float32)
    mask_last = np.asarray(test_mask[:, 0, ...], dtype=bool)
    values = np.abs(target[np.broadcast_to(mask_last, target.shape)])
    if values.size == 0:
        raise ValueError("Encountered an empty test mask while computing WAPE.")
    return float(values.mean())


def _fmt_float(value: float, *, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _fmt_pvalue(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.2e}"


def _fmt_pvalue_paper(value: float) -> str:
    if np.isnan(value):
        return "p=nan"
    if value < 0.001:
        return "p<.001"
    return f"p={value:.3f}".replace("p=0.", "p=.")


def _holm_adjust(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _load_results(specs: list[ExperimentSpec]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    wape_frames: dict[str, pd.DataFrame] = {}

    for spec in specs:
        folder = spec.folder
        results_file = folder / "results.csv"
        if not results_file.exists():
            raise FileNotFoundError(f"Missing results file: {results_file}")

        df = pd.read_csv(results_file)
        frames[spec.key] = df

        rows: list[dict[str, float]] = []
        for _, row in df.iterrows():
            exp_path = _exp_file(folder, str(row["sensor_group"]), str(row["time_window"]))
            with np.load(exp_path, allow_pickle=True) as data:
                denom = _target_mean_abs(
                    data,
                    pollutants=spec.pollutants,
                    normalize=spec.normalize,
                )
            rows.append(
                {
                    "DIP_WAPE": float(row["DIP_L1Loss"]) / denom,
                    "KRG_WAPE": float(row["KRG_L1Loss"]) / denom,
                    "IDW_WAPE": float(row["IDW_L1Loss"]) / denom,
                }
            )
        wape_frames[spec.key] = pd.DataFrame(rows)

    return frames, wape_frames


def build_metrics_table(
    specs: list[ExperimentSpec],
    frames: dict[str, pd.DataFrame],
    wape_frames: dict[str, pd.DataFrame],
) -> str:
    headers = [
        "Experiment",
        "DIP L1",
        "KRG L1",
        "IDW L1",
        "DIP MSE",
        "KRG MSE",
        "IDW MSE",
        "DIP WAPE",
        "KRG WAPE",
        "IDW WAPE",
    ]
    rows: list[list[str]] = []
    for spec in specs:
        df = frames[spec.key]
        wdf = wape_frames[spec.key]
        rows.append(
            [
                spec.key,
                _fmt_float(df["DIP_L1Loss"].mean()),
                _fmt_float(df["KRG_L1Loss"].mean()),
                _fmt_float(df["IDW_L1Loss"].mean()),
                _fmt_float(df["DIP_MSELoss"].mean()),
                _fmt_float(df["KRG_MSELoss"].mean()),
                _fmt_float(df["IDW_MSELoss"].mean()),
                _fmt_float(wdf["DIP_WAPE"].mean()),
                _fmt_float(wdf["KRG_WAPE"].mean()),
                _fmt_float(wdf["IDW_WAPE"].mean()),
            ]
        )
    return _markdown_table(headers, rows)


def build_dip_vs_krg_table(
    specs: list[ExperimentSpec],
    frames: dict[str, pd.DataFrame],
    wape_frames: dict[str, pd.DataFrame],
) -> str:
    headers = [
        "Experiment",
        "L1 wins D/K",
        "L1 p",
        "MSE wins D/K",
        "MSE p",
        "WAPE wins D/K",
        "WAPE p",
    ]
    rows: list[list[str]] = []

    for spec in specs:
        df = frames[spec.key]
        wdf = wape_frames[spec.key]
        triplets = [
            ("L1", df["DIP_L1Loss"], df["KRG_L1Loss"]),
            ("MSE", df["DIP_MSELoss"], df["KRG_MSELoss"]),
            ("WAPE", wdf["DIP_WAPE"], wdf["KRG_WAPE"]),
        ]

        values: list[str] = [spec.key]
        for _, dip_values, krg_values in triplets:
            diff = dip_values - krg_values
            dip_win = (dip_values < krg_values).mean() * 100
            krg_win = (krg_values < dip_values).mean() * 100
            values.append(f"{dip_win:.1f} / {krg_win:.1f}")
            values.append(_fmt_pvalue(wilcoxon(diff).pvalue))
        rows.append(values)

    return _markdown_table(headers, rows)


def _experiment_parts(experiment: str) -> tuple[str, str, str]:
    dataset, pollutant, setup = experiment.split("_", 2)
    return dataset.upper(), pollutant.upper(), setup


def _clean_setup_name(folder_name: str) -> str:
    setup = folder_name
    if setup.startswith("experiment_"):
        setup = setup[len("experiment_") :]
    if setup.endswith("_newnorm"):
        setup = setup[: -len("_newnorm")]
    return setup


def _load_config(folder: Path) -> dict[str, object]:
    config_file = folder / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config file: {config_file}")
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return dict(config or {})


def _spec_from_config(
    folder: Path,
    *,
    dataset: str,
    pollutant: str,
    setup: str | None = None,
) -> ExperimentSpec:
    config = _load_config(folder)
    pollutants = [int(value) for value in config["pollutants"]]
    return ExperimentSpec(
        key=folder.name,
        folder=folder,
        dataset=dataset.upper(),
        pollutant=pollutant.upper(),
        setup=setup or _clean_setup_name(folder.name),
        pollutants=pollutants,
        normalize=bool(config.get("normalize", False)),
    )


def _default_specs(root: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for experiment in EXPERIMENTS:
        dataset, pollutant, setup = _experiment_parts(experiment)
        meta = POLLUTANT_META[experiment]
        specs.append(
            ExperimentSpec(
                key=experiment,
                folder=root / experiment,
                dataset=dataset,
                pollutant=pollutant,
                setup=setup,
                pollutants=list(meta["pollutants"]),
                normalize=bool(meta["normalize"]),
            )
        )
    return specs


def _resolve_specs(
    *,
    root: Path,
    experiment_folders: tuple[Path, ...] = (),
    dataset: str = "METRAQ",
    pollutant: str = "NO",
) -> list[ExperimentSpec]:
    if experiment_folders:
        return [
            _spec_from_config(folder.resolve(), dataset=dataset, pollutant=pollutant)
            for folder in experiment_folders
        ]
    return _default_specs(root)


def _metric_triplet(values: tuple[float, float, float]) -> str:
    return " / ".join(_fmt_float(value, digits=3) for value in values)


def _win_rate_and_pvalue(dip_values: pd.Series, krg_values: pd.Series) -> tuple[str, str]:
    diff = dip_values - krg_values
    dip_win = (dip_values < krg_values).mean() * 100
    krg_win = (krg_values < dip_values).mean() * 100
    return f"{dip_win:.1f} / {krg_win:.1f}", _fmt_pvalue(wilcoxon(diff).pvalue)


def _metric_columns(metric: str) -> tuple[str, str, str]:
    if metric == "MAE":
        return "DIP_L1Loss", "KRG_L1Loss", "IDW_L1Loss"
    if metric == "MSE":
        return "DIP_MSELoss", "KRG_MSELoss", "IDW_MSELoss"
    if metric == "WAPE":
        return "DIP_WAPE", "KRG_WAPE", "IDW_WAPE"
    raise ValueError(f"Unknown metric: {metric}")


def _metric_frame(
    *,
    metric: str,
    df: pd.DataFrame,
    wdf: pd.DataFrame,
) -> pd.DataFrame:
    source = wdf if metric == "WAPE" else df
    columns = _metric_columns(metric)
    return pd.DataFrame({method: source[column].astype(float) for method, column in zip(METHODS, columns)})


def _friedman_pvalue(metric_values: pd.DataFrame) -> float:
    _, pvalue = friedmanchisquare(
        metric_values["DIP"].astype(float),
        metric_values["KRG"].astype(float),
        metric_values["IDW"].astype(float),
    )
    return float(pvalue)


def _mean_ranks(metric_values: pd.DataFrame) -> dict[str, float]:
    ranks = metric_values.astype(float).rank(axis=1, method="average", ascending=True)
    return {
        method: float(ranks[method].mean())
        for method in METHODS
    }


def _format_metric_means(values: pd.Series, *, digits: int = 3) -> dict[str, str]:
    best = float(values.min())
    formatted: dict[str, str] = {}
    for method, value in values.items():
        text = _fmt_float(float(value), digits=digits)
        if np.isclose(float(value), best):
            text = f"**{text}**"
        formatted[str(method)] = text
    return formatted


def _winner_rate_values(metric_values: pd.DataFrame) -> dict[str, float]:
    row_min = metric_values.min(axis=1)
    winners = metric_values.eq(row_min, axis=0)
    return {
        method: winners[method].mean() * 100
        for method in METHODS
    }


def _format_win_rates(rates: dict[str, float]) -> dict[str, str]:
    highest = max(rates.values())
    return {
        method: (
            f"**{rates[method]:.1f}**"
            if np.isclose(rates[method], highest)
            else f"{rates[method]:.1f}"
        )
        for method in METHODS
    }


def _format_mean_ranks(ranks: dict[str, float], *, digits: int = 3) -> dict[str, str]:
    best = min(ranks.values())
    return {
        method: (
            f"**{_fmt_float(ranks[method], digits=digits)}**"
            if np.isclose(ranks[method], best)
            else _fmt_float(ranks[method], digits=digits)
        )
        for method in METHODS
    }


def _directional_wilcoxon_result(metric_values: pd.DataFrame, left: str, right: str) -> tuple[str, float]:
    left_values = metric_values[left].astype(float)
    right_values = metric_values[right].astype(float)
    left_mean = float(left_values.mean())
    right_mean = float(right_values.mean())
    if np.isclose(left_mean, right_mean):
        direction = "tie"
    else:
        direction = left if left_mean < right_mean else right

    pvalue = float(wilcoxon(left_values - right_values).pvalue)
    return direction, pvalue


def _pairwise_posthoc(metric_values: pd.DataFrame) -> dict[str, dict[str, object]]:
    pairs = [("DIP", "KRG"), ("DIP", "IDW"), ("KRG", "IDW")]
    labels: list[str] = []
    directions: list[str] = []
    raw_pvalues: list[float] = []

    for left, right in pairs:
        direction, raw_pvalue = _directional_wilcoxon_result(metric_values, left, right)
        labels.append(f"{left} vs {right}")
        directions.append(direction)
        raw_pvalues.append(raw_pvalue)

    holm_pvalues = _holm_adjust(raw_pvalues)
    return {
        label: {
            "direction": direction,
            "raw_pvalue": raw_pvalue,
            "holm_pvalue": holm_pvalue,
        }
        for label, direction, raw_pvalue, holm_pvalue in zip(labels, directions, raw_pvalues, holm_pvalues)
    }


def _format_directional_posthoc(direction: str, holm_pvalue: float) -> str:
    _ = direction
    if holm_pvalue >= 0.05:
        return "n.s."
    return _fmt_pvalue_paper(holm_pvalue)


def build_paper_performance_data(
    specs: list[ExperimentSpec],
    frames: dict[str, pd.DataFrame],
    wape_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for spec in specs:
        df = frames[spec.key]
        wdf = wape_frames[spec.key]
        for metric in ("MAE", "MSE", "WAPE"):
            metric_values = _metric_frame(metric=metric, df=df, wdf=wdf)
            friedman_pvalue = _friedman_pvalue(metric_values)
            mean_ranks = _mean_ranks(metric_values)
            posthoc = _pairwise_posthoc(metric_values)
            means = metric_values.mean()
            win_rates = _winner_rate_values(metric_values)

            rows.append(
                {
                    "Dataset": spec.dataset,
                    "Pollutant": spec.pollutant,
                    "Setup": spec.setup,
                    "Metric": metric,
                    "DIP mean": float(means["DIP"]),
                    "KRG mean": float(means["KRG"]),
                    "IDW mean": float(means["IDW"]),
                    "DIP win pct": float(win_rates["DIP"]),
                    "KRG win pct": float(win_rates["KRG"]),
                    "IDW win pct": float(win_rates["IDW"]),
                    "DIP mean rank": float(mean_ranks["DIP"]),
                    "KRG mean rank": float(mean_ranks["KRG"]),
                    "IDW mean rank": float(mean_ranks["IDW"]),
                    "Friedman pvalue": friedman_pvalue,
                    "DIP vs KRG direction": str(posthoc["DIP vs KRG"]["direction"]),
                    "DIP vs KRG pvalue": float(posthoc["DIP vs KRG"]["raw_pvalue"]),
                    "DIP vs KRG holm pvalue": float(posthoc["DIP vs KRG"]["holm_pvalue"]),
                    "DIP vs IDW direction": str(posthoc["DIP vs IDW"]["direction"]),
                    "DIP vs IDW pvalue": float(posthoc["DIP vs IDW"]["raw_pvalue"]),
                    "DIP vs IDW holm pvalue": float(posthoc["DIP vs IDW"]["holm_pvalue"]),
                    "KRG vs IDW direction": str(posthoc["KRG vs IDW"]["direction"]),
                    "KRG vs IDW pvalue": float(posthoc["KRG vs IDW"]["raw_pvalue"]),
                    "KRG vs IDW holm pvalue": float(posthoc["KRG vs IDW"]["holm_pvalue"]),
                }
            )

    return pd.DataFrame(rows)


def render_paper_performance_table(summary: pd.DataFrame) -> str:
    headers = [
        "Dataset",
        "Pollutant",
        "Setup",
        "Metric",
        "DIP",
        "KRG",
        "IDW",
        "DIP <br> win %",
        "KRG <br> win %",
        "IDW <br> win %",
        "DIP <br> rank",
        "KRG <br> rank",
        "IDW <br> rank",
        "Friedman p",
        "DIP vs KRG",
        "DIP vs IDW",
        "KRG vs IDW",
    ]
    rows: list[list[str]] = []

    for _, row in summary.iterrows():
        means = _format_metric_means(
            pd.Series(
                {
                    "DIP": row["DIP mean"],
                    "KRG": row["KRG mean"],
                    "IDW": row["IDW mean"],
                }
            )
        )
        win_rates = _format_win_rates(
            {
                "DIP": float(row["DIP win pct"]),
                "KRG": float(row["KRG win pct"]),
                "IDW": float(row["IDW win pct"]),
            }
        )
        mean_ranks = _format_mean_ranks(
            {
                "DIP": float(row["DIP mean rank"]),
                "KRG": float(row["KRG mean rank"]),
                "IDW": float(row["IDW mean rank"]),
            }
        )
        rows.append(
            [
                str(row["Dataset"]),
                str(row["Pollutant"]),
                str(row["Setup"]),
                str(row["Metric"]),
                means["DIP"],
                means["KRG"],
                means["IDW"],
                win_rates["DIP"],
                win_rates["KRG"],
                win_rates["IDW"],
                mean_ranks["DIP"],
                mean_ranks["KRG"],
                mean_ranks["IDW"],
                _fmt_pvalue_paper(float(row["Friedman pvalue"])),
                _format_directional_posthoc(
                    str(row["DIP vs KRG direction"]),
                    float(row["DIP vs KRG holm pvalue"]),
                ),
                _format_directional_posthoc(
                    str(row["DIP vs IDW direction"]),
                    float(row["DIP vs IDW holm pvalue"]),
                ),
                _format_directional_posthoc(
                    str(row["KRG vs IDW direction"]),
                    float(row["KRG vs IDW holm pvalue"]),
                ),
            ]
        )

    note = (
        "\n\n"
        "Note: lower is better. Bold marks the lowest mean error in each row. "
        "Win % is the fraction of windows where the method has the lowest error among DIP, KRG, and IDW; "
        "ties count for each tied method. Mean ranks are Friedman within-window ranks averaged across windows; "
        "lower ranks indicate better overall ordering.\n\n"
        "Friedman p-values are from the omnibus repeated-measures test across DIP, KRG, and IDW.\n\n"
        "Pairwise entries report post-hoc p-values from paired two-sided Wilcoxon signed-rank tests over windows "
        "with Holm adjustment; n.s. denotes p >= .05."
    )
    return "## Experiment Summary\n\n" + _markdown_table(headers, rows) + note


def build_compact_comparison_table(
    specs: list[ExperimentSpec],
    frames: dict[str, pd.DataFrame],
    wape_frames: dict[str, pd.DataFrame],
) -> str:
    headers = [
        "Dataset",
        "Pollutant",
        "Setup",
        "MAE D/K/I",
        "MAE win D/K %",
        "MAE p",
        "MSE D/K/I",
        "MSE win D/K %",
        "MSE p",
        "WAPE D/K/I",
        "WAPE win D/K %",
        "WAPE p",
    ]
    rows: list[list[str]] = []

    for spec in specs:
        df = frames[spec.key]
        wdf = wape_frames[spec.key]

        mae_win, mae_p = _win_rate_and_pvalue(df["DIP_L1Loss"], df["KRG_L1Loss"])
        mse_win, mse_p = _win_rate_and_pvalue(df["DIP_MSELoss"], df["KRG_MSELoss"])
        wape_win, wape_p = _win_rate_and_pvalue(wdf["DIP_WAPE"], wdf["KRG_WAPE"])

        rows.append(
            [
                spec.dataset,
                spec.pollutant,
                spec.setup,
                _metric_triplet(
                    (
                        df["DIP_L1Loss"].mean(),
                        df["KRG_L1Loss"].mean(),
                        df["IDW_L1Loss"].mean(),
                    )
                ),
                mae_win,
                mae_p,
                _metric_triplet(
                    (
                        df["DIP_MSELoss"].mean(),
                        df["KRG_MSELoss"].mean(),
                        df["IDW_MSELoss"].mean(),
                    )
                ),
                mse_win,
                mse_p,
                _metric_triplet(
                    (
                        wdf["DIP_WAPE"].mean(),
                        wdf["KRG_WAPE"].mean(),
                        wdf["IDW_WAPE"].mean(),
                    )
                ),
                wape_win,
                wape_p,
            ]
        )

    return _markdown_table(headers, rows)


def build_add24h_addons_table(
    frames: dict[str, pd.DataFrame],
    wape_frames: dict[str, pd.DataFrame],
) -> str:
    headers = [
        "Pair",
        "DIP L1 delta",
        "DIP L1 p",
        "DIP MSE delta",
        "DIP MSE p",
        "DIP WAPE delta",
        "DIP WAPE p",
    ]
    rows: list[list[str]] = []

    for prefix in ("airparif_no2", "airparif_nox", "metraq_no2", "metraq_nox"):
        add24h = frames[f"{prefix}_add24h"].sort_values(["time_window", "sensor_group"]).reset_index(drop=True)
        addons = frames[f"{prefix}_addons"].sort_values(["time_window", "sensor_group"]).reset_index(drop=True)
        add24h_wape = wape_frames[f"{prefix}_add24h"]
        addons_wape = wape_frames[f"{prefix}_addons"]

        l1_diff = addons["DIP_L1Loss"] - add24h["DIP_L1Loss"]
        mse_diff = addons["DIP_MSELoss"] - add24h["DIP_MSELoss"]
        wape_diff = addons_wape["DIP_WAPE"] - add24h_wape["DIP_WAPE"]

        rows.append(
            [
                prefix,
                _fmt_float(l1_diff.mean()),
                _fmt_pvalue(wilcoxon(l1_diff).pvalue),
                _fmt_float(mse_diff.mean()),
                _fmt_pvalue(wilcoxon(mse_diff).pvalue),
                _fmt_float(wape_diff.mean()),
                _fmt_pvalue(wilcoxon(wape_diff).pvalue),
            ]
        )

    return _markdown_table(headers, rows)


def main(
    *,
    root: Path = Path("output/experiments/cross_dataset_no2_nox"),
    outdir: Path | None = None,
    experiment_folders: tuple[Path, ...] = (),
    dataset: str = "METRAQ",
    pollutant: str = "NO",
) -> None:
    root = root.resolve()
    outdir = (outdir or root).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    specs = _resolve_specs(
        root=root,
        experiment_folders=experiment_folders,
        dataset=dataset,
        pollutant=pollutant,
    )
    frames, wape_frames = _load_results(specs)

    metrics_table = build_metrics_table(specs, frames, wape_frames)
    dip_vs_krg_table = build_dip_vs_krg_table(specs, frames, wape_frames)
    compact_comparison_table = build_compact_comparison_table(specs, frames, wape_frames)
    paper_performance_data = build_paper_performance_data(specs, frames, wape_frames)
    paper_performance_table = render_paper_performance_table(paper_performance_data)

    (outdir / "summary_metrics.md").write_text(metrics_table + "\n", encoding="utf-8")
    (outdir / "summary_dip_vs_krg.md").write_text(dip_vs_krg_table + "\n", encoding="utf-8")
    (outdir / "summary_compact_comparison.md").write_text(
        compact_comparison_table + "\n",
        encoding="utf-8",
    )
    paper_performance_data.to_csv(outdir / "summary_paper_performance_data.csv", index=False)
    (outdir / "summary_paper_performance.md").write_text(
        paper_performance_table + "\n",
        encoding="utf-8",
    )

    print("Wrote:")
    print(outdir / "summary_metrics.md")
    print(outdir / "summary_dip_vs_krg.md")
    print(outdir / "summary_compact_comparison.md")
    print(outdir / "summary_paper_performance_data.csv")
    print(outdir / "summary_paper_performance.md")
    if not experiment_folders:
        add24h_addons_table = build_add24h_addons_table(frames, wape_frames)
        (outdir / "summary_add24h_vs_addons.md").write_text(
            add24h_addons_table + "\n",
            encoding="utf-8",
        )
        print(outdir / "summary_add24h_vs_addons.md")


if __name__ == "__main__":
    import click

    @click.command(context_settings={"show_default": True})
    @click.option(
        "--root",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=Path("output/experiments/cross_dataset_no2_nox"),
        help="Experiment bundle root folder.",
    )
    @click.option(
        "--outdir",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=None,
        help="Output directory for the generated markdown tables. Defaults to ROOT.",
    )
    @click.option(
        "--experiment-folder",
        "experiment_folders",
        multiple=True,
        type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
        help="Experiment folder to summarize. Repeat to compare multiple folders. When omitted, ROOT uses the cross-dataset layout.",
    )
    @click.option(
        "--dataset",
        default="METRAQ",
        help="Dataset label used with --experiment-folder.",
    )
    @click.option(
        "--pollutant",
        default="NO",
        help="Pollutant label used with --experiment-folder.",
    )
    def cli(
        root: Path,
        outdir: Path | None,
        experiment_folders: tuple[Path, ...],
        dataset: str,
        pollutant: str,
    ) -> None:
        main(
            root=root,
            outdir=outdir,
            experiment_folders=experiment_folders,
            dataset=dataset,
            pollutant=pollutant,
        )

    cli()
