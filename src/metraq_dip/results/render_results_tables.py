#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def fmt_sci(x: float, digits: int = 2) -> str:
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def latex_overall_table(tbl: pd.DataFrame) -> str:
    mins = {c: tbl[c].min() for c in ["Mean MAE", "Median MAE", "Mean MSE", "Median MSE"]}
    maxs = {c: tbl[c].max() for c in ["MAE wins", "MSE wins"]}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Overall quantitative performance of DIP, KRG, and IDW over the 2400 paired experiments. Lower values indicate better performance. Best results are shown in bold.}",
        r"\label{tab:overall_performance}",
        r"\begin{tabular}{lcc|cc|cc}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{MAE} & \multicolumn{2}{c|}{MSE} & \multicolumn{2}{c}{Wins} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Method & Mean & Median & Mean & Median & MAE & MSE \\",
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        mae_mean = rf"\textbf{{{fmt(row['Mean MAE'])}}}" if np.isclose(row["Mean MAE"], mins["Mean MAE"]) else fmt(row["Mean MAE"])
        mae_med  = rf"\textbf{{{fmt(row['Median MAE'])}}}" if np.isclose(row["Median MAE"], mins["Median MAE"]) else fmt(row["Median MAE"])
        mse_mean = rf"\textbf{{{fmt(row['Mean MSE'])}}}" if np.isclose(row["Mean MSE"], mins["Mean MSE"]) else fmt(row["Mean MSE"])
        mse_med  = rf"\textbf{{{fmt(row['Median MSE'])}}}" if np.isclose(row["Median MSE"], mins["Median MSE"]) else fmt(row["Median MSE"])
        mae_wins = rf"\textbf{{{int(row['MAE wins'])}}}" if row["MAE wins"] == maxs["MAE wins"] else f"{int(row['MAE wins'])}"
        mse_wins = rf"\textbf{{{int(row['MSE wins'])}}}" if row["MSE wins"] == maxs["MSE wins"] else f"{int(row['MSE wins'])}"
        lines.append(
            f"{row['Method']} & {mae_mean} & {mae_med} & {mse_mean} & {mse_med} & {mae_wins} & {mse_wins} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_friedman_table(tbl: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Friedman test results and mean ranks for MAE and MSE. Lower ranks indicate better performance.}",
        r"\label{tab:friedman_results}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Metric & $\chi^2_F$ & $p$-value & DIP rank & KRG rank & IDW rank \\",
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        dip = rf"\textbf{{{fmt(row['DIP rank'])}}}" if row["DIP rank"] == tbl.loc[tbl["Metric"] == row["Metric"], ["DIP rank", "KRG rank", "IDW rank"]].min(axis=1).iloc[0] else fmt(row["DIP rank"])
        krg = rf"\textbf{{{fmt(row['KRG rank'])}}}" if row["KRG rank"] == tbl.loc[tbl["Metric"] == row["Metric"], ["DIP rank", "KRG rank", "IDW rank"]].min(axis=1).iloc[0] else fmt(row["KRG rank"])
        idw = rf"\textbf{{{fmt(row['IDW rank'])}}}" if row["IDW rank"] == tbl.loc[tbl["Metric"] == row["Metric"], ["DIP rank", "KRG rank", "IDW rank"]].min(axis=1).iloc[0] else fmt(row["IDW rank"])
        lines.append(
            f"{row['Metric']} & {row['Friedman chi2']:.2f} & {fmt_sci(row['p-value'])} & {dip} & {krg} & {idw} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_posthoc_table(tbl: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Post-hoc pairwise Wilcoxon signed-rank tests with Holm correction.}",
        r"\label{tab:wilcoxon_holm_results}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Metric & Comparison & Holm-adjusted $p$-value & Significant \\",
        r"\midrule",
    ]
    current_metric = None
    for _, row in tbl.iterrows():
        if current_metric is not None and row["Metric"] != current_metric:
            lines.append(r"\midrule")
        current_metric = row["Metric"]
        lines.append(
            f"{row['Metric']} & {row['Comparison']} & {fmt_sci(row['Holm-adjusted p-value'])} & {row['Significant']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def bold_min(a: float, b: float, c: float) -> tuple[str, str, str]:
    vals = np.array([a, b, c], dtype=float)
    min_v = vals.min()
    out = []
    for v in vals:
        text = fmt(v)
        if np.isclose(v, min_v):
            text = rf"\textbf{{{text}}}"
        out.append(text)
    return tuple(out)


def latex_group_stats_table(tbl: pd.DataFrame, loss_suffix: str, caption: str, label: str) -> str:
    prefix = {
        "L1Loss": "MAE",
        "MSELoss": "MSE",
    }[loss_suffix]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{lccc|ccc|ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c|}{DIP} & \multicolumn{3}{c|}{KRG} & \multicolumn{3}{c}{IDW} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"Group & Mean & Median & Std & Mean & Median & Std & Mean & Median & Std \\",
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        m1 = row[f"DIP_{loss_suffix} mean"]; med1 = row[f"DIP_{loss_suffix} median"]; s1 = row[f"DIP_{loss_suffix} std"]
        m2 = row[f"KRG_{loss_suffix} mean"]; med2 = row[f"KRG_{loss_suffix} median"]; s2 = row[f"KRG_{loss_suffix} std"]
        m3 = row[f"IDW_{loss_suffix} mean"]; med3 = row[f"IDW_{loss_suffix} median"]; s3 = row[f"IDW_{loss_suffix} std"]
        mean_text = bold_min(m1, m2, m3)
        med_text = bold_min(med1, med2, med3)
        std_text = bold_min(s1, s2, s3)
        lines.append(
            f"{row['Group']} & {mean_text[0]} & {med_text[0]} & {std_text[0]} & "
            f"{mean_text[1]} & {med_text[1]} & {std_text[1]} & "
            f"{mean_text[2]} & {med_text[2]} & {std_text[2]} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_temporal_narrow_table(tbl: pd.DataFrame, split_col: str, caption: str, label: str) -> str:
    first_col = "Hour" if split_col == "hour" else "Day type"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        f"{first_col} & Metric & DIP & KRG & IDW \\\\",
        r"\midrule",
    ]
    metrics = ["Mean MAE", "Median MAE", "Mean MSE", "Median MSE"]
    for idx, (_, row) in enumerate(tbl.iterrows()):
        group_value = row[split_col]
        first = True
        lines.append(rf"\multirow{{4}}{{*}}{{{group_value}}}")
        for metric in metrics:
            a, b, c = row[f"DIP {metric}"], row[f"KRG {metric}"], row[f"IDW {metric}"]
            ta, tb, tc = bold_min(a, b, c)
            prefix = " & " if first else ""
            lines.append(f"& {metric} & {ta} & {tb} & {tc} \\\\")
            first = False
        if idx < len(tbl) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_hardest_cases_table(tbl: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Characteristics of the top 5\% worst DIP cases under MAE and MSE. Percentages indicate the fraction of cases in the corresponding tail.}",
        r"\label{tab:hardest_cases}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Metric & Threshold & Cases & 08{:}00 (\%) & Weekend (\%) & Mean data std & DIP wins (\%) \\",
        r"\midrule",
    ]
    for _, row in tbl.iterrows():
        lines.append(
            f"{row['Metric']} & {fmt(row['Threshold'], 3)} & {int(row['Cases'])} & "
            f"{fmt(row['08:00 (%)'], 1)} & {fmt(row['Weekend (%)'], 1)} & "
            f"{fmt(row['Mean data std'], 3)} & {fmt(row['DIP wins (%)'], 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, help="Directory created by compute_results_data.py")
    parser.add_argument("--outdir", default="latex_tables", help="Directory for .tex table files")
    args = parser.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(datadir / "overall_performance.csv")
    friedman = pd.read_csv(datadir / "friedman_results.csv")
    posthoc = pd.read_csv(datadir / "wilcoxon_holm_results.csv")
    group_mae = pd.read_csv(datadir / "group_mae_stats.csv")
    group_mse = pd.read_csv(datadir / "group_mse_stats.csv")
    temporal_hour = pd.read_csv(datadir / "temporal_hour.csv")
    temporal_day = pd.read_csv(datadir / "temporal_daytype.csv")
    hardest = pd.read_csv(datadir / "hardest_cases.csv")

    write_text(outdir / "overall_performance.tex", latex_overall_table(overall))
    write_text(outdir / "friedman_results.tex", latex_friedman_table(friedman))
    write_text(outdir / "wilcoxon_holm_results.tex", latex_posthoc_table(posthoc))
    write_text(
        outdir / "group_mae_stats.tex",
        latex_group_stats_table(
            group_mae,
            "L1Loss",
            "Group-wise MAE statistics for each interpolation method. Lower values indicate better performance. Best result for each statistic within a group is shown in bold.",
            "tab:group_mae_stats",
        ),
    )
    write_text(
        outdir / "group_mse_stats.tex",
        latex_group_stats_table(
            group_mse,
            "MSELoss",
            "Group-wise MSE statistics for each interpolation method. Lower values indicate better performance. Best result for each statistic within a group is shown in bold.",
            "tab:group_mse_stats",
        ),
    )
    write_text(
        outdir / "temporal_hour_narrow.tex",
        latex_temporal_narrow_table(
            temporal_hour,
            "hour",
            "Performance across the two sampled start hours. Lower values indicate better performance. Best result in each row is shown in bold.",
            "tab:temporal_hour",
        ),
    )
    write_text(
        outdir / "temporal_daytype_narrow.tex",
        latex_temporal_narrow_table(
            temporal_day,
            "day_type",
            "Performance across weekday and weekend windows. Lower values indicate better performance. Best result in each row is shown in bold.",
            "tab:temporal_daytype",
        ),
    )
    write_text(outdir / "hardest_cases.tex", latex_hardest_cases_table(hardest))

    print(f"Wrote LaTeX tables to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
