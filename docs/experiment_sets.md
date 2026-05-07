# Experiments

This page is the entry point for the documented experiment result blocks.

## Result Blocks

| Block | Scope | Pages |
| --- | --- | --- |
| [METRAQ NO](experiments_NO/index.md) | METRAQ NO baseline, add24h, trafficdata, and addons setups. | [Performance + Wilcoxon](experiments_NO/summary_tables/summary_paper_performance.md), [Window diagnostics](experiments_NO/window_diagnostics/summary.md) |
| [Cross-Dataset UNet](cross_dataset_no2_nox_unet/index.md) | AIRPARIF and METRAQ NO2/NOX experiments using the UNet model. | [Performance + Wilcoxon](cross_dataset_no2_nox_unet/summary_paper_performance.md), [Window diagnostics](cross_dataset_no2_nox_unet/window_diagnostics/summary.md) |
| [Cross-Dataset Autoencoder](cross_dataset_no2_nox_autoencoder/index.md) | AIRPARIF and METRAQ NO2/NOX experiments using the autoencoder model. | [Performance + Wilcoxon](cross_dataset_no2_nox_autoencoder/summary_paper_performance.md), [Window diagnostics](cross_dataset_no2_nox_autoencoder/window_diagnostics/summary.md) |

## Reference

- [Artifacts Schema](experiments.md): session folder structure, `results.csv` columns, `.npz` artifact fields, and visualization commands.
