# Experiments

This page is the entry point for the documented experiment result blocks.

## Result Blocks

| Block | Scope | Pages                                                                                                                                                                        |
| --- | --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [METRAQ NO](experiments_NO/index.md) | METRAQ NO baseline, add24h, trafficdata, and addons setups. | [Performance summary](experiments_NO/summary_tables/summary_paper_performance.md), [Window diagnostics](experiments_NO/window_diagnostics/summary.md)                        |
| [Cross-Dataset UNet](cross_dataset_no2_nox_unet/index.md) | AIRPARIF and METRAQ NO2/NOX experiments using the UNet model. | [Performance summary](cross_dataset_no2_nox_unet/summary_paper_performance.md), [Window diagnostics](cross_dataset_no2_nox_unet/window_diagnostics/summary.md)               |
| [Cross-Dataset Autoencoder](cross_dataset_no2_nox_autoencoder/index.md) | AIRPARIF and METRAQ NO2/NOX experiments using the autoencoder model. | [Performance summary](cross_dataset_no2_nox_autoencoder/summary_paper_performance.md), [Window diagnostics](cross_dataset_no2_nox_autoencoder/window_diagnostics/summary.md) |
| [Surface Selection 2023](surface_selection_2023/index.md) | METRAQ NOX 2023 comparison of ensemble vs no ensemble and the `validation`, `validation_weighted`, `validation_median`, and `last` surface selectors. | [Performance summary](surface_selection_2023/summary_paper_performance.md), [Narrative summary](surface_selection_2023/interim_summary.md)                                   |

## Experiment Inventory

This list covers the configured experiment folders under `output/experiments`, excluding `experiment_test_airparif` because it is a test run. `pollutant_diagnostics` is a derived diagnostics bundle, not a configured experiment.

Common settings unless noted: normalized inputs, ensemble size 5, best 3 models retained, 250 epochs, learning rate 0.01, MAE optimization loss, 10 spread test groups of 4 sensors, max 2 uses per sensor, 20 windows per month in 2024, start hours 06:00-20:00, and weekend fraction 0.4.

| Block | Experiment | Dataset | Pollutant | Model | Hours | Extra channels |
| --- | --- | --- | --- | --- | ---: | --- |
| METRAQ NO | baseline | METRAQ | NO | autoencoder | 1 | none |
| METRAQ NO | add24h | METRAQ | NO | autoencoder | 24 | none |
| METRAQ NO | trafficdata | METRAQ | NO | autoencoder | 24 | distance-to-sensors, traffic |
| METRAQ NO | addons | METRAQ | NO | autoencoder | 24 | distance-to-sensors |
| Cross-Dataset UNet | airparif_no2_baseline | AIRPARIF | NO2 | unet | 1 | none |
| Cross-Dataset UNet | airparif_no2_add24h | AIRPARIF | NO2 | unet | 24 | none |
| Cross-Dataset UNet | airparif_no2_addons | AIRPARIF | NO2 | unet | 24 | coordinates, distance-to-sensors |
| Cross-Dataset UNet | airparif_nox_baseline | AIRPARIF | NOX | unet | 1 | none |
| Cross-Dataset UNet | airparif_nox_add24h | AIRPARIF | NOX | unet | 24 | none |
| Cross-Dataset UNet | airparif_nox_addons | AIRPARIF | NOX | unet | 24 | coordinates, distance-to-sensors |
| Cross-Dataset UNet | metraq_no2_baseline | METRAQ | NO2 | unet | 1 | none |
| Cross-Dataset UNet | metraq_no2_add24h | METRAQ | NO2 | unet | 24 | none |
| Cross-Dataset UNet | metraq_no2_addons | METRAQ | NO2 | unet | 24 | coordinates, distance-to-sensors |
| Cross-Dataset UNet | metraq_nox_baseline | METRAQ | NOX | unet | 1 | none |
| Cross-Dataset UNet | metraq_nox_add24h | METRAQ | NOX | unet | 24 | none |
| Cross-Dataset UNet | metraq_nox_addons | METRAQ | NOX | unet | 24 | coordinates, distance-to-sensors |
| Cross-Dataset Autoencoder | airparif_no2_baseline | AIRPARIF | NO2 | autoencoder | 1 | none |
| Cross-Dataset Autoencoder | airparif_no2_add24h | AIRPARIF | NO2 | autoencoder | 24 | none |
| Cross-Dataset Autoencoder | airparif_no2_addons | AIRPARIF | NO2 | autoencoder | 24 | coordinates, distance-to-sensors |
| Cross-Dataset Autoencoder | airparif_nox_baseline | AIRPARIF | NOX | autoencoder | 1 | none |
| Cross-Dataset Autoencoder | airparif_nox_add24h | AIRPARIF | NOX | autoencoder | 24 | none |
| Cross-Dataset Autoencoder | airparif_nox_addons | AIRPARIF | NOX | autoencoder | 24 | coordinates, distance-to-sensors |
| Cross-Dataset Autoencoder | metraq_no2_baseline | METRAQ | NO2 | autoencoder | 1 | none |
| Cross-Dataset Autoencoder | metraq_no2_add24h | METRAQ | NO2 | autoencoder | 24 | none |
| Cross-Dataset Autoencoder | metraq_no2_addons | METRAQ | NO2 | autoencoder | 24 | coordinates, distance-to-sensors |
| Cross-Dataset Autoencoder | metraq_nox_baseline | METRAQ | NOX | autoencoder | 1 | none |
| Cross-Dataset Autoencoder | metraq_nox_add24h | METRAQ | NOX | autoencoder | 24 | none |
| Cross-Dataset Autoencoder | metraq_nox_addons | METRAQ | NOX | autoencoder | 24 | coordinates, distance-to-sensors |
| Surface Selection 2023 | ensemble_best | METRAQ | NOX | autoencoder | 24 | none |
| Surface Selection 2023 | ensemble_kbest_mean | METRAQ | NOX | autoencoder | 24 | none |
| Surface Selection 2023 | ensemble_last | METRAQ | NOX | autoencoder | 24 | none |
| Surface Selection 2023 | noensemble_best | METRAQ | NOX | autoencoder | 24 | none |
| Surface Selection 2023 | noensemble_kbest_mean | METRAQ | NOX | autoencoder | 24 | none |
| Surface Selection 2023 | noensemble_last | METRAQ | NOX | autoencoder | 24 | none |

## Reference

- [Artifacts Schema](experiments.md): session folder structure, `results.csv` columns, `.npz` artifact fields, and visualization commands.
