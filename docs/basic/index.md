# Basic Supervision Window Comparison

This section compares METRAQ add24h setups under two pollutant supervision windows.

All runs use the same 24-hour input context, normalized inputs, 5-member ensemble, 250 epochs, MAE optimization, and the same 2,400 sensor-group/time-window evaluation rows for each pollutant. The only intended objective difference inside each pollutant/model pair is the pollutant timestep scope used for optimization and validation:

- `last`: optimize and validate on the final pollutant hour only.
- `all`: optimize and validate on all pollutant hours in the 24-hour target window.

## Compared Runs

| Pollutant | Last-hour supervision | All-hours supervision | Model |
| --- | --- | --- | --- |
| NO2 | `basic/single_channel_supervision_NO2` | `cross_dataset_no2_nox_autoencoder/metraq_no2_add24h` | autoencoder |
| NOX | `basic/single_channel_supervision_NOX` | `cross_dataset_no2_nox_unet/metraq_nox_add24h` | UNet |

## Headline Result

Last-hour-only supervision does not improve the matched add24h setup for either pollutant.

| Pollutant | Metric | Last-hour mean | All-hours mean | Relative delta | Last-hour better rows | All-hours better rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NO2 | MAE | 6.542 | 6.392 | +2.35% | 1146 | 1254 |
| NO2 | MSE | 92.183 | 84.995 | +8.46% | 1169 | 1231 |
| NOX | MAE | 13.097 | 12.148 | +7.82% | 1230 | 1170 |
| NOX | MSE | 3557.385 | 595.580 | +497.30% | 1212 | 1188 |

For NO2, all-hours supervision is modestly but consistently better on aggregate. For NOX, the row win counts are close and slightly favor last-hour supervision, but the means are much worse because the last-hour UNet run has severe outlier failures, especially in MSE. The largest NOX failure is `2024-03-17 16:00` for sensor group `28079018-28079027-28079054-28079058`, where DIP MAE rises to 1307.154 and DIP MSE rises to 6,793,226.

Overall, these results support keeping all-hours pollutant supervision as the default training signal for the 24-hour input experiments, even though final evaluation is on the last-hour surface.

## Summary Tables

- [Performance Summary](summary_paper_performance.md): aggregate means, win rates, Friedman mean ranks, Friedman omnibus tests, and pairwise Wilcoxon post-hoc comparisons with Holm adjustment for DIP, KRG, and IDW.
- [Compact Comparison](summary_compact_comparison.md): compact DIP/KRG/IDW mean and win-rate table.
- [Supervision Window Pairwise CSV](supervision_window_pairwise.csv): direct paired last-hour vs all-hours deltas by pollutant and metric.
- [Raw Performance Data](summary_paper_performance_data.csv): source data for the performance summary table.

## Window Diagnostics

- [Window Diagnostics](window_diagnostics/summary.md): aggregate window-level diagnostics for both supervision variants.
- [Exemplary windows CSV](window_diagnostics/exemplary_windows.csv): best and worst windows by DIP-vs-KRG gap.
- [Row-level metrics CSV](window_diagnostics/row_level_metrics.csv): per-window metrics used for diagnostics.
