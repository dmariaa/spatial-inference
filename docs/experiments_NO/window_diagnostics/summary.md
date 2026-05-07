# NO Window Diagnostics

Aggregate diagnostics for the METRAQ NO windows. Lower values are better for L1 and MSE.

| Experiment | DIP L1 | KRG L1 | IDW L1 | DIP<KRG L1 % | DIP MSE | KRG MSE | IDW MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| METRAQ_NO_ADD24H | 4.478 | 4.761 | 4.702 | 64.958 | 159.477 | 157.072 | 160.993 |
| METRAQ_NO_ADDONS | 4.442 | 4.761 | 4.702 | 63.958 | 150.374 | 157.072 | 160.993 |
| METRAQ_NO_BASELINE | 4.655 | 4.761 | 4.702 | 58.583 | 165.690 | 157.072 | 160.993 |
| METRAQ_NO_TRAFFICDATA | 4.489 | 4.761 | 4.702 | 63.708 | 158.642 | 157.072 | 160.993 |

## Exemplary Window Visualizations

These standalone Plotly HTML pages show selected best and worst windows by DIP-versus-KRG L1 gap.

| Experiment | Case | Time window | Sensor group | DIP L1 | KRG L1 | IDW L1 | Visualization |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| METRAQ_NO_BASELINE | Best L1 vs KRG | 2024-01-10 08:00:00 | 28079018-28079039-28079040-28079059 | 14.189 | 20.461 | 21.387 | [Open HTML](window_visualizations/metraq_no_baseline_best_20240110T080000_28079018-28079039-28079040-28079059.html) |
| METRAQ_NO_BASELINE | Worst L1 vs KRG | 2024-12-02 20:00:00 | 28079036-28079038-28079056-28079057 | 69.218 | 57.113 | 53.239 | [Open HTML](window_visualizations/metraq_no_baseline_worst_20241202T200000_28079036-28079038-28079056-28079057.html) |
| METRAQ_NO_ADD24H | Best L1 vs KRG | 2024-12-31 08:00:00 | 28079016-28079038-28079056-28079057 | 10.518 | 13.553 | 11.224 | [Open HTML](window_visualizations/metraq_no_add24h_best_20241231T080000_28079016-28079038-28079056-28079057.html) |
| METRAQ_NO_ADD24H | Worst L1 vs KRG | 2024-12-02 20:00:00 | 28079024-28079027-28079054-28079058 | 105.115 | 100.185 | 98.177 | [Open HTML](window_visualizations/metraq_no_add24h_worst_20241202T200000_28079024-28079027-28079054-28079058.html) |
| METRAQ_NO_TRAFFICDATA | Best L1 vs KRG | 2024-12-31 08:00:00 | 28079018-28079027-28079054-28079058 | 12.305 | 16.575 | 12.693 | [Open HTML](window_visualizations/metraq_no_trafficdata_best_20241231T080000_28079018-28079027-28079054-28079058.html) |
| METRAQ_NO_TRAFFICDATA | Worst L1 vs KRG | 2024-12-02 20:00:00 | 28079017-28079039-28079049-28079059 | 104.458 | 103.712 | 97.897 | [Open HTML](window_visualizations/metraq_no_trafficdata_worst_20241202T200000_28079017-28079039-28079049-28079059.html) |
| METRAQ_NO_ADDONS | Best L1 vs KRG | 2024-01-08 09:00:00 | 28079036-28079038-28079056-28079057 | 74.928 | 77.779 | 74.846 | [Open HTML](window_visualizations/metraq_no_addons_best_20240108T090000_28079036-28079038-28079056-28079057.html) |
| METRAQ_NO_ADDONS | Worst L1 vs KRG | 2024-12-16 11:00:00 | 28079017-28079039-28079049-28079059 | 14.719 | 13.431 | 16.165 | [Open HTML](window_visualizations/metraq_no_addons_worst_20241216T110000_28079017-28079039-28079049-28079059.html) |

The source selection data is also available as [exemplary_windows.csv](exemplary_windows.csv), and all row-level diagnostics are available as [row_level_metrics.csv](row_level_metrics.csv).
