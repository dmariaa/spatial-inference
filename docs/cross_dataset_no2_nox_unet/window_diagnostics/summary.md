# Cross-Dataset NO2/NOX Window Diagnostics

Aggregate diagnostics for add24h UNet cross-dataset windows. Lower values are better for L1 and MSE.

| Experiment | DIP L1 | KRG L1 | IDW L1 | DIP<KRG L1 % | DIP MSE | KRG MSE | IDW MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIRPARIF_NO2 | 9.221 | 8.102 | 8.905 | 32.792 | 170.213 | 148.164 | 168.072 |
| AIRPARIF_NOX | 25.101 | 25.508 | 28.640 | 50.375 | 1903.931 | 1881.671 | 2177.152 |
| METRAQ_NO2 | 6.501 | 6.276 | 6.567 | 44.042 | 86.529 | 81.132 | 87.392 |
| METRAQ_NOX | 12.148 | 12.340 | 12.565 | 49.917 | 595.580 | 609.279 | 624.846 |

## Exemplary Window Visualizations

These standalone Plotly HTML pages show selected best and worst windows by DIP-versus-KRG L1 gap.

| Experiment | Case | Time window | Sensor group | DIP L1 | KRG L1 | IDW L1 | Visualization |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| AIRPARIF_NO2 | Best L1 vs KRG | 2024-08-19 06:00:00 | 3-5-11-41 | 10.200 | 11.612 | 10.687 | [Open HTML](window_visualizations/airparif_no2_best_20240819T060000_3-5-11-41.html) |
| AIRPARIF_NO2 | Worst L1 vs KRG | 2024-05-17 20:00:00 | 2-4-6-42 | 21.784 | 14.960 | 21.132 | [Open HTML](window_visualizations/airparif_no2_worst_20240517T200000_2-4-6-42.html) |
| AIRPARIF_NOX | Best L1 vs KRG | 2024-12-29 17:00:00 | 4-38-41-42 | 71.737 | 83.296 | 90.662 | [Open HTML](window_visualizations/airparif_nox_best_20241229T170000_4-38-41-42.html) |
| AIRPARIF_NOX | Worst L1 vs KRG | 2024-03-19 09:00:00 | 8-15-27-32 | 36.203 | 20.780 | 30.750 | [Open HTML](window_visualizations/airparif_nox_worst_20240319T090000_8-15-27-32.html) |
| METRAQ_NO2 | Best L1 vs KRG | 2024-02-13 14:00:00 | 28079004-28079016-28079036-28079050 | 2.639 | 4.422 | 3.443 | [Open HTML](window_visualizations/metraq_no2_best_20240213T140000_28079004-28079016-28079036-28079050.html) |
| METRAQ_NO2 | Worst L1 vs KRG | 2024-10-19 16:00:00 | 28079004-28079047-28079048-28079050 | 8.614 | 4.756 | 4.727 | [Open HTML](window_visualizations/metraq_no2_worst_20241019T160000_28079004-28079047-28079048-28079050.html) |
| METRAQ_NOX | Best L1 vs KRG | 2024-01-08 09:00:00 | 28079024-28079040-28079055-28079060 | 78.358 | 98.454 | 64.681 | [Open HTML](window_visualizations/metraq_nox_best_20240108T090000_28079024-28079040-28079055-28079060.html) |
| METRAQ_NOX | Worst L1 vs KRG | 2024-07-17 08:00:00 | 28079016-28079038-28079056-28079057 | 18.856 | 13.678 | 11.907 | [Open HTML](window_visualizations/metraq_nox_worst_20240717T080000_28079016-28079038-28079056-28079057.html) |

The source selection data is also available as [exemplary_windows.csv](exemplary_windows.csv), and all row-level diagnostics are available as [row_level_metrics.csv](row_level_metrics.csv).
