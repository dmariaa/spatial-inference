# Cross-Dataset NO2/NOX Autoencoder Window Diagnostics

Aggregate diagnostics for add24h autoencoder cross-dataset windows. Lower values are better for L1 and MSE.

| Experiment | DIP L1 | KRG L1 | IDW L1 | DIP<KRG L1 % | DIP MSE | KRG MSE | IDW MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIRPARIF_NO2 | 9.977 | 8.107 | 8.930 | 22.083 | 187.272 | 148.149 | 168.606 |
| AIRPARIF_NOX | 26.768 | 25.548 | 28.639 | 42.208 | 1985.004 | 1883.347 | 2175.593 |
| METRAQ_NO2 | 6.392 | 6.276 | 6.567 | 44.500 | 84.995 | 81.132 | 87.392 |
| METRAQ_NOX | 12.078 | 12.340 | 12.565 | 50.417 | 582.693 | 609.279 | 624.846 |

## Exemplary Window Visualizations

These standalone Plotly HTML pages show selected best and worst windows by DIP-versus-KRG L1 gap.

| Experiment | Case | Time window | Sensor group | DIP L1 | KRG L1 | IDW L1 | Visualization |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| AIRPARIF_NO2 | Best L1 vs KRG | 2024-09-06 10:00:00 | 2-4-6-42 | 12.152 | 14.700 | 15.554 | [Open HTML](window_visualizations/airparif_no2_best_20240906T100000_2-4-6-42.html) |
| AIRPARIF_NO2 | Worst L1 vs KRG | 2024-03-19 09:00:00 | 3-5-11-41 | 19.460 | 14.742 | 14.789 | [Open HTML](window_visualizations/airparif_no2_worst_20240319T090000_3-5-11-41.html) |
| AIRPARIF_NOX | Best L1 vs KRG | 2024-06-17 06:00:00 | 3-5-11-41 | 11.821 | 18.913 | 23.649 | [Open HTML](window_visualizations/airparif_nox_best_20240617T060000_3-5-11-41.html) |
| AIRPARIF_NOX | Worst L1 vs KRG | 2024-04-26 07:00:00 | 2-4-6-42 | 47.351 | 36.528 | 52.354 | [Open HTML](window_visualizations/airparif_nox_worst_20240426T070000_2-4-6-42.html) |
| METRAQ_NO2 | Best L1 vs KRG | 2024-02-13 14:00:00 | 28079036-28079038-28079056-28079057 | 5.756 | 8.538 | 7.193 | [Open HTML](window_visualizations/metraq_no2_best_20240213T140000_28079036-28079038-28079056-28079057.html) |
| METRAQ_NO2 | Worst L1 vs KRG | 2024-11-25 19:00:00 | 28079024-28079040-28079055-28079060 | 20.643 | 17.615 | 18.208 | [Open HTML](window_visualizations/metraq_no2_worst_20241125T190000_28079024-28079040-28079055-28079060.html) |
| METRAQ_NOX | Best L1 vs KRG | 2024-01-08 09:00:00 | 28079016-28079038-28079056-28079057 | 64.655 | 76.933 | 58.112 | [Open HTML](window_visualizations/metraq_nox_best_20240108T090000_28079016-28079038-28079056-28079057.html) |
| METRAQ_NOX | Worst L1 vs KRG | 2024-07-17 08:00:00 | 28079018-28079039-28079040-28079059 | 22.498 | 16.418 | 16.535 | [Open HTML](window_visualizations/metraq_nox_worst_20240717T080000_28079018-28079039-28079040-28079059.html) |

The source selection data is also available as [exemplary_windows.csv](exemplary_windows.csv), and all row-level diagnostics are available as [row_level_metrics.csv](row_level_metrics.csv).
