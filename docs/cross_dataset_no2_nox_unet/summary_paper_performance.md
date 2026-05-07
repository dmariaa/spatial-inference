# Cross-Dataset NO2/NOX Performance Summary

Aggregate UNet cross-dataset results for AIRPARIF and METRAQ NO2/NOX experiments.

| Dataset | Pollutant | Setup | Metric | DIP | KRG | IDW | DIP win % | KRG win % | IDW win % | DIP vs KRG | DIP vs IDW | KRG vs IDW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIRPARIF | NO2 | baseline | MAE | 9.221 | **8.102** | 8.905 | 31.4 | **41.0** | 27.5 | KRG lower (p<.001) | IDW lower (p=.048) | KRG lower (p<.001) |
| AIRPARIF | NO2 | baseline | MSE | 170.213 | **148.164** | 168.072 | **34.0** | 32.7 | 33.3 | KRG lower (p<.001) | IDW lower, n.s. (p=.457) | KRG lower (p<.001) |
| AIRPARIF | NO2 | baseline | WAPE | 0.612 | **0.559** | 0.645 | 31.4 | **41.0** | 27.5 | KRG lower (p<.001) | DIP lower (p=.007) | KRG lower (p<.001) |
| AIRPARIF | NO2 | spatial | MAE | 9.328 | **8.104** | 8.917 | 30.2 | **41.7** | 28.2 | KRG lower (p<.001) | IDW lower (p=.002) | KRG lower (p<.001) |
| AIRPARIF | NO2 | spatial | MSE | 173.123 | **148.080** | 168.365 | 31.8 | 33.9 | **34.3** | KRG lower (p<.001) | IDW lower, n.s. (p=.871) | KRG lower (p<.001) |
| AIRPARIF | NO2 | spatial | WAPE | 0.615 | **0.558** | 0.646 | 30.2 | **41.7** | 28.2 | KRG lower (p<.001) | DIP lower, n.s. (p=.082) | KRG lower (p<.001) |
| AIRPARIF | NOX | baseline | MAE | **25.101** | 25.508 | 28.640 | **49.3** | 35.1 | 15.6 | DIP lower (p=.021) | DIP lower (p<.001) | KRG lower (p<.001) |
| AIRPARIF | NOX | baseline | MSE | 1903.931 | **1881.671** | 2177.152 | **48.0** | 24.9 | 27.1 | KRG lower, n.s. (p=.521) | DIP lower (p<.001) | KRG lower (p<.001) |
| AIRPARIF | NOX | baseline | WAPE | **0.935** | 1.056 | 1.250 | **49.3** | 35.1 | 15.6 | DIP lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| AIRPARIF | NOX | spatial | MAE | **25.463** | 25.535 | 28.637 | **47.7** | 36.9 | 15.4 | DIP lower, n.s. (p=.819) | DIP lower (p<.001) | KRG lower (p<.001) |
| AIRPARIF | NOX | spatial | MSE | 1915.468 | **1883.016** | 2176.222 | **46.9** | 25.7 | 27.4 | KRG lower, n.s. (p=.643) | DIP lower (p<.001) | KRG lower (p<.001) |
| AIRPARIF | NOX | spatial | WAPE | **0.964** | 1.059 | 1.251 | **47.7** | 36.9 | 15.4 | DIP lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NO2 | baseline | MAE | 6.501 | **6.276** | 6.567 | **37.6** | 34.6 | 27.8 | KRG lower (p<.001) | DIP lower, n.s. (p=.094) | KRG lower (p<.001) |
| METRAQ | NO2 | baseline | MSE | 86.529 | **81.132** | 87.392 | **36.8** | 34.3 | 28.8 | KRG lower (p<.001) | DIP lower, n.s. (p=.789) | KRG lower (p<.001) |
| METRAQ | NO2 | baseline | WAPE | 0.363 | **0.341** | 0.368 | **37.6** | 34.6 | 27.8 | KRG lower (p<.001) | DIP lower (p=.002) | KRG lower (p<.001) |
| METRAQ | NO2 | spatial | MAE | 6.452 | **6.276** | 6.567 | **38.7** | 34.2 | 27.1 | KRG lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NO2 | spatial | MSE | 87.057 | **81.132** | 87.392 | **37.3** | 35.5 | 27.3 | KRG lower (p<.001) | DIP lower (p=.006) | KRG lower (p<.001) |
| METRAQ | NO2 | spatial | WAPE | 0.357 | **0.341** | 0.368 | **38.7** | 34.2 | 27.1 | KRG lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NOX | baseline | MAE | **12.148** | 12.340 | 12.565 | **42.8** | 30.7 | 26.5 | DIP lower, n.s. (p=.958) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NOX | baseline | MSE | **595.580** | 609.279 | 624.846 | **41.4** | 31.4 | 27.3 | DIP lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NOX | baseline | WAPE | 0.380 | **0.375** | 0.400 | **42.8** | 30.7 | 26.5 | KRG lower, n.s. (p=.987) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NOX | spatial | MAE | **12.101** | 12.340 | 12.565 | **43.9** | 30.5 | 25.6 | DIP lower, n.s. (p=.279) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NOX | spatial | MSE | **580.209** | 609.279 | 624.846 | **42.7** | 31.0 | 26.4 | DIP lower (p=.020) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ | NOX | spatial | WAPE | 0.376 | **0.375** | 0.400 | **43.9** | 30.5 | 25.6 | KRG lower, n.s. (p=.176) | DIP lower (p<.001) | KRG lower (p<.001) |

Note: lower is better. Bold marks the lowest mean error in each row. Win % is the fraction of windows where the method has the lowest error among DIP, KRG, and IDW; ties count for each tied method.

Wilcoxon entries name the method with the lower mean error for that pair. P-values are from paired two-sided Wilcoxon signed-rank tests over windows; n.s. denotes p >= .05.
