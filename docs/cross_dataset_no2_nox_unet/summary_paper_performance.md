## Experiment Summary

| Dataset | Pollutant | Setup | Metric | DIP | KRG | IDW | DIP <br> win % | KRG <br> win % | IDW <br> win % | DIP <br> rank | KRG <br> rank | IDW <br> rank | Friedman p | DIP vs KRG | DIP vs IDW | KRG vs IDW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIRPARIF | NO2 | baseline | MAE | 9.221 | **8.102** | 8.905 | 31.4 | **41.0** | 27.5 | 2.167 | **1.643** | 2.190 | p<.001 | p<.001 | p=.048 | p<.001 |
| AIRPARIF | NO2 | baseline | MSE | 170.213 | **148.164** | 168.072 | **34.0** | 32.7 | 33.3 | 2.106 | **1.750** | 2.143 | p<.001 | p<.001 | n.s. | p<.001 |
| AIRPARIF | NO2 | baseline | WAPE | 0.612 | **0.559** | 0.645 | 31.4 | **41.0** | 27.5 | 2.167 | **1.643** | 2.190 | p<.001 | p<.001 | p=.007 | p<.001 |
| AIRPARIF | NO2 | spatial | MAE | 9.328 | **8.104** | 8.917 | 30.2 | **41.7** | 28.2 | 2.191 | **1.635** | 2.175 | p<.001 | p<.001 | p=.002 | p<.001 |
| AIRPARIF | NO2 | spatial | MSE | 173.123 | **148.080** | 168.365 | 31.8 | 33.9 | **34.3** | 2.130 | **1.725** | 2.144 | p<.001 | p<.001 | n.s. | p<.001 |
| AIRPARIF | NO2 | spatial | WAPE | 0.615 | **0.558** | 0.646 | 30.2 | **41.7** | 28.2 | 2.191 | **1.635** | 2.175 | p<.001 | p<.001 | n.s. | p<.001 |
| AIRPARIF | NOX | baseline | MAE | **25.101** | 25.508 | 28.640 | **49.3** | 35.1 | 15.6 | 1.846 | **1.710** | 2.443 | p<.001 | p=.021 | p<.001 | p<.001 |
| AIRPARIF | NOX | baseline | MSE | 1903.931 | **1881.671** | 2177.152 | **48.0** | 24.9 | 27.1 | 1.851 | **1.846** | 2.303 | p<.001 | n.s. | p<.001 | p<.001 |
| AIRPARIF | NOX | baseline | WAPE | **0.935** | 1.056 | 1.250 | **49.3** | 35.1 | 15.6 | 1.846 | **1.710** | 2.443 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NOX | spatial | MAE | **25.463** | 25.535 | 28.637 | **47.7** | 36.9 | 15.4 | 1.885 | **1.692** | 2.423 | p<.001 | n.s. | p<.001 | p<.001 |
| AIRPARIF | NOX | spatial | MSE | 1915.468 | **1883.016** | 2176.222 | **46.9** | 25.7 | 27.4 | 1.880 | **1.835** | 2.285 | p<.001 | n.s. | p<.001 | p<.001 |
| AIRPARIF | NOX | spatial | WAPE | **0.964** | 1.059 | 1.251 | **47.7** | 36.9 | 15.4 | 1.885 | **1.692** | 2.423 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | baseline | MAE | 6.501 | **6.276** | 6.567 | **37.6** | 34.6 | 27.8 | 2.050 | **1.809** | 2.141 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NO2 | baseline | MSE | 86.529 | **81.132** | 87.392 | **36.8** | 34.3 | 28.8 | 2.072 | **1.803** | 2.125 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NO2 | baseline | WAPE | 0.363 | **0.341** | 0.368 | **37.6** | 34.6 | 27.8 | 2.050 | **1.809** | 2.141 | p<.001 | p<.001 | p=.002 | p<.001 |
| METRAQ | NO2 | spatial | MAE | 6.452 | **6.276** | 6.567 | **38.7** | 34.2 | 27.1 | 2.008 | **1.830** | 2.162 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | spatial | MSE | 87.057 | **81.132** | 87.392 | **37.3** | 35.5 | 27.3 | 2.035 | **1.811** | 2.154 | p<.001 | p<.001 | p=.006 | p<.001 |
| METRAQ | NO2 | spatial | WAPE | 0.357 | **0.341** | 0.368 | **38.7** | 34.2 | 27.1 | 2.008 | **1.830** | 2.162 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | baseline | MAE | **12.148** | 12.340 | 12.565 | **42.8** | 30.7 | 26.5 | 1.940 | **1.884** | 2.176 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | baseline | MSE | **595.580** | 609.279 | 624.846 | **41.4** | 31.4 | 27.3 | 1.975 | **1.850** | 2.175 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | baseline | WAPE | 0.380 | **0.375** | 0.400 | **42.8** | 30.7 | 26.5 | 1.940 | **1.884** | 2.176 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | spatial | MAE | **12.101** | 12.340 | 12.565 | **43.9** | 30.5 | 25.6 | 1.922 | **1.899** | 2.179 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | spatial | MSE | **580.209** | 609.279 | 624.846 | **42.7** | 31.0 | 26.4 | 1.954 | **1.863** | 2.184 | p<.001 | p=.020 | p<.001 | p<.001 |
| METRAQ | NOX | spatial | WAPE | 0.376 | **0.375** | 0.400 | **43.9** | 30.5 | 25.6 | 1.922 | **1.899** | 2.179 | p<.001 | n.s. | p<.001 | p<.001 |

Note: lower is better. Bold marks the lowest mean error in each row. Win % is the fraction of windows where the method has the lowest error among DIP, KRG, and IDW; ties count for each tied method. Mean ranks are Friedman within-window ranks averaged across windows; lower ranks indicate better overall ordering.

Friedman p-values are from the omnibus repeated-measures test across DIP, KRG, and IDW.

Pairwise entries report post-hoc p-values from paired two-sided Wilcoxon signed-rank tests over windows with Holm adjustment; n.s. denotes p >= .05.
