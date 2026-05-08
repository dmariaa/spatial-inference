## Experiment Summary

| Dataset | Pollutant | Setup | Metric | DIP | KRG | IDW | DIP <br> win % | KRG <br> win % | IDW <br> win % | DIP <br> rank | KRG <br> rank | IDW <br> rank | Friedman p | DIP vs KRG | DIP vs IDW | KRG vs IDW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIRPARIF | NO2 | baseline | MAE | 9.977 | **8.107** | 8.930 | 20.8 | **50.8** | 28.4 | 2.380 | **1.535** | 2.085 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NO2 | baseline | MSE | 187.272 | **148.149** | 168.606 | 22.3 | **43.1** | 34.6 | 2.323 | **1.642** | 2.035 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NO2 | baseline | WAPE | 0.674 | **0.558** | 0.646 | 20.8 | **50.8** | 28.4 | 2.380 | **1.535** | 2.085 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NO2 | spatial | MAE | 10.044 | **8.096** | 8.916 | 19.1 | **52.2** | 28.7 | 2.406 | **1.523** | 2.071 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NO2 | spatial | MSE | 188.770 | **147.994** | 168.449 | 20.8 | **44.1** | 35.0 | 2.351 | **1.625** | 2.024 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NO2 | spatial | WAPE | 0.678 | **0.558** | 0.645 | 19.1 | **52.2** | 28.7 | 2.406 | **1.523** | 2.071 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NOX | baseline | MAE | 26.768 | **25.548** | 28.639 | 41.1 | **42.8** | 16.1 | 1.992 | **1.633** | 2.375 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NOX | baseline | MSE | 1985.004 | **1883.347** | 2175.593 | **38.8** | 34.2 | 27.0 | 1.988 | **1.772** | 2.240 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NOX | baseline | WAPE | **1.050** | 1.058 | 1.250 | 41.1 | **42.8** | 16.1 | 1.992 | **1.633** | 2.375 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NOX | spatial | MAE | 27.059 | **25.452** | 28.586 | 38.6 | **44.5** | 16.8 | 2.044 | **1.609** | 2.347 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NOX | spatial | MSE | 2020.204 | **1879.168** | 2174.059 | **37.0** | 34.5 | 28.4 | 2.019 | **1.760** | 2.221 | p<.001 | p<.001 | p<.001 | p<.001 |
| AIRPARIF | NOX | spatial | WAPE | 1.054 | **1.052** | 1.249 | 38.6 | **44.5** | 16.8 | 2.044 | **1.609** | 2.347 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | baseline | MAE | 6.392 | **6.276** | 6.567 | 37.3 | **38.8** | 23.9 | 2.015 | **1.814** | 2.172 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | baseline | MSE | 84.995 | **81.132** | 87.392 | 36.8 | **37.8** | 25.5 | 2.037 | **1.818** | 2.145 | p<.001 | p<.001 | p=.003 | p<.001 |
| METRAQ | NO2 | baseline | WAPE | 0.351 | **0.341** | 0.368 | 37.3 | **38.8** | 23.9 | 2.015 | **1.814** | 2.172 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | spatial | MAE | 6.383 | **6.276** | 6.567 | 37.6 | **38.5** | 23.9 | 2.020 | **1.811** | 2.169 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | spatial | MSE | 85.353 | **81.132** | 87.392 | 36.9 | **38.1** | 25.0 | 2.035 | **1.815** | 2.151 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | spatial | WAPE | 0.350 | **0.341** | 0.368 | 37.6 | **38.5** | 23.9 | 2.020 | **1.811** | 2.169 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | baseline | MAE | **12.078** | 12.340 | 12.565 | **42.7** | 34.1 | 23.2 | 1.908 | **1.889** | 2.203 | p<.001 | p=.048 | p<.001 | p<.001 |
| METRAQ | NOX | baseline | MSE | **582.693** | 609.279 | 624.846 | **41.8** | 34.2 | 24.0 | 1.939 | **1.865** | 2.196 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | baseline | WAPE | **0.371** | 0.375 | 0.400 | **42.7** | 34.1 | 23.2 | 1.908 | **1.889** | 2.203 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | spatial | MAE | **12.079** | 12.340 | 12.565 | **44.3** | 32.6 | 23.1 | **1.888** | 1.904 | 2.208 | p<.001 | p=.012 | p<.001 | p<.001 |
| METRAQ | NOX | spatial | MSE | **587.156** | 609.279 | 624.846 | **42.8** | 33.2 | 24.0 | 1.920 | **1.882** | 2.198 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | spatial | WAPE | **0.371** | 0.375 | 0.400 | **44.3** | 32.6 | 23.1 | **1.888** | 1.904 | 2.208 | p<.001 | p=.024 | p<.001 | p<.001 |

Note: lower is better. Bold marks the lowest mean error in each row. Win % is the fraction of windows where the method has the lowest error among DIP, KRG, and IDW; ties count for each tied method. Mean ranks are Friedman within-window ranks averaged across windows; lower ranks indicate better overall ordering.

Friedman p-values are from the omnibus repeated-measures test across DIP, KRG, and IDW.

Pairwise entries report post-hoc p-values from paired two-sided Wilcoxon signed-rank tests over windows with Holm adjustment; n.s. denotes p >= .05.
