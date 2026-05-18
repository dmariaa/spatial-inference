## Experiment Summary

| Dataset | Pollutant | Setup | Metric | DIP | KRG | IDW | DIP <br> win % | KRG <br> win % | IDW <br> win % | DIP <br> rank | KRG <br> rank | IDW <br> rank | Friedman p | DIP vs KRG | DIP vs IDW | KRG vs IDW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| METRAQ | NO2 | last_hour_supervision_autoencoder | MAE | 6.542 | **6.276** | 6.567 | 37.2 | **38.8** | 24.0 | 2.026 | **1.811** | 2.163 | p<.001 | p<.001 | p=.016 | p<.001 |
| METRAQ | NO2 | last_hour_supervision_autoencoder | MSE | 92.183 | **81.132** | 87.392 | 36.3 | **37.9** | 25.8 | 2.050 | **1.809** | 2.141 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NO2 | last_hour_supervision_autoencoder | WAPE | 0.359 | **0.341** | 0.368 | 37.2 | **38.8** | 24.0 | 2.026 | **1.811** | 2.163 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | all_hours_supervision_autoencoder | MAE | 6.392 | **6.276** | 6.567 | 37.3 | **38.8** | 23.9 | 2.015 | **1.814** | 2.172 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO2 | all_hours_supervision_autoencoder | MSE | 84.995 | **81.132** | 87.392 | 36.8 | **37.8** | 25.5 | 2.037 | **1.818** | 2.145 | p<.001 | p<.001 | p=.003 | p<.001 |
| METRAQ | NO2 | all_hours_supervision_autoencoder | WAPE | 0.351 | **0.341** | 0.368 | 37.3 | **38.8** | 23.9 | 2.015 | **1.814** | 2.172 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | last_hour_supervision_unet | MAE | 13.097 | **12.340** | 12.565 | **43.7** | 33.6 | 22.7 | 1.925 | **1.891** | 2.184 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | last_hour_supervision_unet | MSE | 3557.385 | **609.279** | 624.846 | **41.1** | 34.3 | 24.6 | 1.969 | **1.859** | 2.172 | p<.001 | p=.029 | p=.002 | p<.001 |
| METRAQ | NOX | last_hour_supervision_unet | WAPE | 0.421 | **0.375** | 0.400 | **43.7** | 33.6 | 22.7 | 1.925 | **1.891** | 2.184 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | all_hours_supervision_unet | MAE | **12.148** | 12.340 | 12.565 | **42.8** | 30.7 | 26.5 | 1.940 | **1.884** | 2.176 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | all_hours_supervision_unet | MSE | **595.580** | 609.279 | 624.846 | **41.4** | 31.4 | 27.3 | 1.975 | **1.850** | 2.175 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | all_hours_supervision_unet | WAPE | 0.380 | **0.375** | 0.400 | **42.8** | 30.7 | 26.5 | 1.940 | **1.884** | 2.176 | p<.001 | n.s. | p<.001 | p<.001 |

Note: lower is better. Bold marks the lowest mean error in each row. Win % is the fraction of windows where the method has the lowest error among DIP, KRG, and IDW; ties count for each tied method. Mean ranks are Friedman within-window ranks averaged across windows; lower ranks indicate better overall ordering.

Friedman p-values are from the omnibus repeated-measures test across DIP, KRG, and IDW.

Pairwise entries report post-hoc p-values from paired two-sided Wilcoxon signed-rank tests over windows with Holm adjustment; n.s. denotes p >= .05.
