## Experiment Summary

| Dataset | Pollutant | Setup | Metric | DIP | KRG | IDW | DIP <br> win % | KRG <br> win % | IDW <br> win % | DIP <br> rank | KRG <br> rank | IDW <br> rank | Friedman p | DIP vs KRG | DIP vs IDW | KRG vs IDW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| METRAQ | NOX | ensemble_best | MAE | 15.773 | **15.771** | 16.207 | **45.0** | 33.8 | 21.2 | 1.870 | **1.856** | 2.275 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_best | MSE | 1035.689 | 1004.898 | **999.204** | **45.1** | 32.2 | 22.7 | 1.875 | **1.869** | 2.256 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_best | WAPE | **0.365** | 0.376 | 0.406 | **45.0** | 33.8 | 21.2 | 1.870 | **1.856** | 2.275 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_kbest_mean | MAE | 15.801 | **15.771** | 16.207 | **45.4** | 33.2 | 21.3 | **1.861** | 1.862 | 2.277 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_kbest_mean | MSE | 1039.147 | 1004.898 | **999.204** | **43.8** | 33.9 | 22.4 | 1.894 | **1.851** | 2.255 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_kbest_mean | WAPE | **0.366** | 0.376 | 0.406 | **45.4** | 33.2 | 21.3 | **1.861** | 1.862 | 2.277 | p<.001 | p=.001 | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_weighted_kbest_mean | MAE | 15.885 | **15.771** | 16.207 | **44.8** | 34.3 | 21.0 | 1.867 | **1.852** | 2.280 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_weighted_kbest_mean | MSE | 1059.022 | 1004.898 | **999.204** | **44.9** | 33.1 | 22.0 | 1.875 | **1.864** | 2.260 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_weighted_kbest_mean | WAPE | **0.367** | 0.376 | 0.406 | **44.8** | 34.3 | 21.0 | 1.867 | **1.852** | 2.280 | p<.001 | p=.004 | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_top_k_median | MAE | 15.858 | **15.771** | 16.207 | **43.9** | 35.0 | 21.1 | 1.897 | **1.846** | 2.257 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_top_k_median | MSE | 1061.581 | 1004.898 | **999.204** | **43.0** | 34.3 | 22.7 | 1.920 | **1.840** | 2.240 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_top_k_median | WAPE | **0.370** | 0.376 | 0.406 | **43.9** | 35.0 | 21.1 | 1.897 | **1.846** | 2.257 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NOX | ensemble_last | MAE | 16.745 | **15.771** | 16.207 | 37.7 | **42.4** | 19.9 | 2.037 | **1.773** | 2.190 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NOX | ensemble_last | MSE | 1135.991 | 1004.898 | **999.204** | 36.6 | **42.3** | 21.1 | 2.058 | **1.770** | 2.172 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NOX | ensemble_last | WAPE | 0.404 | **0.376** | 0.406 | 37.7 | **42.4** | 19.9 | 2.037 | **1.773** | 2.190 | p<.001 | p<.001 | p=.007 | p<.001 |
| METRAQ | NOX | noensemble_best | MAE | 17.455 | **15.771** | 16.207 | 36.3 | **40.5** | 23.2 | 2.126 | **1.740** | 2.134 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_best | MSE | 1216.300 | 1004.898 | **999.204** | 35.5 | **39.9** | 24.6 | 2.140 | **1.742** | 2.119 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_best | WAPE | 0.408 | **0.376** | 0.406 | 36.3 | **40.5** | 23.2 | 2.126 | **1.740** | 2.134 | p<.001 | p<.001 | p=.043 | p<.001 |
| METRAQ | NOX | noensemble_kbest_mean | MAE | 17.352 | **15.771** | 16.207 | 35.5 | **41.2** | 23.3 | 2.106 | **1.741** | 2.153 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_kbest_mean | MSE | 1208.224 | 1004.898 | **999.204** | 35.0 | **40.8** | 24.2 | 2.129 | **1.738** | 2.132 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_kbest_mean | WAPE | 0.406 | **0.376** | 0.406 | 35.5 | **41.2** | 23.3 | 2.106 | **1.741** | 2.153 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NOX | noensemble_weighted_kbest_mean | MAE | 17.378 | **15.771** | 16.207 | 37.4 | **39.5** | 23.1 | 2.076 | **1.757** | 2.167 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_weighted_kbest_mean | MSE | 1208.582 | 1004.898 | **999.204** | 36.6 | **39.6** | 23.8 | 2.103 | **1.753** | 2.143 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_weighted_kbest_mean | WAPE | 0.401 | **0.376** | 0.406 | 37.4 | **39.5** | 23.1 | 2.076 | **1.757** | 2.167 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NOX | noensemble_top_k_median | MAE | 17.450 | **15.771** | 16.207 | 35.6 | **40.8** | 23.7 | 2.113 | **1.746** | 2.140 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_top_k_median | MSE | 1256.546 | 1004.898 | **999.204** | 36.1 | **39.3** | 24.6 | 2.128 | **1.745** | 2.127 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_top_k_median | WAPE | 0.406 | **0.376** | 0.406 | 35.6 | **40.8** | 23.7 | 2.113 | **1.746** | 2.140 | p<.001 | p<.001 | n.s. | p<.001 |
| METRAQ | NOX | noensemble_last | MAE | 18.672 | **15.771** | 16.207 | 28.3 | **48.6** | 23.1 | 2.287 | **1.664** | 2.049 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_last | MSE | 1357.232 | 1004.898 | **999.204** | 28.1 | **48.6** | 23.3 | 2.300 | **1.664** | 2.036 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NOX | noensemble_last | WAPE | 0.459 | **0.376** | 0.406 | 28.3 | **48.6** | 23.1 | 2.287 | **1.664** | 2.049 | p<.001 | p<.001 | p<.001 | p<.001 |

Note: lower is better. Bold marks the lowest mean error in each row. Win % is the fraction of windows where the method has the lowest error among DIP, KRG, and IDW; ties count for each tied method. Mean ranks are Friedman within-window ranks averaged across windows; lower ranks indicate better overall ordering.

Friedman p-values are from the omnibus repeated-measures test across DIP, KRG, and IDW.

Pairwise entries report post-hoc p-values from paired two-sided Wilcoxon signed-rank tests over windows with Holm adjustment; n.s. denotes p >= .05.
