## Experiment Summary

| Dataset | Pollutant | Setup | Metric | DIP | KRG | IDW | DIP <br> win % | KRG <br> win % | IDW <br> win % | DIP <br> rank | KRG <br> rank | IDW <br> rank | Friedman p | DIP vs KRG | DIP vs IDW | KRG vs IDW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| METRAQ | NO | baseline | MAE | **4.655** | 4.761 | 4.702 | **51.3** | 24.7 | 24.2 | **1.803** | 2.019 | 2.178 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO | baseline | MSE | 165.690 | **157.072** | 160.993 | **44.8** | 31.9 | 23.5 | 1.924 | **1.865** | 2.211 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NO | baseline | WAPE | **0.627** | 0.668 | 0.692 | **51.3** | 24.7 | 24.2 | **1.803** | 2.019 | 2.178 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO | add24h | MAE | **4.478** | 4.761 | 4.702 | **57.5** | 20.9 | 21.8 | **1.684** | 2.083 | 2.234 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO | add24h | MSE | 159.477 | **157.072** | 160.993 | **47.7** | 28.0 | 24.5 | **1.858** | 1.900 | 2.242 | p<.001 | p=.002 | p<.001 | p<.001 |
| METRAQ | NO | add24h | WAPE | **0.570** | 0.668 | 0.692 | **57.5** | 20.9 | 21.8 | **1.684** | 2.083 | 2.234 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO | trafficdata | MAE | **4.489** | 4.761 | 4.702 | **56.1** | 21.4 | 22.8 | **1.710** | 2.070 | 2.220 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO | trafficdata | MSE | 158.642 | **157.072** | 160.993 | **45.8** | 29.1 | 25.2 | 1.891 | **1.886** | 2.223 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NO | trafficdata | WAPE | **0.577** | 0.668 | 0.692 | **56.1** | 21.4 | 22.8 | **1.710** | 2.070 | 2.220 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO | addons | MAE | **4.442** | 4.761 | 4.702 | **56.6** | 21.2 | 22.5 | **1.702** | 2.072 | 2.225 | p<.001 | p<.001 | p<.001 | p<.001 |
| METRAQ | NO | addons | MSE | **150.374** | 157.072 | 160.993 | **48.0** | 28.3 | 23.9 | **1.861** | 1.900 | 2.239 | p<.001 | n.s. | p<.001 | p<.001 |
| METRAQ | NO | addons | WAPE | **0.575** | 0.668 | 0.692 | **56.6** | 21.2 | 22.5 | **1.702** | 2.072 | 2.225 | p<.001 | p<.001 | p<.001 | p<.001 |

Note: lower is better. Bold marks the lowest mean error in each row. Win % is the fraction of windows where the method has the lowest error among DIP, KRG, and IDW; ties count for each tied method. Mean ranks are Friedman within-window ranks averaged across windows; lower ranks indicate better overall ordering.

Friedman p-values are from the omnibus repeated-measures test across DIP, KRG, and IDW.

Pairwise entries report post-hoc p-values from paired two-sided Wilcoxon signed-rank tests over windows with Holm adjustment; n.s. denotes p >= .05.
