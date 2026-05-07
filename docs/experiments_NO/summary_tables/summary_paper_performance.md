# NO Performance Summary

Aggregate METRAQ NO results across the baseline, add24h, trafficdata, and addons setups.

| Dataset | Pollutant |       Setup | Metric |         DIP |         KRG |     IDW | DIP win % | KRG win % | IDW win % |               DIP vs KRG |         DIP vs IDW |         KRG vs IDW |
|---------|----------:|------------:|-------:|------------:|------------:|--------:|----------:|----------:|----------:|-------------------------:|-------------------:|-------------------:|
| METRAQ  |        NO |    baseline |    MAE |   **4.655** |       4.761 |   4.702 |  **51.3** |      24.7 |      24.2 |       DIP lower (p<.001) | DIP lower (p<.001) | IDW lower (p<.001) |
| METRAQ  |        NO |    baseline |    MSE |     165.690 | **157.072** | 160.993 |  **44.8** |      31.9 |      23.5 | KRG lower, n.s. (p=.367) | IDW lower (p<.001) | KRG lower (p<.001) |
| METRAQ  |        NO |    baseline |   WAPE |   **0.627** |       0.668 |   0.692 |  **51.3** |      24.7 |      24.2 |       DIP lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ  |        NO |      add24h |    MAE |   **4.478** |       4.761 |   4.702 |  **57.5** |      20.9 |      21.8 |       DIP lower (p<.001) | DIP lower (p<.001) | IDW lower (p<.001) |
| METRAQ  |        NO |      add24h |    MSE |     159.477 | **157.072** | 160.993 |  **47.7** |      28.0 |      24.5 |       KRG lower (p=.002) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ  |        NO |      add24h |   WAPE |   **0.570** |       0.668 |   0.692 |  **57.5** |      20.9 |      21.8 |       DIP lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ  |        NO | trafficdata |    MAE |   **4.489** |       4.761 |   4.702 |  **56.1** |      21.4 |      22.8 |       DIP lower (p<.001) | DIP lower (p<.001) | IDW lower (p<.001) |
| METRAQ  |        NO | trafficdata |    MSE |     158.642 | **157.072** | 160.993 |  **45.8** |      29.1 |      25.2 | KRG lower, n.s. (p=.182) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ  |        NO | trafficdata |   WAPE |   **0.577** |       0.668 |   0.692 |  **56.1** |      21.4 |      22.8 |       DIP lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ  |        NO |      addons |    MAE |   **4.442** |       4.761 |   4.702 |  **56.6** |      21.2 |      22.5 |       DIP lower (p<.001) | DIP lower (p<.001) | IDW lower (p<.001) |
| METRAQ  |        NO |      addons |    MSE | **150.374** |     157.072 | 160.993 |  **48.0** |      28.3 |      23.9 | DIP lower, n.s. (p=.069) | DIP lower (p<.001) | KRG lower (p<.001) |
| METRAQ  |        NO |      addons |   WAPE |   **0.575** |       0.668 |   0.692 |  **56.6** |      21.2 |      22.5 |       DIP lower (p<.001) | DIP lower (p<.001) | KRG lower (p<.001) |

Note: lower is better. Bold marks the lowest mean error in each row. Win % is the fraction of windows where the method
has the lowest error among DIP, KRG, and IDW; ties count for each tied method.

