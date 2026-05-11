# Surface Selection 2023 Summary

## DIP ranking

| Experiment | DIP L1 | DIP MSE |
| --- | ---: | ---: |
| `ensemble_best` | **15.773** | **1035.689** |
| `ensemble_kbest_mean` | 15.801 | 1039.147 |
| `ensemble_top_k_median` | 15.858 | 1061.581 |
| `ensemble_weighted_kbest_mean` | 15.885 | 1059.022 |
| `ensemble_last` | 16.745 | 1135.991 |
| `noensemble_kbest_mean` | 17.352 | 1208.224 |
| `noensemble_weighted_kbest_mean` | 17.378 | 1208.582 |
| `noensemble_top_k_median` | 17.450 | 1256.546 |
| `noensemble_best` | 17.455 | 1216.300 |
| `noensemble_last` | 18.672 | 1357.232 |

Current DIP ordering:

1. `ensemble_best`
2. `ensemble_kbest_mean`
3. `ensemble_top_k_median`
4. `ensemble_weighted_kbest_mean`
5. `ensemble_last`
6. `noensemble_kbest_mean`
7. `noensemble_weighted_kbest_mean`
8. `noensemble_top_k_median`
9. `noensemble_best`
10. `noensemble_last`

## Main takeaways

- Using the **best validation surface** is better than taking the **last epoch**.
- Using an **ensemble** is better than not using one.
- `k_best_n = 3` does **not** improve over the single best validation surface in the ensemble case.
- The new `validation_weighted` and `validation_median` selectors do **not** beat the existing `ensemble_best` or `ensemble_kbest_mean` setups.
- In the no-ensemble case, `k_best_n = 3` and `validation_weighted` stay very close, while `validation_median` is clearly weaker on `MSE`.
- Overall, **`ensemble_best` is the strongest DIP setup**.

## Paired DIP comparisons

The main new paired comparisons are:

| Comparison | Metric | Better mean | Win split | Wilcoxon p |
| --- | --- | ---: | ---: | ---: |
| `ensemble_weighted_kbest_mean` vs `ensemble_best` | `L1` | 15.885 vs 15.773 | 50.5 / 49.5 | 8.49e-01 |
| `ensemble_weighted_kbest_mean` vs `ensemble_best` | `MSE` | 1059.022 vs 1035.689 | 50.1 / 49.9 | 8.23e-01 |
| `ensemble_weighted_kbest_mean` vs `ensemble_kbest_mean` | `L1` | 15.885 vs 15.801 | 51.3 / 48.7 | 9.83e-01 |
| `ensemble_weighted_kbest_mean` vs `ensemble_kbest_mean` | `MSE` | 1059.022 vs 1039.147 | 51.2 / 48.8 | 8.53e-01 |
| `ensemble_top_k_median` vs `ensemble_best` | `L1` | 15.858 vs 15.773 | 49.5 / 50.5 | 6.12e-01 |
| `ensemble_top_k_median` vs `ensemble_best` | `MSE` | 1061.581 vs 1035.689 | 48.3 / 51.7 | 2.56e-01 |
| `ensemble_top_k_median` vs `ensemble_kbest_mean` | `L1` | 15.858 vs 15.801 | 49.5 / 50.5 | 1.89e-01 |
| `ensemble_top_k_median` vs `ensemble_kbest_mean` | `MSE` | 1061.581 vs 1039.147 | 48.3 / 51.7 | 4.10e-02 |
| `noensemble_weighted_kbest_mean` vs `noensemble_best` | `L1` | 17.378 vs 17.455 | 49.5 / 50.5 | 6.82e-01 |
| `noensemble_weighted_kbest_mean` vs `noensemble_best` | `MSE` | 1208.582 vs 1216.300 | 50.7 / 49.3 | 1.71e-01 |
| `noensemble_weighted_kbest_mean` vs `noensemble_kbest_mean` | `L1` | 17.378 vs 17.352 | 50.2 / 49.8 | 9.52e-01 |
| `noensemble_weighted_kbest_mean` vs `noensemble_kbest_mean` | `MSE` | 1208.582 vs 1208.224 | 50.2 / 49.8 | 6.40e-01 |
| `noensemble_top_k_median` vs `noensemble_best` | `L1` | 17.450 vs 17.455 | 50.5 / 49.5 | 4.29e-01 |
| `noensemble_top_k_median` vs `noensemble_best` | `MSE` | 1256.546 vs 1216.300 | 51.5 / 48.5 | 4.69e-01 |
| `noensemble_top_k_median` vs `noensemble_kbest_mean` | `L1` | 17.450 vs 17.352 | 49.3 / 50.7 | 2.23e-01 |
| `noensemble_top_k_median` vs `noensemble_kbest_mean` | `MSE` | 1256.546 vs 1208.224 | 49.0 / 51.0 | 1.60e-01 |

So:

- `validation_weighted` does **not** beat either `ensemble_best` or `ensemble_kbest_mean`
- `validation_median` also does **not** beat them, and is a bit worse on ensemble `MSE`
- on the no-ensemble side, neither new selector shows a clear advantage over the existing `best` / `kbest_mean` options

## Method-level picture inside each run

For all ten runs:

- Friedman is highly significant on both `L1` and `MSE`
- `KRG` is still best on average `L1`
- `IDW` is slightly best on average `MSE`
- the strongest DIP family remains the ensemble validation-selected family
- the new weighted and median selectors do not change the DIP vs KRG picture in a meaningful way

The closest case is `ensemble_best`:

- `L1`: `DIP 15.773`, `KRG 15.771`, `IDW 16.207`
- `MSE`: `DIP 1035.689`, `KRG 1004.898`, `IDW 999.204`

So on `L1`, `ensemble_best` makes DIP essentially level with KRG, while on `MSE` DIP still trails both kriging baselines.

## Current conclusion

For this `METRAQ NOX 2023` surface-selection bundle:

- **ensemble beats no ensemble**
- **validation-best beats last epoch**
- **`k_best_n = 3` does not beat `k_best_n = 1` in a meaningful way**
- **`validation_weighted` and `validation_median` do not improve over the existing ensemble selectors**
- the best current configuration is **`ensemble_best`**
