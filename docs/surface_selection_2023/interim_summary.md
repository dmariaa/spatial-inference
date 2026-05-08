# Surface Selection 2023 Summary

## Current status

All six runs are now finished and cleanly comparable:

- `ensemble_best`
- `ensemble_kbest_mean`
- `ensemble_last`
- `noensemble_best`
- `noensemble_kbest_mean`
- `noensemble_last`

All six folders currently share the same `data.npz`, and the six `config.yaml` files are aligned apart from the intended differences:

- `use_ensemble` / `ensemble_size`
- `surface_selection`
- `k_best_n`

## DIP ranking

| Experiment | DIP L1 | DIP MSE |
| --- | ---: | ---: |
| `ensemble_best` | **15.773** | **1035.689** |
| `ensemble_kbest_mean` | 15.801 | 1039.147 |
| `ensemble_last` | 16.745 | 1135.991 |
| `noensemble_best` | 17.455 | 1216.300 |
| `noensemble_kbest_mean` | 17.352 | 1208.224 |
| `noensemble_last` | 18.672 | 1357.232 |

Current DIP ordering:

1. `ensemble_best`
2. `ensemble_kbest_mean`
3. `ensemble_last`
4. `noensemble_kbest_mean`
5. `noensemble_best`
6. `noensemble_last`

## Main takeaways

- Using the **best validation surface** is better than taking the **last epoch**.
- Using an **ensemble** is better than not using one.
- `k_best_n = 3` does **not** improve over the single best validation surface in the ensemble case.
- In the no-ensemble case, `k_best_n = 3` is only slightly better than `k_best_n = 1`, and the difference is not statistically clear.
- Overall, **`ensemble_best` is the strongest DIP setup**.

## Paired DIP comparisons

| Comparison | Metric | Better mean | Win split | Wilcoxon p |
| --- | --- | ---: | ---: | ---: |
| `ensemble_best` vs `ensemble_kbest_mean` | `L1` | 15.773 vs 15.801 | 49.3 / 50.7 | 5.96e-01 |
| `ensemble_best` vs `ensemble_kbest_mean` | `MSE` | 1035.689 vs 1039.147 | 48.8 / 51.2 | 4.43e-01 |
| `ensemble_best` vs `ensemble_last` | `L1` | 15.773 vs 16.745 | 57.8 / 42.2 | 2.56e-16 |
| `ensemble_best` vs `ensemble_last` | `MSE` | 1035.689 vs 1135.991 | 57.4 / 42.6 | 2.97e-11 |
| `ensemble_best` vs `noensemble_best` | `L1` | 15.773 vs 17.455 | 61.4 / 38.6 | 7.52e-43 |
| `ensemble_best` vs `noensemble_best` | `MSE` | 1035.689 vs 1216.300 | 61.8 / 38.2 | 3.63e-43 |
| `ensemble_best` vs `noensemble_kbest_mean` | `L1` | 15.773 vs 17.352 | 60.3 / 39.7 | 3.70e-35 |
| `ensemble_best` vs `noensemble_kbest_mean` | `MSE` | 1035.689 vs 1208.224 | 60.4 / 39.6 | 6.98e-35 |
| `ensemble_best` vs `noensemble_last` | `L1` | 15.773 vs 18.672 | 68.5 / 31.5 | 6.00e-91 |
| `ensemble_best` vs `noensemble_last` | `MSE` | 1035.689 vs 1357.232 | 69.1 / 30.9 | 1.04e-81 |
| `ensemble_kbest_mean` vs `ensemble_last` | `L1` | 15.801 vs 16.745 | 58.0 / 42.0 | 5.78e-17 |
| `ensemble_kbest_mean` vs `ensemble_last` | `MSE` | 1039.147 vs 1135.991 | 57.7 / 42.3 | 4.27e-11 |
| `ensemble_kbest_mean` vs `noensemble_best` | `L1` | 15.801 vs 17.455 | 61.9 / 38.1 | 3.25e-44 |
| `ensemble_kbest_mean` vs `noensemble_best` | `MSE` | 1039.147 vs 1216.300 | 62.7 / 37.3 | 2.27e-44 |
| `ensemble_kbest_mean` vs `noensemble_kbest_mean` | `L1` | 15.801 vs 17.352 | 60.8 / 39.2 | 8.65e-39 |
| `ensemble_kbest_mean` vs `noensemble_kbest_mean` | `MSE` | 1039.147 vs 1208.224 | 60.5 / 39.5 | 2.71e-35 |
| `ensemble_kbest_mean` vs `noensemble_last` | `L1` | 15.801 vs 18.672 | 69.4 / 30.6 | 7.70e-98 |
| `ensemble_kbest_mean` vs `noensemble_last` | `MSE` | 1039.147 vs 1357.232 | 69.1 / 30.9 | 7.11e-84 |
| `ensemble_last` vs `noensemble_best` | `L1` | 16.745 vs 17.455 | 54.5 / 45.5 | 1.36e-08 |
| `ensemble_last` vs `noensemble_best` | `MSE` | 1135.991 vs 1216.300 | 54.5 / 45.5 | 2.18e-10 |
| `ensemble_last` vs `noensemble_kbest_mean` | `L1` | 16.745 vs 17.352 | 53.1 / 46.9 | 4.39e-05 |
| `ensemble_last` vs `noensemble_kbest_mean` | `MSE` | 1135.991 vs 1208.224 | 54.6 / 45.4 | 9.49e-09 |
| `ensemble_last` vs `noensemble_last` | `L1` | 16.745 vs 18.672 | 63.8 / 36.2 | 3.12e-54 |
| `ensemble_last` vs `noensemble_last` | `MSE` | 1135.991 vs 1357.232 | 64.4 / 35.6 | 1.07e-54 |
| `noensemble_best` vs `noensemble_kbest_mean` | `L1` | 17.455 vs 17.352 | 48.5 / 51.5 | 8.47e-02 |
| `noensemble_best` vs `noensemble_kbest_mean` | `MSE` | 1216.300 vs 1208.224 | 48.0 / 52.0 | 8.56e-02 |
| `noensemble_best` vs `noensemble_last` | `L1` | 17.455 vs 18.672 | 56.6 / 43.4 | 1.20e-13 |
| `noensemble_best` vs `noensemble_last` | `MSE` | 1216.300 vs 1357.232 | 57.2 / 42.8 | 3.21e-11 |
| `noensemble_kbest_mean` vs `noensemble_last` | `L1` | 17.352 vs 18.672 | 58.8 / 41.2 | 2.84e-21 |
| `noensemble_kbest_mean` vs `noensemble_last` | `MSE` | 1208.224 vs 1357.232 | 58.8 / 41.2 | 3.57e-17 |

The two places where the difference is *not* clear are:

- `ensemble_best` vs `ensemble_kbest_mean`
- `noensemble_best` vs `noensemble_kbest_mean`

So averaging the best 3 surfaces does not show a reliable gain over the single best validation surface.

## Method-level picture inside each run

For each of the six runs:

- Friedman is highly significant on both `L1` and `MSE`
- `KRG` is best on average `L1`
- `IDW` is slightly best on average `MSE`
- the two ensemble runs selected from validation (`ensemble_best` and `ensemble_kbest_mean`) bring DIP much closest to the baselines

The closest case is `ensemble_best`:

- `L1`: `DIP 15.773`, `KRG 15.771`, `IDW 16.207`
- `MSE`: `DIP 1035.689`, `KRG 1004.898`, `IDW 999.204`

So on `L1`, `ensemble_best` makes DIP essentially level with KRG, while on `MSE` DIP still trails both kriging baselines.

## Current conclusion

For this `METRAQ NOX 2023` surface-selection bundle:

- **ensemble beats no ensemble**
- **validation-best beats last epoch**
- **`k_best_n = 3` does not beat `k_best_n = 1` in a meaningful way**
- the best current configuration is **`ensemble_best`**
