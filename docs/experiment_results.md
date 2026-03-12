# Experiment Results

- Generated at: `2026-03-12 10:23:40`
- Source root: `output/experiments`
- Data used per session: `config.yaml`, `data.npz`, `results.csv`
- Ignored by design: detailed `exp_*.npz` files

## Session Comparison

| Session | Processed / Expected | Completion | Norm | Meteo | TimeCh | Dist2Sens | DIP L1 mean | KRG L1 mean | IDW L1 mean | DIP vs KRG | DIP vs IDW | DIP MSE mean | KRG MSE mean | IDW MSE mean | Best L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `experiment_test` | 2400/2400 | 100.0% | no | no | no | no | 7.0831 | 6.3769 | 6.2931 | -11.1% | -12.6% | 384.7295 | 243.2277 | 237.0764 | IDW |
| `experiment_test_2` | 2400/2400 | 100.0% | no | no | no | yes | 7.0801 | 6.3769 | 6.2931 | -11.0% | -12.5% | 384.3781 | 243.2277 | 237.0764 | IDW |
| `experiment_test_3` | 2400/2400 | 100.0% | yes | no | no | yes | 0.0053 | 0.0032 | 0.0031 | -66.6% | -68.8% | 0.0002 | 0.0001 | 0.0001 | IDW |
| `experiment_test_4` | 2400/2400 | 100.0% | yes | no | no | yes | 0.6080 | 0.6629 | 0.6715 | 8.3% | 9.5% | 1.5264 | 1.4411 | 1.5031 | DIP |
| `experiment_test_5` | 2400/2400 | 100.0% | yes | yes | no | yes | 0.6097 | 0.6511 | 0.6506 | 6.4% | 6.3% | 1.7307 | 1.5486 | 1.5761 | DIP |
| `experiment_test_6` | 1044/2400 | 43.5% | yes | no | yes | yes | 0.5924 | 0.6495 | 0.6423 | 8.8% | 7.8% | 1.5325 | 1.4593 | 1.4482 | DIP |
| `experiment_test_parallel` | 2400/2400 | 100.0% | yes | no | no | yes | 0.6007 | 0.6511 | 0.6506 | 7.7% | 7.7% | 1.6568 | 1.5486 | 1.5761 | DIP |

## DIP Ranking (Complete Sessions Only)

### By DIP L1 mean (lower is better)

1. `experiment_test_3`: 0.0053
2. `experiment_test_parallel`: 0.6007
3. `experiment_test_4`: 0.6080
4. `experiment_test_5`: 0.6097
5. `experiment_test_2`: 7.0801
6. `experiment_test`: 7.0831

### By DIP MSE mean (lower is better)

1. `experiment_test_3`: 0.0002
2. `experiment_test_4`: 1.5264
3. `experiment_test_parallel`: 1.6568
4. `experiment_test_5`: 1.7307
5. `experiment_test_2`: 384.3781
6. `experiment_test`: 384.7295

## Session Metadata

| Session | Pollutants | Epochs | Ensemble | Hours | Sensor groups | Time windows | Date range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `experiment_test` | `[7]` | 250 | 5 | 24 | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_2` | `[7]` | 250 | 5 | 24 | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_3` | `[7]` | 250 | 5 | 24 | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_4` | `[7]` | 250 | 5 | 24 | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_5` | `[7]` | 250 | 5 | 24 | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_6` | `[7]` | 250 | 5 | 24 | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_parallel` | `[7]` | 250 | 5 | 24 | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |

## Notes

- Means are computed from rows where `processed = True`.
- `DIP vs KRG` and `DIP vs IDW` are relative L1 changes: positive means DIP is better.
- Direct comparison across sessions can be affected by configuration changes (for example normalization and extra channels).
