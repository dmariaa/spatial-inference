# Experiment Results

- Generated at: `2026-03-27 13:45:19`
- Source root: `output/experiments`
- Data used per session: `config.yaml`, `data.npz`, `results.csv`
- Ignored by design: detailed `exp_*.npz` files

## 1. Configuration Differences

| Session | pollut. | norm | meteo | time_ch | dist2sens | m.skip | grp_n | grp_sz | grp_max_use | tw_mode | tw_year | tw_win_mo | tw_start_h | tw_weekend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DGX/experiment_NOX` | [12] | yes | no | yes | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [7, 8, 9] | 0.25 |
| `DGX/experiment_hours` | [7] | yes | no | no | no | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [7, 8, 9] | 0.25 |
| `DGX/experiment_range_hours` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |
| `ESCOBAR/experiment_hourrange` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |
| `ESCOBAR/experiment_skipconn` | [7] | yes | no | no | yes | yes | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `ESCOBAR/experiment_skipconn2` | [7] | yes | no | no | yes | yes | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `ESCOBAR/experiment_skipconn_hourrange.old` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |
| `ESCOBAR/experiment_traffic_data` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |
| `experiment_test_2` | [7] | no | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [8, 17] | 0.4 |
| `experiment_test_3` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [8, 17] | 0.4 |
| `experiment_test_4` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [8, 17] | 0.4 |
| `experiment_test_5` | [7] | yes | yes | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [8, 17] | 0.4 |
| `experiment_test_6` | [7] | yes | no | yes | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [8, 17] | 0.4 |
| `experiment_test_parallel` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [8, 17] | 0.4 |
| `norm/experiment_baseline` | [7] | yes | no | no | no | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |
| `norm/experiment_traffic_data` | [7] | yes | no | no | yes | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |
| `unnorm/experiment_baseline_unnorm` | [7] | no | no | no | no | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |
| `unnorm/experiment_traffic_data_unnorm` | [7] | no | no | no | no | no | 10 | 4 | 2 | random_time_windows | 2024 | 20 | [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | 0.4 |

Legend: `pollut.` = `pollutants`, `norm` = `normalize`, `meteo` = `add_meteo`, `time_ch` = `add_time_channels`, `dist2sens` = `add_distance_to_sensors`, `m.skip` = `model.skip_connections`, `grp_n` = `spread_test_groups.n_groups`, `grp_sz` = `spread_test_groups.group_size`, `grp_max_use` = `spread_test_groups.max_uses_per_sensor`, `tw_mode` = `time_windows.strategy`, `tw_year` = `time_windows.year`, `tw_win_mo` = `time_windows.windows_per_month`, `tw_start_h` = `time_windows.start_hours`, `tw_weekend` = `time_windows.weekend_fraction`

## 2. Mean L1/MSE by Model

| Session | Processed / Expected | Completion | DIP L1 mean | DIP MSE mean | KRG L1 mean | KRG MSE mean | IDW L1 mean | IDW MSE mean | Best L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `DGX/experiment_NOX` | 2400/2400 | 100.0% | 0.6046 | 0.9944 | 0.6210 | 1.0264 | 0.6459 | 1.0631 | DIP |
| `DGX/experiment_hours` | 2400/2400 | 100.0% | 0.9288 | 3.3539 | 0.9977 | 3.2143 | 1.0048 | 3.3095 | DIP |
| `DGX/experiment_range_hours` | 2400/2400 | 100.0% | 0.5222 | 1.3073 | 0.5757 | 1.2656 | 0.5873 | 1.3074 | DIP |
| `ESCOBAR/experiment_hourrange` | 2400/2400 | 100.0% | 0.4988 | 1.0666 | 0.5544 | 1.0277 | 0.5619 | 1.0724 | DIP |
| `ESCOBAR/experiment_skipconn` | 2400/2400 | 100.0% | 0.6547 | 1.9800 | 0.7040 | 1.7993 | 0.7140 | 1.8610 | DIP |
| `ESCOBAR/experiment_skipconn2` | 2400/2400 | 100.0% | 0.6016 | 1.6889 | 0.6511 | 1.5486 | 0.6506 | 1.5761 | DIP |
| `ESCOBAR/experiment_skipconn_hourrange.old` | 2400/2400 | 100.0% | 0.5235 | 1.2978 | 0.5757 | 1.2656 | 0.5873 | 1.3074 | DIP |
| `ESCOBAR/experiment_traffic_data` | 2400/2400 | 100.0% | 0.7698 | 2.0341 | 0.8498 | 2.0321 | 0.8678 | 2.0853 | DIP |
| `experiment_test_2` | 2400/2400 | 100.0% | 7.0801 | 384.3781 | 6.3769 | 243.2277 | 6.2931 | 237.0764 | IDW |
| `experiment_test_3` | 2400/2400 | 100.0% | 0.0053 | 0.0002 | 0.0032 | 0.0001 | 0.0031 | 0.0001 | IDW |
| `experiment_test_4` | 2400/2400 | 100.0% | 0.6080 | 1.5264 | 0.6629 | 1.4411 | 0.6715 | 1.5031 | DIP |
| `experiment_test_5` | 2400/2400 | 100.0% | 0.6097 | 1.7307 | 0.6511 | 1.5486 | 0.6506 | 1.5761 | DIP |
| `experiment_test_6` | 2400/2400 | 100.0% | 0.6014 | 1.6436 | 0.6511 | 1.5486 | 0.6506 | 1.5761 | DIP |
| `experiment_test_parallel` | 2400/2400 | 100.0% | 0.6007 | 1.6568 | 0.6511 | 1.5486 | 0.6506 | 1.5761 | DIP |
| `norm/experiment_baseline` | 2400/2400 | 100.0% | 0.7730 | 2.0496 | 0.8498 | 2.0321 | 0.8678 | 2.0853 | DIP |
| `norm/experiment_traffic_data` | 2400/2400 | 100.0% | 0.7698 | 2.0341 | 0.8498 | 2.0321 | 0.8678 | 2.0853 | DIP |
| `unnorm/experiment_baseline_unnorm` | 2400/2400 | 100.0% | 5.3632 | 333.8465 | 4.7612 | 157.0719 | 4.7019 | 160.9932 | IDW |
| `unnorm/experiment_traffic_data_unnorm` | 2400/2400 | 100.0% | 5.3345 | 331.5199 | 4.7612 | 157.0719 | 4.7019 | 160.9932 | IDW |

## 3. Friedman Test (DIP vs KRG vs IDW)

| Session | Samples | Friedman stat | p-value | Significant (p < 0.05) | DIP mean rank | KRG mean rank | IDW mean rank |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `DGX/experiment_NOX` | 2400 | 281.0033 | 9.570e-62 | yes | 1.8700 | 1.8508 | 2.2792 |
| `DGX/experiment_hours` | 2400 | 246.5689 | 2.872e-54 | yes | 1.7383 | 2.1302 | 2.1315 |
| `DGX/experiment_range_hours` | 2400 | 486.8033 | 1.959e-106 | yes | 1.6425 | 2.1042 | 2.2533 |
| `ESCOBAR/experiment_hourrange` | 2400 | 408.0365 | 2.489e-89 | yes | 1.6721 | 2.0981 | 2.2298 |
| `ESCOBAR/experiment_skipconn` | 2400 | 437.1324 | 1.196e-95 | yes | 1.6700 | 2.0681 | 2.2619 |
| `ESCOBAR/experiment_skipconn2` | 2400 | 320.5300 | 2.499e-70 | yes | 1.7108 | 2.0808 | 2.2083 |
| `ESCOBAR/experiment_skipconn_hourrange.old` | 2400 | 484.0358 | 7.815e-106 | yes | 1.6442 | 2.1012 | 2.2546 |
| `ESCOBAR/experiment_traffic_data` | 2400 | 543.5972 | 9.107e-119 | yes | 1.6196 | 2.1221 | 2.2583 |
| `experiment_test_2` | 2400 | 216.5004 | 9.717e-48 | yes | 1.7808 | 2.0144 | 2.2048 |
| `experiment_test_3` | 2400 | 815.4700 | 8.374e-178 | yes | 2.4633 | 1.6744 | 1.8623 |
| `experiment_test_4` | 2400 | 455.2504 | 1.392e-99 | yes | 1.6604 | 2.0785 | 2.2610 |
| `experiment_test_5` | 2400 | 289.4633 | 1.393e-63 | yes | 1.7275 | 2.0683 | 2.2042 |
| `experiment_test_6` | 2400 | 319.7725 | 3.650e-70 | yes | 1.7108 | 2.0821 | 2.2071 |
| `experiment_test_parallel` | 2400 | 340.7425 | 1.020e-74 | yes | 1.7004 | 2.0892 | 2.2104 |
| `norm/experiment_baseline` | 2400 | 509.5485 | 2.254e-111 | yes | 1.6333 | 2.1108 | 2.2558 |
| `norm/experiment_traffic_data` | 2400 | 543.5972 | 9.107e-119 | yes | 1.6196 | 2.1221 | 2.2583 |
| `unnorm/experiment_baseline_unnorm` | 2400 | 171.4396 | 5.920e-38 | yes | 1.7971 | 2.0321 | 2.1708 |
| `unnorm/experiment_traffic_data_unnorm` | 2400 | 190.5891 | 4.112e-42 | yes | 1.7875 | 2.0300 | 2.1825 |

## Session Metadata

| Session | Pollutants | Epochs | Ensemble | Hours | LR | Norm | Meteo | TimeCh | Coord | Dist2Sens | Sensor groups | Time windows | Date range |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `DGX/experiment_NOX` | `[12]` | 250 | 5 | 24 | 0.01 | yes | no | yes | no | yes | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-30 08:00:00 |
| `DGX/experiment_hours` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | no | 10 | 240 | 2024-01-01 08:00:00 -> 2024-12-31 09:00:00 |
| `DGX/experiment_range_hours` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |
| `ESCOBAR/experiment_hourrange` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-01 14:00:00 -> 2024-12-28 11:00:00 |
| `ESCOBAR/experiment_skipconn` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `ESCOBAR/experiment_skipconn2` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |
| `ESCOBAR/experiment_skipconn_hourrange.old` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-05 06:00:00 -> 2024-12-28 12:00:00 |
| `ESCOBAR/experiment_traffic_data` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-04 10:00:00 -> 2024-12-31 13:00:00 |
| `experiment_test_2` | `[7]` | 250 | 5 | 24 | 0.01 | no | no | no | no | yes | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_3` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_4` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-02 08:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_5` | `[7]` | 250 | 5 | 24 | 0.01 | yes | yes | no | no | yes | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_6` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | yes | no | yes | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |
| `experiment_test_parallel` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-01 17:00:00 -> 2024-12-31 17:00:00 |
| `norm/experiment_baseline` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | no | 10 | 240 | 2024-01-04 10:00:00 -> 2024-12-31 13:00:00 |
| `norm/experiment_traffic_data` | `[7]` | 250 | 5 | 24 | 0.01 | yes | no | no | no | yes | 10 | 240 | 2024-01-04 10:00:00 -> 2024-12-31 13:00:00 |
| `unnorm/experiment_baseline_unnorm` | `[7]` | 250 | 5 | 24 | 0.01 | no | no | no | no | no | 10 | 240 | 2024-01-04 10:00:00 -> 2024-12-31 13:00:00 |
| `unnorm/experiment_traffic_data_unnorm` | `[7]` | 250 | 5 | 24 | 0.01 | no | no | no | no | no | 10 | 240 | 2024-01-04 10:00:00 -> 2024-12-31 13:00:00 |

## Notes

- Means are computed from rows where `processed = True`.
- Friedman test is computed with `DIP_L1Loss`, `KRG_L1Loss`, and `IDW_L1Loss` from processed rows.
- Mean rank interpretation: lower is better.
- Direct comparison across sessions can be affected by configuration changes (for example normalization and extra channels).
