# Experiments

`src/metraq_dip/experiments.py` runs one configured experiment session.
The session folder is the directory containing the `config.yaml` passed to `run_experiments(config_file=...)`.
That folder is reused for generated grid data, the aggregate result CSV, failure logs, and one compressed `.npz` artifact per processed `(sensor_group, time_window)` row.

The number of sensor groups, sensors per group, and time windows is config-driven. Current sessions use either `random_time_windows` or `all_time_windows` plus `spread_test_groups`; the runner does not hard-code a fixed 2400-row grid.

## Output Folder Structure

| File / pattern | Count | Created by | Purpose |
| --- | ---: | --- | --- |
| `config.yaml` | 1 | user/session setup | Defines session configuration. |
| `data.npz` | 1 | `_ensure_base_files()` | Stores generated `test_sensors` and `time_windows`. Reused on resume. |
| `results.csv` | 1 | `_ensure_base_files()` and `run_experiments()` | Stores one summary row per scheduled `(sensor_group, time_window)` experiment. Updated after each successful run. |
| `exp_<sensor_group>_<timestamp>.npz` | up to one per row | `_run_single_experiment()` | Stores detailed arrays, masks, predictions, losses, and normalization metadata for one experiment. |
| `failures.log` | 0 or 1 | `run_experiments()` | Written when one or more jobs fail. Removed at the start of a run. |
| `video.html` | 0 or 1 | `src/metraq_dip/utils/plot_surface_video.py` | Optional Plotly visualization for the first experiment artifact in the session folder. |
| `video_unnormalized.html` | 0 or 1 | `src/metraq_dip/utils/plot_surface_video.py --unnormalize` | Optional Plotly visualization in original units when saved normalization stats are available. |

## `config.yaml`

This file stores session-level configuration.
Runtime values such as `date`, `validation_sensors`, and `test_sensors` are injected per experiment and are not written back into `config.yaml`.

| Key | Type | Meaning |
| --- | --- | --- |
| `aq_dataset` | `string` | Dataset family used by the air-quality backend. |
| `aq_backend` | `string` | Concrete air-quality backend. |
| `pollutants` | `list[int]` | Pollutant channel ids used in the experiment. |
| `hours` | `int` | Length of the temporal input window. |
| `epochs` | `int` | Training iterations per ensemble member. |
| `ensemble_size` | `int` | Number of independent optimizer members. |
| `lr` | `float` | Adam learning rate. |
| `optimization_loss` | `mae`, `mse`, or `rmse` | Loss used for optimization. |
| `optimization_timesteps` | `all` or `last` | Pollutant target timesteps included in train/validation loss. `all` keeps the current full-window supervision; `last` optimizes only the final hour. Defaults to `all`. |
| `surface_selection` | `validation` or `last` | Chooses the final DIP surface from the best validation epochs or the last epoch. Defaults to `validation`. |
| `normalize` | `bool` | Enables dataset normalization when true. |
| `add_meteo` | `bool` | Adds meteorological input channels. |
| `add_time_channels` | `bool` | Adds time-derived channels. |
| `add_coordinates` | `bool` | Adds coordinate channels. |
| `add_distance_to_sensors` | `bool` | Adds distance-to-sensor channels. |
| `add_traffic_data` | `bool` | Adds traffic input channels. |
| `use_ensemble` | `bool` | Runs the ensemble optimizer when true; runs one direct DIP optimizer member when false. Defaults to true. |
| `k_best_n` | `int`, optional | Number of lowest-validation epochs averaged per ensemble member; defaults to `10` when omitted. |
| `model.architecture` | `autoencoder` or `unet` | DIP backbone. |
| `model.base_channels` | `int` | Base model width. |
| `model.levels` | `int` | Encoder/decoder depth. |
| `model.preserve_time` | `bool` | Controls whether the model preserves the time axis. |
| `model.learned_upsampling` | `bool` | Enables learned upsampling. |
| `model.skip_connections` | `bool`, autoencoder only | Enables autoencoder skip connections. |
| `spread_test_groups.*` | object | Controls held-out sensor group generation. |
| `random_time_windows.*` | object, optional | Generates sampled windows for a year. |
| `all_time_windows.*` | object, optional | Generates all windows for selected start hours. |

## `data.npz`

This file defines the scheduled experiment grid for the session.

| Array | Type | Typical shape | Description |
| --- | --- | --- | --- |
| `test_sensors` | integer array | `(G, S)` | Matrix of held-out sensor groups. Each row becomes one `test_sensors` list. |
| `time_windows` | datetime-like object array | `(T,)` | Sequence of experiment timestamps. Each value becomes the per-experiment `date`. |

## `results.csv`

This is the aggregate session table.
It contains one row per scheduled `(sensor_group, time_window)` pair and is safe for resume/restart workflows.

| Column | Type | Description |
| --- | --- | --- |
| `time_window` | datetime-like string | Experiment timestamp. |
| `sensor_group` | string | Hyphen-joined held-out sensor ids. |
| `processed` | bool-like | True after the experiment row finishes successfully. |
| `DIP_L1Loss` | float | Final DIP MAE on the test mask, in original units when normalization is enabled. |
| `DIP_MSELoss` | float | Final DIP MSE on the test mask, in original units when normalization is enabled. |
| `KRG_L1Loss` | float | Kriging MAE baseline. |
| `KRG_MSELoss` | float | Kriging MSE baseline. |
| `IDW_L1Loss` | float | Inverse-distance weighting MAE baseline. |
| `IDW_MSELoss` | float | Inverse-distance weighting MSE baseline. |
| `data_mean` | float | Mean observed value over train and validation cells. |
| `data_median` | float | Median observed value over train and validation cells. |
| `data_max` | float | Max observed value over train and validation cells. |
| `data_std` | float | Standard deviation over train and validation cells. |
| `data_p90_p10` | float | Difference between the 90th and 10th observed percentiles. |

## `exp_<sensor_group>_<timestamp>.npz`

This file stores the detailed output of a single experiment.
The filename is generated by `get_experiment_name(sensor_group_key, time_window)`.

| Part | Example | Meaning |
| --- | --- | --- |
| Prefix | `exp_` | Marks the file as a single experiment artifact. |
| `<sensor_group>` | `28079004-28079016-28079036-28079050` | Same group key stored in `results.csv.sensor_group`. |
| `<timestamp>` | `20240102T080000` | Experiment time formatted as `%Y%m%dT%H%M%S`. |

### Shape Reference

| Symbol | Meaning |
| --- | --- |
| `K` | Number of optimizer members: `ensemble_size` when `use_ensemble` is true, otherwise `1`. |
| `C` | number of pollutant channels |
| `E` | `epochs` |
| `H` | grid height |
| `W` | grid width |

### Canonical Arrays

These fields are written by the current `DipEnsembleOptimizer` code path.

| Array | Type | Typical shape | Description |
| --- | --- | --- | --- |
| `train_data` | `float32` | `(K, C, 1, H, W)` | Training target values for the final time step. |
| `val_data` | `float32` | `(K, C, 1, H, W)` | Validation target values for the final time step. |
| `test_data` | `float32` | `(K, C, 1, H, W)` | Held-out test values. |
| `train_mask` | `bool` | `(K, C, 1, H, W)` | Training observation mask. |
| `val_mask` | `bool` | `(K, C, 1, H, W)` | Validation observation mask. |
| `test_mask` | `bool` | `(K, C, 1, H, W)` | Test observation mask. |
| `train_output` | `float32` | `(H, W)` or `(C, H, W)` | Final selected prediction surface in model space. |
| `train_output_real` | `float32` | `(H, W)` or `(C, H, W)` | Final selected prediction surface in real/original space. |
| `val_min_idx` | integer array | `(K, k_best_n)` | Selected best validation epoch indices for each ensemble member. |
| `train_k_output` | `float32` | `(K, C, E, H, W)` | Prediction trace for all ensemble members and epochs. |
| `train_k_loss` | `float32` | `(K, C, E, 2)` | Training loss trace. Last axis is `[L1Loss, MSELoss]`. |
| `val_k_loss` | `float32` | `(K, C, E, 2)` | Validation loss trace. Last axis is `[L1Loss, MSELoss]`. |
| `test_k_loss` | `float32` | `(K, C, E, 2)` | Test loss trace recomputed across epoch outputs. |
| `normalization_stats` | object, optional | mapping | Saved normalization statistics keyed by pollutant id when `normalize = true`. |

### Older Variants

Some existing sample folders were written by older or compact code paths. Utility code should continue to tolerate these variants when reading historical artifacts.

| Variant | Meaning |
| --- | --- |
| `val_min_idx = None` | Older files were written before best-epoch indices were stored. |
| `test_k_loss` missing | Older files only stored training and validation loss traces. |
| `train_output_real` missing | Older files only stored `train_output`. |
| `normalization_stats` missing | Older normalized artifacts may need repair before unnormalized plots can be rendered. |
| `train_data`, `val_data`, `test_data` missing | Some compact files store only masks, final output, and loss traces. |
| masks shaped as `(H, W)` | Compact files omit ensemble/channel axes. |
| `train_output` shaped as `(1, H, W)` | Compact files keep a singleton channel axis. |

## Visualizations

`src/metraq_dip/utils/plot_surface_video.py` reads the first `exp_*.npz` file in a session folder and writes a standalone Plotly HTML animation.

```powershell
uv run python src/metraq_dip/utils/plot_surface_video.py plot output/experiments/<session>
uv run python src/metraq_dip/utils/plot_surface_video.py plot --unnormalize output/experiments/<session>
```

The visualization includes the averaged prediction surface across epochs plus train, validation, and test observation markers.

## Generated Result Pages

The old cross-session `experiment_results.md` page has been removed from the docs. If a fresh cross-session summary is needed, regenerate it from the utility script:

```powershell
uv run python src/metraq_dip/utils/generate_experiment_results_doc.py output/experiments --output-file docs/experiment_results.md
```

Then add it back to `mkdocs.yml` if it should be published.
