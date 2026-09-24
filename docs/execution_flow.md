# Experiment Execution Flow

One experiment is one `(sensor_group, time_window)` pair. The input contains
`T` hourly observations ending at the evaluation timestamp. Every method is
evaluated at `T-1`, at the held-out stations. Arrows below show execution order;
the following diagrams expand the DIP and interpolation calls.

The source entry point is `src/metraq_dip/experiments.py::run_experiments()`.
See [Artifacts Schema](experiments.md) for the saved arrays and CSV columns.
Diagrams use Mermaid loaded from a CDN, so rendering requires internet access.

## Session and Single Experiment

```mermaid
flowchart TD
    A["run_experiments()"] --> B["_ensure_base_files()<br/>Configuration, groups, windows and results"]
    B --> C["Select pending experiments"]
    C --> D["_run_single_experiment()<br/>Sequentially or in a worker"]
    subgraph EXP["Inside _run_single_experiment()"]
        D --> E["get_aq_backend_for_config()"]
        E --> F["collect_data()<br/>Complete T-hour window"]
        F --> G{"use_ensemble"}
        G -->|true| H["DipEnsembleOptimizer.optimize()"]
        G -->|false| I["collect_ensemble_data()<br/>One train/validation split"]
        I --> J["DipOptimizer.optimize()"]
        H --> K["optimizer.get_artifacts()"]
        J --> K
        K --> L["Combine train and validation of the first member<br/>Complete observed window and masks"]
        L --> M["Read test targets from static_data<br/>Only T-1"]
        M --> N["_denormalize_masked_channels()<br/>When normalization is enabled"]
        N --> O["_compute_masked_losses()<br/>DIP test MAE and MSE"]
        O --> P["get_interpolation_loss()<br/>GP, KRG and IDW"]
        P --> Q["_get_method_losses()<br/>Metrics identified by method name"]
        Q --> R["_build_experiment_artifacts()<br/>Prepare serialization and diagnostic histories"]
        R --> S["compute_results_data_stats()"]
        S --> U["np.savez_compressed()<br/>Save experiment .npz"]
        U --> V["Return metrics and processed=True"]
    end
    V --> W["_apply_row_result()"]
    W --> X["DataFrame.to_csv()<br/>Save results.csv"]
    D -. "On failure" .-> Y["_build_failure_record()"]
    Y --> Z["_append_failure_log()<br/>Record failure and continue"]
```

`_ensure_base_files()` loads existing groups and timestamps from `data.npz`, or
generates them with `get_spread_test_groups()` and `_get_time_windows()` when
starting a session. Completed rows are skipped according to the current resume
logic. Adding GP columns to an old CSV does not itself run GP for completed rows;
historical results require a separate backfill.

## DIP Optimization

The ensemble repeats the member block `ensemble_size` times. With
`use_ensemble=false`, the experiment calls `DipOptimizer.optimize()` directly.

```mermaid
flowchart TD
    A["DipEnsembleOptimizer.optimize()"] --> B["_collect_split_data()"]
    B --> C["collect_ensemble_data()<br/>Train/validation split and T-hour input"]
    C --> D["_create_optimizer()<br/>Create DipOptimizer"]
    D --> E["DipOptimizer.optimize()"]
    subgraph MEMBER["One DIP member"]
        E --> F["_prepare_split_tensors()"]
        F --> G["_initialize_artifacts()"]
        G --> H["_get_model()<br/>Autoencoder3D or UNet3D"]
        H --> I["_get_optimizer()<br/>Adam"]
        I --> J["_run_epoch()"]
        J --> K["_call_model()"]
        K --> L["_get_optimization_tensors()<br/>All hours or only T-1"]
        L --> M["get_losses() → _get_optimization_loss()"]
        M --> N["backward() → optimizer.step()"]
        N --> O["_call_model()<br/>Forward pass after weight update"]
        O --> P["_get_optimization_tensors() → get_losses()<br/>Validation loss"]
        P --> Q["_record_epoch()<br/>Keep T-1 surface and loss values"]
        Q -->|Next epoch| J
        Q -->|Epochs completed| R{"surface_selection"}
        R -->|last| S["Take final epoch surface"]
        R -->|Validation selectors| T["select_surface_from_validation()"]
        S --> U["_restore_surface_to_real_values()"]
        T --> U
    end
    U --> V["_get_member_artifacts()"]
    V -->|Next member| B
    V -->|Ensemble completed| W["reduce_surface_ensemble()"]
    W --> X["Return final DIP surface"]
```

`optimization_timesteps=all` applies training and validation losses to the full
window; `last` restricts both losses to the final hour. In either case, stored
epoch predictions and the final evaluation concern `T-1`.

Source: `src/metraq_dip/trainer/dip_optimizer.py` and
`src/metraq_dip/trainer/dip_ensemble_optimizer.py`.

## GP, Kriging and IDW

All interpolators receive `(T, H, W)` values and masks for one pollutant. The
experiment combines training and validation observations from the first DIP
member, giving all available stations (20 in the Madrid 24-station setup).
Held-out test measurements are supplied separately to the metric calculation.

```mermaid
flowchart TD
    A["get_interpolation_loss()"] --> B["Next method: GP, then KRG, then IDW"]
    B --> C["calculate_interpolations()"]
    C --> D["For each pollutant: construct interpolator<br/>Complete data and mask of shape T,H,W"]
    D --> E{"Interpolator"}
    E -->|GP| F["SpatioTemporalGPInterpolator.__init__()<br/>All valid observations across time<br/>_kernel() → cho_factor() → cho_solve()"]
    E -->|KRG| G["KrigingInterpolator.__init__()<br/>target_observations(): select T-1<br/>Construct OrdinaryKriging"]
    E -->|IDW| H["IdwInterpolator.__init__()<br/>target_observations(): select T-1<br/>Construct cKDTree"]
    F --> I["Interpolator.__call__() → interpolate()<br/>Predict grid at T-1"]
    G --> I
    H --> I
    I --> J["Restore known T-1 measurements<br/>at observed cells"]
    J --> K["Return surface of shape C,1,H,W"]
    K --> L["_get_numpy_metrics()<br/>MAE and MSE at test stations only"]
    L --> M{"More methods?"}
    M -->|Yes| B
    M -->|No| N["Return method metrics"]
```

GP uses all observed hours; KRG and IDW extract the last hour internally with
`target_observations()`. Spatial coordinates are grid indices and GP time
coordinates are hourly indices. GP currently uses fixed kernel parameters.

Source: `src/metraq_dip/tools/tools.py` and
`src/metraq_dip/tools/interpolator.py`.

## Persistence and Diagnostics

`_build_experiment_artifacts()` runs after all final model metrics are computed.
It combines model-space surfaces with `reduce_surface_ensemble()`, collects
selected epoch indices and builds training/validation histories with
`_build_loss_history_cube()`. `_build_test_loss_history_cube()` evaluates stored
epoch surfaces against test measurements for retrospective diagnostics; these
test losses do not select epochs or update model weights.

Saved training and validation arrays retain only `T-1`, while the interpolators
receive the complete window directly from the optimizer's in-memory member
artifacts. Final test targets come directly from `static_data`. Prediction and
evaluation therefore do not depend on the compact serialization format.
