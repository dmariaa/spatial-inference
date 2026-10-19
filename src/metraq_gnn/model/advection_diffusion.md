# AdvectionDiffusionGNN

Independent current-hour model. SensorInitializer uses PyG MessagePassing:
shared neural vector messages, distance decay, and a nonnegative downwind
advection gate. A learned output correction is added to a distance-weighted
initial field; no previous kernel checkpoint is required. Initialization and
local coefficients train jointly.

Eight-neighbour local updates combine symmetric diffusion with directed upwind
transport. Every transfer is subtracted from the source and added to the
destination. The step is bounded by outgoing rates, preserving positivity for
nonnegative concentrations. Closed boundaries conserve the grid sum before
visible observations are restored. Restoring sensors exchanges mass with the
observational constraints. There are no emission or chemical reaction terms.

Transport operates in physical concentration units, then returns normalized
outputs for the shared trainer. Coefficients and steps describe reconstruction;
they are not calibrated atmospheric diffusivities or physical elapsed seconds.
Available current wind u/v is interpolated with distance weights to the grid;
observed vectors are preserved. Missing wind everywhere gives zero advection.
All available wind cache cells are exposed independently of AQ partitions.
Training/evaluation require observed_only=True wind provenance. The current
source-cache format stores wind at AQ cells; it does not retain arbitrary
external weather-station locations. Grid interpolation is an estimate and is
not treated as an additional measured wind observation.
The first implementation requires jointly available pollutant channels; use
one pollutant for initial experiments. Geometry and batches use a fixed grid.

Train using the existing TRAIN/validation-only infrastructure:

```powershell
uv run python scripts/train/train_advection_diffusion.py --cache <aq-cache> --wind-cache <wind-cache> --partition-summary <reference-summary.json> --output <new-run-directory> --propagation-steps 4 --device cpu
```

Use scripts/evaluation/evaluate_cached_model.py for a frozen TEST evaluation.
It detects architecture from config.json. No TEST evaluation occurs in training.
The legacy 24-hour eligibility convention is retained for matching windows;
only the final hour is used by this model.
