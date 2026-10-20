"""Frozen, validation-only component ablations; never train or evaluate TEST."""
import json
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from metraq_gnn.data import load_aq_sensor_cache
from metraq_gnn.data.kernel_window import CachedKernelDataset
from metraq_gnn.data.wind_cache import file_sha256, load_wind_source_cache
from metraq_gnn.evaluation import collect_sensor_predictions, summarize_all_predictions, summarize_sensor_predictions
from metraq_gnn.model import AdvectionDiffusionGNN


VARIANTS = {
    "idw_initial": dict(initialization="idw", steps=0, diffusion=False, advection=False),
    "learned_initial": dict(initialization="learned", steps=0, diffusion=False, advection=False),
    "diffusion_only": dict(initialization="learned", steps=None, diffusion=True, advection=False),
    "advection_only": dict(initialization="learned", steps=None, diffusion=False, advection=True),
    "full": dict(initialization="learned", steps=None, diffusion=True, advection=True),
}


class TracedModel(nn.Module):
    def __init__(self, model, options):
        super().__init__()
        self.model, self.options, self.records = model, options, []

    def forward(self, data):
        graphs = getattr(data, "num_graphs", 1)
        nodes = len(data.pos) // graphs

        def trace(graph, step, before, transported, anchored):
            start, stop = graph * nodes, (graph + 1) * nodes
            target = data.target_mask[start:stop].bool()
            truth = data.y[start:stop] * self.model.std + self.model.mean
            error = anchored[target] - truth[target]
            pos = data.pos[start:stop]
            boundary = ((pos == pos.amin(0)) | (pos == pos.amax(0))).any(-1)
            metrics = torch.stack((
                error.abs().sum(), error.square().sum(), truth[target].abs().sum(),
                before.sum(), transported.sum(), anchored.sum(),
                (anchored - transported).abs().sum(),
                (anchored - before).abs().mean(),
                before[boundary].sum(), transported[boundary].sum(), anchored[boundary].sum(),
            )).detach().cpu().tolist()
            names = ["absolute_error", "squared_error", "absolute_target", "before_sum",
                     "transported_sum", "anchored_sum", "anchor_absolute_change",
                     "field_mean_absolute_change", "boundary_before_sum",
                     "boundary_transported_sum", "boundary_anchored_sum"]
            self.records.append(dict(window_end=pd.Timestamp(int(data.window_end_ns.reshape(-1)[graph])),
                step=step, target_count=int(target.sum()), **dict(zip(names, metrics))))

        return self.model(data, trace=trace, **self.options)


@click.command(context_settings={"show_default": True})
@click.option("--run-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--wind-cache", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--device", default="cpu")
@click.option("--batch-size", default=16, type=click.IntRange(min=1))
@click.option("--max-windows", type=click.IntRange(min=1), help="Smoke check only; omit for the full validation.")
def main(run_dir, cache, wind_cache, output, device, batch_size, max_windows):
    started = perf_counter()
    if output.exists():
        raise click.ClickException("output already exists")
    paths = [run_dir / name for name in ["config.json", "summary.json", "partition.npz", "best-model.pt"]]
    hashes = {str(p): file_sha256(p) for p in paths}
    config = json.loads(paths[0].read_text())
    training = json.loads(paths[1].read_text())
    checkpoint = torch.load(paths[3], map_location="cpu", weights_only=True)
    if config["architecture"] != "advection-diffusion" or checkpoint["epoch"] != training["best_epoch"]:
        raise click.ClickException("unexpected architecture or checkpoint epoch")
    if checkpoint["validation_loss"] != training["best_validation_mae"]:
        raise click.ClickException("checkpoint selection differs")
    aq = load_aq_sensor_cache(cache)
    with np.load(paths[2], allow_pickle=False) as partition:
        np.testing.assert_array_equal(partition["sensor_ids"], aq.sensor_ids)
        np.testing.assert_array_equal(partition["sensor_node_indices"], aq.sensor_node_indices)
        train, validation, test = [partition[key].astype(bool) for key in
                                  ["train_mask", "validation_mask", "test_mask"]]
    if any((a & b).any() for a, b in [(train, validation), (train, test), (validation, test)]):
        raise click.ClickException("overlapping partitions")
    np.testing.assert_array_equal(np.flatnonzero(train), config["train_nodes"])
    np.testing.assert_array_equal(np.flatnonzero(validation), config["validation_nodes"])
    wind = load_wind_source_cache(wind_cache)
    if wind.metadata["fingerprint"] != config["wind_cache_fingerprint"] or not wind.metadata["observed_only"]:
        raise click.ClickException("wind provenance differs")
    dataset = CachedKernelDataset(cache=aq, wind_cache=wind, hours=config["hours"],
        seed=config["seed"], normalization_mean=config["normalization_mean"],
        normalization_std=config["normalization_std"], context_sensor_mask=train,
        forbidden_context_mask=validation | test, target_sensor_mask=validation,
        split_start="2023-01-01", split_end="2023-12-31 23:00",
        wind_context_only=config["wind_context_only"])
    if len(dataset) != config["validation_windows"]:
        raise click.ClickException("validation window count differs")
    selected = Subset(dataset, range(min(max_windows, len(dataset)))) if max_windows else dataset
    loader = DataLoader(selected, batch_size=batch_size, shuffle=False)
    model = AdvectionDiffusionGNN(channels=len(aq.magnitude_ids), cell_size_m=config["cell_size_m"],
        propagation_steps=config["propagation_steps"], normalization_mean=config["normalization_mean"],
        normalization_std=config["normalization_std"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    output.mkdir(parents=True)
    summaries, traces, per_sensor, predictions_all = [], [], [], []
    reference_keys = None
    for variant, options in VARIANTS.items():
        click.echo(f"{variant}: {len(selected)} validation hours")
        model.propagation_steps = config["propagation_steps"] if options["steps"] is None else options["steps"]
        wrapper = TracedModel(model, {k: v for k, v in options.items() if k != "steps"})
        predictions = collect_sensor_predictions(wrapper, loader,
            mean=np.asarray(config["normalization_mean"], dtype=np.float32),
            std=np.asarray(config["normalization_std"], dtype=np.float32), device=device,
            sensor_ids=aq.sensor_ids, sensor_node_indices=aq.sensor_node_indices,
            node_positions=aq.graph.pos.numpy(), magnitude_ids=aq.magnitude_ids)
        keys = ["window_end", "sensor_id", "magnitude_id"]
        if predictions.duplicated(keys).any() or not predictions.window_end.dt.year.eq(2023).all():
            raise click.ClickException("invalid validation keys")
        if not np.isfinite(predictions[["target", "prediction"]].to_numpy()).all():
            raise click.ClickException("nonfinite predictions")
        actual_keys = predictions[keys + ["target"]]
        if reference_keys is None:
            reference_keys = actual_keys
        else:
            pd.testing.assert_frame_equal(actual_keys, reference_keys)
        metrics = {key.replace("test_", "validation_", 1): value
                   for key, value in summarize_all_predictions(predictions).items()}
        metrics["validation_mae_normalized"] = metrics["validation_mae_physical"] / float(config["normalization_std"][0])
        if variant == "full" and max_windows is None:
            np.testing.assert_allclose(metrics["validation_mae_normalized"], training["best_validation_mae"],
                                       rtol=2e-6, atol=2e-7)
        summaries.append(dict(variant=variant, **metrics))
        traced = pd.DataFrame(wrapper.records)
        traced["variant"] = variant
        traces.append(traced)
        sensors = summarize_sensor_predictions(predictions)
        sensors.insert(0, "variant", variant)
        per_sensor.append(sensors)
        predictions.insert(0, "variant", variant)
        predictions_all.append(predictions)
        click.echo(json.dumps(summaries[-1]))
    trace = pd.concat(traces, ignore_index=True)
    trace["transport_drift"] = trace.transported_sum - trace.before_sum
    trace["anchor_net_change"] = trace.anchored_sum - trace.transported_sum
    trace["boundary_transport_change"] = trace.boundary_transported_sum - trace.boundary_before_sum
    rows = []
    for (variant, step), frame in trace.groupby(["variant", "step"]):
        rows.append(dict(variant=variant, step=int(step), hours=len(frame),
            mae=frame.absolute_error.sum()/frame.target_count.sum(),
            rmse=np.sqrt(frame.squared_error.sum()/frame.target_count.sum()),
            wape=frame.absolute_error.sum()/frame.absolute_target.sum(),
            transport_abs_drift_max=frame.transport_drift.abs().max(),
            anchor_net_change_mean=frame.anchor_net_change.mean(),
            anchor_absolute_change_mean=frame.anchor_absolute_change.mean(),
            boundary_transport_change_mean=frame.boundary_transport_change.mean(),
            field_mean_absolute_change=frame.field_mean_absolute_change.mean()))
    pd.DataFrame(summaries).to_csv(output / "components.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "by-step.csv", index=False)
    trace.to_csv(output / "traces.csv", index=False)
    pd.concat(per_sensor).to_csv(output / "by-sensor.csv", index=False)
    pd.concat(predictions_all).to_csv(output / "predictions.csv", index=False)
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise click.ClickException("original artifact changed")
    (output / "summary.json").write_text(json.dumps(dict(validation_only=True, test_evaluated=False,
        validation_hours=len(selected), checkpoint_epoch_one_based=checkpoint["epoch"]+1,
        checkpoint_sha256=hashes[str(paths[3])], artifacts_unchanged=True,
        original_hashes=hashes, elapsed_seconds=perf_counter()-started,
        effective_diffusion=float(torch.nn.functional.softplus(model.raw_diffusion)),
        effective_advection=float(torch.nn.functional.softplus(model.raw_advection)),
        units="Concentration in ug/m3; sums are concentration-cell sums, not physical atmospheric mass.",
        ablation="Frozen weights; same full-model stable timestep; wind retained in initializer for all learned variants.",
        results=summaries), indent=2))


if __name__ == "__main__":
    main()
