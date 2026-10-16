"""Train the geometric kernel on TRAIN/validation only; never evaluate TEST."""
import json
from pathlib import Path
from time import perf_counter
import click
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from metraq_gnn.data.cache import load_aq_sensor_cache
from metraq_gnn.data.splits import split_sensor_nodes, compute_training_normalization
from metraq_gnn.data.kernel_window import CachedKernelDataset
from metraq_gnn.model import ConvexGeometricKernel
from metraq_gnn.trainer import GNNTrainer
from metraq_gnn.tracking import WandbLogger


@click.command(context_settings={"show_default": True})
@click.option("--cache", type=click.Path(exists=True, path_type=Path),
              default=Path("cache/metraq/no2-2010-2024"))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--mode", type=click.Choice(["fixed", "power", "neural"]), default="neural")
@click.option("--test-sensor-ids", multiple=True, type=int)
@click.option("--partition-summary", type=click.Path(exists=True, path_type=Path),
              help="Recover the exact split from a previous GNN summary.")
@click.option("--validation-nodes", type=click.IntRange(min=1), default=4)
@click.option("--seed", type=int, default=None, help="Optional reproducibility seed; otherwise random.")
@click.option("--epochs", type=click.IntRange(min=1), default=100)
@click.option("--batch-size", type=click.IntRange(min=1), default=16)
@click.option("--max-train-windows", type=click.IntRange(min=1))
@click.option("--max-validation-windows", type=click.IntRange(min=1))
@click.option("--device", default="cpu")
@click.option("--wandb", is_flag=True)
@click.option("--wandb-project", default="metraq-gnn-test")
@click.option("--wandb-entity", default="dmariaa-team")
@click.option("--wandb-name")
@click.option("--wandb-mode", type=click.Choice(["online", "offline", "disabled"]), default="online")
def main(cache, output, mode, test_sensor_ids, validation_nodes, seed, epochs,
         batch_size, max_train_windows, max_validation_windows, device, partition_summary,
         wandb, wandb_project, wandb_entity, wandb_name, wandb_mode):
    started = perf_counter()
    if output.exists():
        raise click.ClickException("output already exists; use a new directory")
    seed = int(np.random.default_rng().integers(0, 2**31 - 1)) if seed is None else seed
    torch.manual_seed(seed)
    aq = load_aq_sensor_cache(cache)
    reference = None
    if partition_summary is not None:
        if test_sensor_ids:
            raise click.UsageError("partition-summary and test-sensor-ids are exclusive")
        reference = json.loads(partition_summary.read_text())
        train, validation, test = split_sensor_nodes(
            aq, validation_nodes=reference["validation_nodes"],
            test_nodes=reference["test_nodes"], seed=reference["seed"],
            test_sensor_ids=reference.get("test_sensor_ids"))
    else:
        if not test_sensor_ids:
            raise click.UsageError("provide partition-summary or test-sensor-ids")
        train, validation, test = split_sensor_nodes(
            aq, validation_nodes=validation_nodes, test_nodes=len(test_sensor_ids),
            seed=seed, test_sensor_ids=test_sensor_ids)
    mean, std = compute_training_normalization(
        aq, train_node_mask=train, start="2015-01-01", end="2022-12-31 23:00")
    if reference is not None:
        np.testing.assert_allclose(mean, reference["normalization_mean"], rtol=1e-6)
        np.testing.assert_allclose(std, reference["normalization_std"], rtol=1e-6)
        if int(train.sum()) != reference["train_nodes"]:
            raise ValueError("reconstructed TRAIN size differs from reference")
    common = dict(cache=aq, hours=24, context_sensor_mask=train,
                  forbidden_context_mask=validation | test, seed=seed,
                  normalization_mean=mean, normalization_std=std)
    training = CachedKernelDataset(
        **common, target_fraction=.25, split_start="2015-01-01",
        split_end="2022-12-31 23:00")
    validating = CachedKernelDataset(
        **common, target_sensor_mask=validation, split_start="2023-01-01",
        split_end="2023-12-31 23:00")
    def loader(dataset, maximum, shuffle):
        if maximum is not None:
            dataset = Subset(dataset, range(min(maximum, len(dataset))))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    train_loader = loader(training, max_train_windows, True)
    val_loader = loader(validating, max_validation_windows, False)
    model = ConvexGeometricKernel(mode=mode, cell_size_m=aq.metadata["cell_size_m"])
    model.prepare_geometry(aq.graph.pos, np.unique(aq.sensor_node_indices[
        train.reshape(-1)[aq.sensor_node_indices]]))
    output.mkdir(parents=True)
    np.savez(output / "partition.npz", train_mask=train,
             validation_mask=validation, test_mask=test,
             sensor_ids=aq.sensor_ids, sensor_node_indices=aq.sensor_node_indices)
    config = dict(epsilon_m=model.epsilon_m, distance_scale_m=model.distance_scale_m,
                  source_policy="all-current-visible", implementation="pyg-vectorized", mode=mode, seed=seed, test_evaluated=False, hours=24,
                  uses_history=False, train_period="2015-2022", validation_period="2023",
                  train_nodes=np.flatnonzero(train).tolist(),
                  validation_nodes=np.flatnonzero(validation).tolist(),
                  test_sensor_ids=aq.sensor_ids[test.reshape(-1)[aq.sensor_node_indices]].tolist(),
                  partition_summary=str(partition_summary) if partition_summary else None,
                  normalization_mean=mean.tolist(),
                  normalization_std=std.tolist(), cell_size_m=aq.metadata["cell_size_m"],
                  epochs_requested=epochs, batch_size=batch_size,
                  train_windows=len(train_loader.dataset),
                  validation_windows=len(val_loader.dataset))
    logger = (WandbLogger(
        project=wandb_project, entity=wandb_entity, name=wandb_name,
        mode=wandb_mode, config=config, tags=["convex-kernel", "optimized", "pilot"])
        if wandb else None)
    config["wandb_url"] = logger.url if logger is not None else None
    exit_code = 1
    try:
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        trainer = GNNTrainer(model=model, device=device, optimization_loss="mae",
                             show_progress=True, logger=logger)
        training_started = perf_counter()
        if mode == "fixed":
            validation_metrics = trainer.evaluate(val_loader)
            result = dict(validation=validation_metrics, epochs_completed=0, best_epoch=-1,
                          best_validation_mae=validation_metrics["mae"])
            torch.save(dict(epoch=-1, validation_loss=validation_metrics["mae"],
                            optimization_loss="mae", selection="fixed-no-training",
                            model_state_dict=model.state_dict()), output / "best-model.pt")
        else:
            fit = trainer.fit(train_loader, val_loader, epochs=epochs,
                              checkpoint_path=output / "best-model.pt")
            result = dict(best_epoch=fit.best_epoch, best_validation_mae=fit.best_validation_loss,
                          epochs_completed=fit.epochs_completed, history=fit.history)
        result["best_validation_mae_physical"] = (
            result["best_validation_mae"] * float(std[0]) if len(std) == 1 else None)
        result.update(training_seconds=perf_counter()-training_started,
                      elapsed_seconds=perf_counter()-started,
                      setup_seconds=training_started-started, wandb_url=config["wandb_url"])
        result.update(power=float(model.power.detach().cpu()), test_evaluated=False)
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if logger is not None:
            logger.log_summary({key: value for key, value in result.items()
                                if isinstance(value, (int, float))})
        click.echo(json.dumps(result, indent=2))
        exit_code = 0
    finally:
        if logger is not None:
            logger.finish(exit_code=exit_code)



if __name__ == "__main__":
    main()
