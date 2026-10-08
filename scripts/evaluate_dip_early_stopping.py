"""Replay validation-only CNN-DIP early stopping from trusted local NPZ files.

Example: python scripts/evaluate_dip_early_stopping.py --experiment-dir
output/experiments/basic/metraq_nox_add24_ks_355 --patience 50
Writes separate reports; never changes the historical artifacts. Time savings
are iteration estimates, not measured wall-clock savings. NPZ normalization
metadata uses pickle: only run this on trusted experiment files.
"""
from __future__ import annotations
import click
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metraq_dip.trainer.dip_optimizer import select_surface_from_validation


def stopping_epoch(losses, patience, min_delta):
    best, stale = float('inf'), 0
    for step, loss in enumerate(losses):
        if not np.isfinite(loss):
            raise ValueError('Non-finite validation loss')
        if loss < best - min_delta:
            best, stale = float(loss), 0
        else:
            stale += 1
        if stale >= patience:
            return step + 1
    return len(losses)


@click.command(help=__doc__)
@click.option('--experiment-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help='Historical experiment directory.')
@click.option('--output-dir', type=click.Path(file_okay=False, path_type=Path), help='Directory for separate reports.')
@click.option('--patience', type=click.IntRange(min=1), default=50, show_default=True, help='Iterations without validation improvement before stopping.')
@click.option('--min-delta', type=click.FloatRange(min=0), default=0.0, show_default=True, help='Minimum validation improvement.')
@click.option('--limit', type=click.IntRange(min=1), help='Process only this many cases.')
def main(experiment_dir: Path, output_dir: Path | None, patience: int, min_delta: float, limit: int | None):
    cfg = yaml.safe_load((experiment_dir / 'config.yaml').read_text(encoding='utf-8'))
    if cfg.get('surface_selection', 'validation') != 'validation':
        raise ValueError('This utility supports validation mean selection only')
    if cfg.get('optimization_loss', 'mae') != 'mae':
        raise ValueError('This utility supports MAE validation stopping only')
    torch.set_num_threads(1)
    original = pd.read_csv(experiment_dir / 'results.csv')
    lookup = {(str(r.sensor_group), pd.Timestamp(r.time_window)): r for r in original.itertuples()}
    files = sorted(experiment_dir.glob('exp_*.npz'))
    if limit:
        files = files[:limit]
    if not files:
        raise ValueError('No experiment artifacts found')
    rows, members = [], []
    for case_no, file in enumerate(files, 1):
        group, stamp = file.stem[4:].rsplit('_', 1)
        timestamp = pd.to_datetime(stamp, format='%Y%m%dT%H%M%S')
        reference = lookup[(group, timestamp)]
        with np.load(file, allow_pickle=True) as z:
            history = z['train_k_output'].transpose(0, 2, 1, 3, 4)
            losses = z['val_k_loss'][:, 0, :, 0]
            if not np.isfinite(history).all():
                raise ValueError(f'Non-finite surface: {file}')
            if 'epochs_completed' in z and np.any(z['epochs_completed'] != history.shape[1]):
                raise ValueError('Expected complete CNN histories, not padded stopped histories')
            surfaces, full_surfaces, stops = [], [], []
            for member, (h, loss) in enumerate(zip(history, losses)):
                stop = stopping_epoch(loss, patience, min_delta)
                surface, selected = select_surface_from_validation(output_history=h[:stop], val_loss_history=loss[:stop], k_best_n=min(int(cfg['k_best_n']), stop))
                old_idx = z['val_min_idx'][member]
                full_surface = torch.as_tensor(h).index_select(0, torch.as_tensor(old_idx)).mean(0).numpy()
                surfaces.append(surface.numpy())
                full_surfaces.append(full_surface)
                stops.append(stop)
                members.append(dict(sensor_group=group, time_window=timestamp.isoformat(), member=member, epochs_original=len(loss), epochs_completed=stop, epochs_saved=len(loss)-stop, selected_original=json.dumps(old_idx.tolist()), selected_stopped=json.dumps(selected.tolist()), same_epoch_set=set(old_idx.tolist()) == set(selected.tolist())))
            stats = z['normalization_stats'].item() if cfg.get('normalize') else None
            def restore(surface):
                result = surface.copy()
                if stats is not None:
                    for channel, pollutant in enumerate(cfg['pollutants']):
                        mean, std = stats[pollutant]
                        result[channel] = result[channel] * (std + 1e-6) + mean
                return result
            # The original ensemble averages denormalized member surfaces.
            stopped = np.stack([restore(s) for s in surfaces]).mean(0)
            baseline = np.stack([restore(s) for s in full_surfaces]).mean(0)
            target = restore(z['test_data'][0, :, 0])
            mask = z['test_mask'][0, :, 0].astype(bool)
            def metrics(surface):
                diff = surface[mask] - target[mask]
                if not diff.size:
                    raise ValueError('Empty TEST mask')
                return float(np.abs(diff).mean()), float(np.square(diff).mean())
            base_mae, base_mse = metrics(baseline)
            mae, mse = metrics(stopped)
            if not np.allclose([base_mae, base_mse], [reference.DIP_L1Loss, reference.DIP_MSELoss], rtol=2e-5, atol=2e-5):
                raise ValueError(f'Baseline reconstruction mismatch: {file.name}: {base_mae}, {base_mse}')
            rows.append(dict(sensor_group=group, time_window=timestamp.isoformat(), baseline_mae=base_mae, stopped_mae=mae, delta_mae=mae-base_mae, baseline_mse=base_mse, stopped_mse=mse, delta_mse=mse-base_mse, epochs_original=history.shape[0]*history.shape[1], epochs_completed=sum(stops), same_surface=bool(np.array_equal(stopped, baseline))))
        if case_no % 200 == 0:
            print(f'Processed {case_no}/{len(files)}', flush=True)
    cases, detail = pd.DataFrame(rows), pd.DataFrame(members)
    summary = dict(cases=len(cases), members=len(detail), patience=patience, min_delta=min_delta, baseline_reconstruction_verified=True, stopping_signal='validation MAE only', time_estimate='iteration fraction only; excludes initialization, IO and other methods', epochs_original=int(cases.epochs_original.sum()), epochs_completed=int(cases.epochs_completed.sum()), iterations_saved_percent=float(100*(1-cases.epochs_completed.sum()/cases.epochs_original.sum())), members_stopped_early=int((detail.epochs_saved > 0).sum()), identical_surfaces=int(cases.same_surface.sum()))
    for metric in ('mae', 'mse'):
        base, stopped, delta = (cases[f'{prefix}_{metric}'] for prefix in ('baseline', 'stopped', 'delta'))
        summary[metric] = dict(baseline_mean=float(base.mean()), stopped_mean=float(stopped.mean()), mean_change_percent=float(100*(stopped.mean()/base.mean()-1)), improved=int((delta < -1e-5).sum()), worsened=int((delta > 1e-5).sum()), unchanged=int((delta.abs() <= 1e-5).sum()), baseline_p95=float(base.quantile(.95)), stopped_p95=float(stopped.quantile(.95)), worst_case_increase=float(delta.max()))
    out = output_dir or experiment_dir / f'early_stopping_p{patience}'
    out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out / 'cases.csv', index=False)
    detail.to_csv(out / 'members.csv', index=False)
    cases.groupby('sensor_group').mean(numeric_only=True).to_csv(out / 'sensor_groups.csv')
    cases.groupby('time_window').mean(numeric_only=True).to_csv(out / 'time_windows.csv')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
