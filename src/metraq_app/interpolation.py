"""Single-hour reconstructions for the interactive viewer."""
from threading import Lock

import numpy as np

from metraq_dip.tools.interpolator import IdwInterpolator, KrigingInterpolator
from metraq_dip.tools.tools import calculate_interpolations

METHOD_NAMES = {'observed': 'Observaciones', 'dip-cnn': 'DIP-CNN', 'krg': 'KRG', 'idw': 'IDW'}
DIP_LOCK = Lock()


def observation_grid(shape, cells):
    data = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    for cell in cells.itertuples():
        data[int(cell.row), int(cell.col)] = float(cell.value)
        mask[int(cell.row), int(cell.col)] = True
    return data, mask


def dip_split(data, mask, rng):
    """Hold out entire cells for validation across the complete time window."""
    available = np.argwhere(mask.any(axis=0))
    latest = np.argwhere(mask[-1])
    count = min(4, max(1, len(available) // 5), len(latest), len(available) - 1)
    if count < 1:
        raise ValueError('DIP necesita al menos dos celdas observadas para validar.')
    selected = latest[rng.choice(len(latest), count, replace=False)]
    validation = np.zeros(mask.shape[1:], dtype=bool)
    validation[selected[:, 0], selected[:, 1]] = True
    train_mask = mask & ~validation[None]
    val_mask = mask & validation[None]
    values = data[train_mask]
    mean, std = float(values.mean()), max(float(values.std()), 1e-6)
    normalized = np.zeros_like(data)
    normalized[mask] = (data[mask] - mean) / std
    noise = rng.normal(0, 0.1, (8, *data.shape)).astype(np.float32)
    inputs = np.concatenate((noise, (normalized * train_mask)[None], train_mask[None].astype(np.float32)))
    return {'input_data': inputs, 'train_data': (normalized * train_mask)[None],
            'val_data': (normalized * val_mask)[None], 'train_mask': train_mask[None],
            'val_mask': val_mask[None], 'normalization_stats': {0: (mean, std)}}, count


def _dip_surface(data, mask, epochs, details):
    import torch
    from metraq_dip.trainer.dip_optimizer import DipOptimizer
    from metraq_dip.trainer.dip_ensemble_optimizer import reduce_surface_ensemble

    config = {
        'pollutants': [0], 'normalize': True, 'epochs': epochs, 'lr': 0.01,
        'optimization_loss': 'mae', 'optimization_timesteps': 'all',
        'surface_selection': 'validation', 'k_best_n': min(3, epochs),
        'model': {'architecture': 'unet', 'base_channels': 16, 'levels': 2 if max(data.shape) <= 8 else 3,
                  'preserve_time': False, 'kernel_size': (3, 5, 5), 'learned_upsampling': False},
    }
    rng = np.random.default_rng()
    surfaces = []
    with DIP_LOCK:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(min(previous_threads, 4))
            for _ in range(5):
                split, validation_count = dip_split(data, mask, rng)
                optimizer = DipOptimizer(configuration=config, split_data=split, device='cpu', disable_tqdm=True)
                surfaces.append(optimizer.optimize()[0])
        finally:
            torch.set_num_threads(previous_threads)
    details.update(ensemble_size=5, validation_cells=validation_count)
    return reduce_surface_ensemble(surfaces=surfaces)


def reconstruct(shape, cells, method, dip_epochs=250, *, history=None, details=None):
    """Preserve observations exactly and predict only previously empty cells."""
    data, mask = observation_grid(shape, cells)
    if method not in METHOD_NAMES:
        raise ValueError('Unknown interpolation method')
    surface = np.full(shape, np.nan, dtype=np.float32)
    surface[mask] = data[mask]
    if method == 'observed' or not mask.any():
        return surface
    details = details if details is not None else {}
    sequence, sequence_mask = history if history is not None else (data[None], mask[None])
    observed = sequence[sequence_mask] if method == 'dip-cnn' else data[mask]
    if (method != 'dip-cnn' and len(observed) == 1) or np.ptp(observed) == 0 or (method == 'dip-cnn' and sequence_mask.any(axis=0).sum() < 2):
        # A constant field avoids an undefined variogram with too few/constant values.
        surface.fill(float(data[mask].mean()))
        details['constant_field'] = True
    elif method == 'dip-cnn':
        surface = _dip_surface(sequence, sequence_mask, dip_epochs, details)
    else:
        interpolator = {'krg': KrigingInterpolator, 'idw': IdwInterpolator}[method]
        surface = calculate_interpolations(data[None, None], mask[None, None], interpolator)[0, 0]
    if not np.isfinite(surface).all():
        raise ValueError('The reconstruction produced non-finite values')
    surface = np.asarray(surface, dtype=np.float32).copy()
    surface[mask] = data[mask]
    return surface
