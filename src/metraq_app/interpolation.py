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


def _dip_surface(data, mask, epochs):
    import torch
    from metraq_dip.trainer.dip_optimizer import DipOptimizer

    observed = data[mask]
    mean, std = float(observed.mean()), max(float(observed.std()), 1e-6)
    normalized = np.zeros_like(data)
    normalized[mask] = (data[mask] - mean) / std
    target = normalized[None, None]
    target_mask = mask[None, None]
    noise = np.random.default_rng().normal(0, 0.1, (8, 1, *data.shape)).astype(np.float32)
    config = {
        'pollutants': [0], 'normalize': True, 'epochs': epochs, 'lr': 0.01,
        'optimization_loss': 'mae', 'optimization_timesteps': 'last',
        'surface_selection': 'last', 'k_best_n': 1,
        'model': {'architecture': 'unet', 'base_channels': 16, 'levels': 2,
                  'preserve_time': True, 'kernel_size': (1, 3, 3), 'learned_upsampling': False},
    }
    split = {'input_data': noise, 'train_data': target, 'train_mask': target_mask,
             # Diagnostic fit losses only; there is no validation-based selection.
             'val_data': target, 'val_mask': target_mask,
             'normalization_stats': {0: (mean, std)}}
    # Local CPU work only. Serialize training and restore process-wide thread settings.
    with DIP_LOCK:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(min(previous_threads, 4))
            optimizer = DipOptimizer(configuration=config, split_data=split, device='cpu', disable_tqdm=True)
            return optimizer.optimize()[0]
        finally:
            torch.set_num_threads(previous_threads)


def reconstruct(shape, cells, method, dip_epochs=250):
    """Preserve observations exactly and predict only previously empty cells."""
    data, mask = observation_grid(shape, cells)
    if method not in METHOD_NAMES:
        raise ValueError('Unknown interpolation method')
    surface = np.full(shape, np.nan, dtype=np.float32)
    surface[mask] = data[mask]
    if method == 'observed' or not mask.any():
        return surface
    observed = data[mask]
    if len(observed) == 1 or np.ptp(observed) == 0:
        # A constant field avoids an undefined variogram with too few/constant values.
        surface.fill(float(observed[0]))
    elif method == 'dip-cnn':
        surface = _dip_surface(data, mask, dip_epochs)
    else:
        interpolator = {'krg': KrigingInterpolator, 'idw': IdwInterpolator}[method]
        surface = calculate_interpolations(data[None, None], mask[None, None], interpolator)[0, 0]
    if not np.isfinite(surface).all():
        raise ValueError('The reconstruction produced non-finite values')
    surface = np.asarray(surface, dtype=np.float32).copy()
    surface[mask] = data[mask]
    return surface
