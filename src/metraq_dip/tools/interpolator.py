from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def median_nn_distance(x, y):
    xy = np.c_[x, y].astype(float)
    d, _ = cKDTree(xy).query(xy, k=2)   # self + nearest
    return float(np.median(d[:, 1]))


class Interpolator:
    """Base class for methods that reconstruct the final step of a data window."""

    def __init__(self, data, mask):
        self.data = np.asarray(data, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        if self.data.ndim != 3:
            raise ValueError("data and mask must have shape (time, height, width)")
        if self.data.shape != self.mask.shape:
            raise ValueError("data and mask must have the same shape")
        if self.data.shape[0] == 0:
            raise ValueError("the interpolation window must contain at least one timestep")

        self.target_t = self.data.shape[0] - 1

    def target_observations(self):
        target_data = self.data[self.target_t]
        target_mask = self.mask[self.target_t] & np.isfinite(target_data)
        known_points = np.argwhere(target_mask)
        if known_points.size == 0:
            raise ValueError("the target timestep must contain at least one observed cell")
        y = known_points[:, 0].astype(float)
        x = known_points[:, 1].astype(float)
        z = target_data[target_mask].astype(float)
        return x, y, z

    def __call__(self, x, y, mode: str = "grid"):
        return self.interpolate(x, y, mode)

    def interpolate(self, x, y, mode: str = "grid"):
        raise NotImplementedError


class SpatioTemporalGPInterpolator(Interpolator):
    """Gaussian-process interpolation over space and time.

    ``data`` and ``mask`` contain the complete ``(T, H, W)`` window. Grid-cell
    coordinates and hourly indices use the units of their corresponding length
    scales.
    Kernel parameters are fixed; select them using validation data, not test
    locations. Predictions are posterior means in the original units of ``z``.
    """

    def __init__(
        self,
        data,
        mask,
        *,
        spatial_length_scale: float = 3.0,
        temporal_length_scale: float = 6.0,
        noise_fraction: float = 0.1,
        jitter: float = 1e-8,
    ):
        super().__init__(data, mask)
        if not np.isfinite([spatial_length_scale, temporal_length_scale, noise_fraction, jitter]).all():
            raise ValueError("kernel parameters must be finite")
        if spatial_length_scale <= 0 or temporal_length_scale <= 0:
            raise ValueError("length scales must be positive")
        if noise_fraction < 0 or jitter <= 0:
            raise ValueError("noise_fraction must be nonnegative and jitter must be positive")

        valid = self.mask & np.isfinite(self.data)
        t, y, x = np.nonzero(valid)
        if len(t) == 0:
            raise ValueError("the interpolation window must contain at least one finite observation")
        self.x = x.astype(float)
        self.y = y.astype(float)
        self.z = self.data[valid].astype(float)
        self.t = t.astype(float)
        self.spatial_length_scale = float(spatial_length_scale)
        self.temporal_length_scale = float(temporal_length_scale)
        self.mean = float(np.mean(self.z))
        self.variance = max(float(np.var(self.z)), jitter)
        self.locations = np.column_stack((self.x, self.y))

        covariance = self._kernel(self.locations, self.t, self.locations, self.t)
        covariance[np.diag_indices_from(covariance)] += self.variance * noise_fraction + jitter
        self.factor = cho_factor(covariance, lower=True, check_finite=False)
        self.alpha = cho_solve(self.factor, self.z - self.mean, check_finite=False)

    def _kernel(self, locations_a, times_a, locations_b, times_b):
        spatial_distance = cdist(locations_a, locations_b) / self.spatial_length_scale
        scaled_distance = np.sqrt(3.0) * spatial_distance
        spatial_kernel = (1.0 + scaled_distance) * np.exp(-scaled_distance)
        temporal_distance = np.abs(np.subtract.outer(times_a, times_b)) / self.temporal_length_scale
        return self.variance * spatial_kernel * np.exp(-temporal_distance)

    def interpolate(self, x, y, mode: str = "grid"):
        if mode == "grid":
            x_targets, y_targets = np.meshgrid(x, y)
        elif mode == "points":
            x_targets, y_targets = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        else:
            raise ValueError("Unsupported mode. Use 'points' or 'grid'.")

        target_times = np.full(x_targets.shape, self.target_t, dtype=float)
        locations = np.column_stack((x_targets.ravel(), y_targets.ravel()))
        times = target_times.ravel()
        if not np.isfinite(locations).all() or not np.isfinite(times).all():
            raise ValueError("prediction coordinates and times must be finite")
        cross_covariance = self._kernel(locations, times, self.locations, self.t)
        prediction = self.mean + cross_covariance @ self.alpha
        return prediction.reshape(x_targets.shape)


class KrigingInterpolator(Interpolator):
    def __init__(self, data, mask):
        super().__init__(data, mask)
        x, y, z = self.target_observations()

        from pykrige import OrdinaryKriging
        # self.interpolator = OrdinaryKriging(x, y, z,
        #                                     coordinates_type='euclidean',
        #                                     variogram_model='gaussian',
        #                                     variogram_parameters={'sill': (max(z) - min(z)) * 0.9,
        #                                                           'range': 0.8 * ((max(x) - min(x)) ** 2 +
        #                                                                           (max(y) - min(y)) ** 2) ** 0.5,
        #                                                           'nugget': (max(z) - min(z)) * 0.05},
        #                                     verbose=False, enable_plotting=False)

        # UTM coordinates
        range_mult = 3.0
        nugget_frac = 0.1
        z = np.asarray(z, float)
        sill = float(np.var(z, ddof=1)) or 1e-8
        vrange = max(range_mult * median_nn_distance(x, y), 1.0)
        nugget = max(nugget_frac * sill, 1e-8)

        self.interpolator = OrdinaryKriging(
            x, y, z,
            coordinates_type="euclidean",  # UTM meters
            variogram_model="exponential",  # switched from gaussian → exponential
            variogram_parameters={
                "sill": sill,
                "range": vrange,
                "nugget": nugget,
            },
            verbose=False, enable_plotting=False,
        )

    def interpolate(self, x, y, mode: str = "grid"):
        z_values, _ =  self.interpolator.execute(mode, x, y)
        return z_values.data


class RbfInterpolator(Interpolator):
    def __init__(self, data, mask, interpolation_function: str = 'gaussian', epsilon: float = None):
        super().__init__(data, mask)
        x, y, z = self.target_observations()

        from scipy.interpolate import Rbf
        self.interpolator = Rbf(x, y, z, function=interpolation_function, epsilon=epsilon)

    def interpolate(self, x, y, mode: str = "grid"):
        if mode == "grid":
            x, y = np.meshgrid(x, y)

        return self.interpolator(x, y)


class IdwInterpolator(Interpolator):
    def __init__(self, data, mask, power: float = 2.0, k: int = 8, max_radius_m: Optional[float] = None, eps: float = 1e-12):
        super().__init__(data, mask)
        self.x, self.y, self.z = self.target_observations()
        self.tree = cKDTree(np.c_[self.x, self.y])
        self.power = float(power)
        self.k = int(min(k, len(self.z))) if len(self.z) > 0 else 0
        self.max_radius_m = max_radius_m
        self.eps = float(eps)

    def _predict_points(self, x_t, y_t):
        pts = np.c_[np.asarray(x_t, float).ravel(), np.asarray(y_t, float).ravel()]
        if self.k == 0 or len(self.z) == 0:
            return np.full(pts.shape[0], np.nan, dtype=float)

        d, idx = self.tree.query(pts, k=self.k)  # d: (M,k), idx: (M,k) if k>1
        if self.k == 1:  # make 2D for uniform handling
            d = d[:, None]
            idx = idx[:, None]

        # exact matches: any distance <= eps → copy that station
        exact = d <= self.eps
        has_exact = exact.any(axis=1)
        zhat = np.empty(d.shape[0], dtype=float)

        if has_exact.any():
            # pick the first exact-hit value per row
            j_exact = exact.argmax(axis=1)
            zhat[has_exact] = self.z[idx[np.arange(d.shape[0]), j_exact]][has_exact]

        # non-exact rows → standard IDW
        need_idw = ~has_exact
        if need_idw.any():
            dd = d[need_idw]
            ii = idx[need_idw]

            # optional radius cap: mark rows whose nearest is too far as NaN
            if self.max_radius_m is not None:
                too_far = (dd.min(axis=1) > self.max_radius_m)
            else:
                too_far = np.zeros(dd.shape[0], dtype=bool)

            w = 1.0 / np.maximum(dd, self.eps) ** self.power
            w /= np.maximum(w.sum(axis=1, keepdims=True), self.eps)
            zhat_idw = np.sum(w * self.z[ii], axis=1)
            zhat[need_idw] = np.where(too_far, np.nan, zhat_idw)

        return zhat

    def interpolate(self, x, y, mode: str = "points"):
        if mode == "points":
            out = self._predict_points(x, y)
            # reshape to match x's shape if x is array-like
            shp = np.shape(x)
            return out.reshape(shp if shp else (-1,))
        elif mode == "grid":
            X, Y = np.meshgrid(x, y)
            out = self._predict_points(X, Y).reshape(X.shape)
            return out
        else:
            raise ValueError("Unsupported mode. Use 'points' or 'grid'.")


class NearestNeighborInterpolator(Interpolator):
    """1-NN (nearest station) interpolator.

    Parameters
    ----------
    data, mask : array-like (T, H, W)
        Full observation window and its validity mask.
    max_radius_m : float | None, optional (keyword-only)
        If set, predictions farther than this distance return NaN.

    Notes
    -----
    - Supports `mode='points'` (vector of target points) and `mode='grid'` (meshgrid arrays).
    - Exact coincident targets (distance==0) copy the station value.
    - Pure NumPy implementation (O(N*M)) is fine for ~dozens of stations; switch to a KD-tree if you scale to thousands.
    """
    def __init__(self, data, mask, *, max_radius_m: Optional[float] = None):
        import numpy as _np
        super().__init__(data, mask)
        self.x, self.y, self.z = self.target_observations()
        self.max_radius_m = max_radius_m

    def _predict_points(self, x_t, y_t):
        import numpy as _np
        xt = _np.asarray(x_t, float)[:, None]
        yt = _np.asarray(y_t, float)[:, None]
        xs = _np.asarray(self.x, float)[None, :]
        ys = _np.asarray(self.y, float)[None, :]
        d2 = (xt - xs) ** 2 + (yt - ys) ** 2
        jmin = d2.argmin(axis=1)
        dmin = _np.sqrt(d2[_np.arange(d2.shape[0]), jmin])
        zhat = _np.asarray(self.z, float)[jmin]
        if self.max_radius_m is not None:
            zhat = _np.where(dmin <= self.max_radius_m, zhat, _np.nan)
        return zhat

    def interpolate(self, x, y, mode: str = "points"):
        import numpy as _np
        if mode == "points":
            return self._predict_points(x, y)
        elif mode == "grid":
            X, Y = _np.meshgrid(x, y)
            zhat = self._predict_points(X.ravel(), Y.ravel())
            return zhat.reshape(X.shape)
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'points' or 'grid'.")
