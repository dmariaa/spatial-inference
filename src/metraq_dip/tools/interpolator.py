from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


def median_nn_distance(x, y):
    xy = np.c_[x, y].astype(float)
    d, _ = cKDTree(xy).query(xy, k=2)   # self + nearest
    return float(np.median(d[:, 1]))


class Interpolator:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __call__(self, x, y, mode: str = "grid"):
        return self.interpolate(x, y, mode)

    def interpolate(self, x, y, mode: str = "grid"):
        raise NotImplementedError

class KrigingInterpolator(Interpolator):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)

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
    def __init__(self, x, y, z, interpolation_function: str = 'gaussian', epsilon: float = None):
        super().__init__(x, y, z)

        from scipy.interpolate import Rbf
        self.interpolator = Rbf(x, y, z, function=interpolation_function, epsilon=epsilon)

    def interpolate(self, x, y, mode: str = "grid"):
        if mode == "grid":
            x, y = np.meshgrid(x, y)

        return self.interpolator(x, y)


class IdwInterpolator(Interpolator):
    def __init__(self, x, y, z, power: float = 2.0, k: int = 8, max_radius_m: Optional[float] = None, eps: float = 1e-12):
        super().__init__(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
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
    x, y, z : array-like (N,)
        Training coordinates (UTM meters) and values.
    max_radius_m : float | None, optional (keyword-only)
        If set, predictions farther than this distance return NaN.

    Notes
    -----
    - Supports `mode='points'` (vector of target points) and `mode='grid'` (meshgrid arrays).
    - Exact coincident targets (distance==0) copy the station value.
    - Pure NumPy implementation (O(N*M)) is fine for ~dozens of stations; switch to a KD-tree if you scale to thousands.
    """
    def __init__(self, x, y, z, *, max_radius_m: Optional[float] = None):
        import numpy as _np
        super().__init__(_np.asarray(x, float), _np.asarray(y, float), _np.asarray(z, float))
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
            # Accept meshgrid-style arrays and return a grid of predictions
            X, Y = _np.asarray(x, float), _np.asarray(y, float)
            zhat = self._predict_points(X.ravel(), Y.ravel())
            return zhat.reshape(X.shape)
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'points' or 'grid'.")