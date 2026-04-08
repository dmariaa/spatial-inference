from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SurfaceOptimizer(Protocol):
    def optimize(self) -> np.ndarray:
        ...

    def get_artifacts(self) -> dict[str, Any]:
        ...

