"""One reusable train-fit robust feature scaler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(slots=True)
class RobustScaler:
    """Median/IQR transformer using the frozen ``IQR + epsilon`` rule."""

    epsilon: float = 1e-8
    center_: NDArray[np.float64] | None = None
    iqr_: NDArray[np.float64] | None = None
    scale_: NDArray[np.float64] | None = None

    def fit(self, values: ArrayLike) -> "RobustScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("robust scaling requires a non-empty two-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError("robust scaling fit data must be finite")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.center_ = np.median(array, axis=0)
        q25, q75 = np.quantile(array, (0.25, 0.75), axis=0)
        self.iqr_ = q75 - q25
        self.scale_ = self.iqr_ + self.epsilon
        return self

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustScaler must be fit before transform")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.center_.size:
            raise ValueError("transform feature dimension does not match fitted scaler")
        if not np.all(np.isfinite(array)):
            raise ValueError("robust scaling transform data must be finite")
        return (array - self.center_) / self.scale_

    def fit_transform(self, values: ArrayLike) -> NDArray[np.float64]:
        return self.fit(values).transform(values)
