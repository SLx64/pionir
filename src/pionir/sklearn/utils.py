from abc import ABC
from typing import cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class ArrayValidationMixin:
    def _validate_X(  # noqa: N802
        self,
        X: np.ndarray,  # noqa: N803
        *,
        ensure_2d: bool = True,
        dtype: type[np.floating] | None = np.float64,
    ) -> np.ndarray:
        return cast(
            np.ndarray,
            check_array(X, ensure_2d=ensure_2d, dtype=dtype)
        )


class StatelessTransformer(
    BaseEstimator,
    TransformerMixin,
    ArrayValidationMixin,
    ABC
):
    pass
