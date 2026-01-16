from functools import singledispatch
from typing import Any, Self, cast

import numpy as np
from scipy.signal import savgol_filter

from ..core.collection import SpectrumCollection
from ..core.spectrum import Spectrum
from ..core.typing import SpectrumLike
from ..core.utils import apply_transformation
from ..sklearn.utils import StatelessTransformer


def _savgol(
    data: np.ndarray,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    in_place: bool = False
) -> np.ndarray | None:
    """
    Applies the Savitzky-Golay filter to a given dataset and optionally
    modifies it in place.

    This function smooths the input data using the Savitzky-Golay filter,
    which fits successive subsets of the data to a polynomial of a specified
    order via least squares. It can also compute a derivative of the data if
    desired. Results can either be returned or written directly to the input
    array if `in_place` is set to True.

    Parameters
    ----------
    data : np.ndarray
        The input array containing data to be filtered and optionally
        modified in place.
    window_length : int
        The length of the filter window, which must be a positive odd integer.
    polyorder : int
        The order of the polynomial to fit, which must be less
        than `window_length`.
    deriv : int, default 0
        The order of the derivative to compute. 0 means no derivative,
        1 means the first derivative, and so on.
    in_place : bool, default False
        If True, writes the filtered data or its derivative back into the input
        array `data`. If False, the filtered result is returned as a new array.

    Returns
    -------
    np.ndarray or None
        The filtered data or its derivative as a new array if `in_place`
        is False. None is returned if `in_place` is True, in which case the
        input array `data` is modified directly.

    Notes
    -----
    If the input data is multidimensional, the filter is applied along the
    last axis (axis=-1).
    """
    res = savgol_filter(data, window_length, polyorder, deriv=deriv, axis=-1)
    if in_place:
        data[:] = res
        return None
    return res


@singledispatch
def savgol(
    data: SpectrumLike,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    in_place: bool = False
) -> SpectrumLike | None:
    """
    This function is a generic implementation allowing for the transformation
    of data depending on its type.

    Parameters
    ----------
    data : SpectrumLike
        The input spectral-like data for filtering. It must adhere to the
        expected format or type required by the filter implementation.
    window_length : int
        The length of the filter window, which must be a positive odd integer.
    polyorder : int
        The order of the polynomial to fit, which must be less
        than `window_length`.
    deriv : int, default 0
        The order of the derivative to compute. 0 means no derivative,
        1 means the first derivative, and so on.
    in_place : bool, default False
        If True, writes the filtered data or its derivative back into the input
        array `data`. If False, the filtered result is returned as a new array.

    Returns
    -------
    SpectrumLike or None
        If `in_place` is `False`, the function returns a new filtered
        spectrum-like object  of the same type as `data`. If `in_place` is
        `True`, no value is returned and the operation modifies the input
        `data` object directly.

    Raises
    ------
    NotImplementedError
        Raised if no handler for the input data type is registered.
    """
    raise NotImplementedError


@savgol.register(np.ndarray)
def _(
    data: np.ndarray,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    in_place: bool = False
) -> np.ndarray | None:
    return _savgol(data, window_length, polyorder, deriv, in_place)


@savgol.register(Spectrum)
def _(
    data: Spectrum,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    in_place: bool = False
) -> Spectrum | None:
    return apply_transformation(
        data=data,
        transform_fn=_savgol,
        in_place=in_place,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv
    )


@savgol.register(SpectrumCollection)
def _(
    data: SpectrumCollection,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    in_place: bool = False
) -> SpectrumCollection | None:
    return apply_transformation(
        data=data,
        transform_fn=_savgol,
        in_place=in_place,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv
    )


class SGFTransformer(StatelessTransformer):
    """
    Applies the Savitzky-Golay filter to smooth or derive data.

    This class implements a transformer that applies the Savitzky-Golay filter
    to input arrays for smoothing or calculating derivatives of data.

    Attributes
    ----------
    window_length : int
        Length of the filter window (i.e., the number of coefficients). Must
        be a positive odd integer.
    polyorder : int
        Order of the polynomial used to fit the samples. Must be less than
        `window_length`.
    deriv : int
        The order of the derivative to compute. Default is 0, which applies
        smoothing only.
    """
    def __init__(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 0
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:  # noqa: N803
        """
        Fits the model to the provided data. Has no effect on the model in
        this case.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit the model.
        y : Any or None, optional
            Optional target variable.

        Returns
        -------
        Self
            The instance of the fitted model.
        """
        self._validate_X(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """
        Savitzky-Golay filter to smooth the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data array to be transformed. It is expected to be a
            2D array where rows represent samples and columns represent
            features.

        Returns
        -------
        np.ndarray
            Transformed data array after applying the Savitzky-Golay filter.
            The output has the same shape as the input array `X`.
        """
        X = self._validate_X(X)  # noqa: N806
        return cast(np.ndarray, _savgol(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            in_place=False
        ))
