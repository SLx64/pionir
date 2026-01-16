from functools import singledispatch
from typing import Any, Self, cast

import numpy as np

from ..core.collection import SpectrumCollection
from ..core.spectrum import Spectrum
from ..core.typing import SpectrumLike
from ..core.utils import apply_transformation
from ..sklearn.utils import StatelessTransformer


def _rnv(
    data: np.ndarray,
    iqr: tuple[float, float] = (25, 75),
    in_place: bool = False
) -> np.ndarray | None:
    """
    Applies the Robust Normal Variate (RNV) transformation to the input data.

    The normalization process involves centering the data to its median and
    scaling it using the interquartile range, defined by the provided lower and
    upper percentile bounds.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be normalized. Must be a NumPy array of any shape.
        Data is either modified in-place or a new normalized array is returned
        based on the `in_place` parameter.
    iqr : tuple of float, optional
        A pair specifying the lower and upper percentiles used to compute the
        interquartile range (IQR). Default is (25, 75), representing the first
        and third quartiles.
    in_place : bool, optional
        If True, normalization is performed in-place, modifying the original
        array and returning None. If False (default), a new normalized array is
        returned.

    Returns
    -------
    numpy.ndarray or None
        A new normalized NumPy array if `in_place` is False. Otherwise, None
        when the method modifies the original data in-place.
    """
    q_low, q_high = iqr
    q1, q3 = np.percentile(data, [q_low, q_high], axis=-1, keepdims=True)
    center = np.mean(data, axis=-1, keepdims=True)
    scale = q3 - q1
    if in_place:
        data -= center
        data /= scale
        return None
    return (data - center) / scale


@singledispatch
def rnv(
    data: SpectrumLike,
    iqr: tuple[float, float] = (25, 75),
    in_place: bool = False
) -> SpectrumLike | None:
    """
    This function is a generic implementation allowing for the transformation
    of data depending on its type.

    Parameters
    ----------
    data : SpectrumLike
        The input data to be normalized. Typically, this is a multidimensional
        data structure or array representing signals or spectra.
    iqr : tuple of float, optional
        A tuple specifying the lower and upper percentiles (as percentages)
        of the interquartile range used for scaling. Defaults to (25, 75).
    in_place : bool, optional
        If True, the normalization modifies the input `data` directly.
        Otherwise, it returns a new, normalized copy. Defaults to False.

    Returns
    -------
    SpectrumLike or None
        The normalized data. If `in_place` is False, it returns a new instance
        of the normalized data. If `in_place` is True, the function modifies
        the input directly and returns None.

    Raises
    ------
    NotImplementedError
        Raised if no handler for the input data type is registered.
    """
    raise NotImplementedError


@rnv.register(np.ndarray)
def _(
    data: np.ndarray,
    iqr: tuple[float, float] = (25, 75),
    in_place: bool = False
) -> np.ndarray | None:
    return _rnv(data, iqr, in_place)


@rnv.register(Spectrum)
def _(
    data: Spectrum,
    iqr: tuple[float, float] = (25, 75),
    in_place: bool = False
) -> Spectrum | None:
    return apply_transformation(
        data=data,
        transform_fn=_rnv,
        in_place=in_place,
        iqr=iqr
    )


@rnv.register(SpectrumCollection)
def _(
    data: SpectrumCollection,
    iqr: tuple[float, float] = (25, 75),
    in_place: bool = False
) -> SpectrumCollection | None:
    return apply_transformation(
        data=data,
        transform_fn=_rnv,
        in_place=in_place,
        iqr=iqr
    )


class RNVTransformer(StatelessTransformer):
    """
    Robust Normal Variate (RNV) Transformer.

    This class implements the Robust Normal Variate (RNV) transformation
    to remove scatter and normalize data. RNV uses the mean and a
    user-defined Interquartile Range (IQR) to standardize data.

    Attributes
    ----------
    iqr : tuple of float
        Interquartile range (IQR) used for robust normalization.
    """
    def __init__(self, iqr: tuple[float, float] = (25, 75)):
        self.iqr = iqr

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
        Transforms the input data using a Robust Normal Variate (RNV)
        transformation.

        Parameters
        ----------
        X : np.ndarray
            Input data array to be transformed. It is expected to be a
            2D array where rows represent samples and columns represent
            features.

        Returns
        -------
        np.ndarray
            Transformed data array after applying the RNV transformation.
            The output has the same shape as the input array `X`.
        """
        X = self._validate_X(X)  # noqa: N806
        return cast(np.ndarray, _rnv(X, iqr=self.iqr, in_place=False))
