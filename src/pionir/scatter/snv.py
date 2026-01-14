from functools import singledispatch
from typing import Any, Self, cast

import numpy as np

from ..core.collection import SpectrumCollection
from ..core.spectrum import Spectrum
from ..core.transform import apply_transformation
from ..core.typing import SpectrumLike
from ..sklearn.utils import StatelessTransformer


def _snv(
    data: np.ndarray,
    norm: bool = False,
    in_place: bool = False
) -> np.ndarray | None:
    """
    Applies the Standard Normal Variate (SNV) transformation to the input data.

    This function standardizes the input data by removing the mean and scaling
    by the standard deviation along the specified axis. The `norm` parameter
    controls whether the mean subtraction is skipped, while the `in_place`
    parameter allows modification of the input array directly.

    SNV transformation formula:
        For each row or data instance:
            standardized = (row - mean) / std

    Parameters
    ----------
    data : numpy.ndarray
        The input array to be transformed. It can have any dimensionality.

    norm : bool, optional
        If True, the mean subtraction step is skipped, and only
        standardization is applied using the standard deviation.
        Defaults to False.

    in_place : bool, optional
        If True, the transformation is applied directly to the input data
        array, modifying it. If False, a new transformed array is returned
        without altering the original data. Defaults to False.

    Returns
    -------
    numpy.ndarray or None
        If `in_place` is False, returns a new array where the SNV
        transformation has been applied. If `in_place` is True,
        the function returns None.
    """
    mean = 0 if norm else data.mean(axis=data.ndim - 1, keepdims=True)
    std = data.std(axis=data.ndim - 1, keepdims=True)
    if in_place:
        data -= mean
        data /= std
        return None
    return (data - mean) / std


@singledispatch
def snv(
    data: SpectrumLike,
    norm: bool = False,
    in_place: bool = False
) -> SpectrumLike | None:
    """
    This function is a generic implementation allowing for the transformation
    of data depending on its type.

    Parameters
    ----------
    data : SpectrumLike
        The data to be transformed. The type of `data` will depend on its
        registered handlers.
    norm : bool, default=False
        If True, no mean substraction is performed.
    in_place : bool, default=False
        Determines whether the operation is performed in-place (if applicable)
        or returns a new transformed object.

    Returns
    -------
    SpectrumLike or None
        Returns the transformed data of the same type as the input if
        `in_place` is False. If `in_place` is True, it operates directly on
        the input data and returns None.

    Raises
    ------
    NotImplementedError
        Raised if no handler for the input data type is registered.
    """
    raise NotImplementedError


@snv.register(np.ndarray)
def _(
    data: np.ndarray,
    norm: bool = False,
    in_place: bool = False
) -> np.ndarray | None:
    return _snv(data, norm, in_place)


@snv.register(Spectrum)
def _(
    data: Spectrum,
    norm: bool = False,
    in_place: bool = False
) -> Spectrum | None:
    return apply_transformation(
        data=data,
        transform_fn=_snv,
        in_place=in_place,
        norm=norm
    )


@snv.register(SpectrumCollection)
def _(
    data: SpectrumCollection,
    norm: bool = False,
    in_place: bool = False
) -> SpectrumCollection | None:
    return apply_transformation(
        data=data,
        transform_fn=_snv,
        in_place=in_place,
        norm=norm
    )


class SNVTransformer(StatelessTransformer):
    """
    Standard Normal Variate (SNV) Transformer.

    This class implements the Standard Normal Variate (SNV) transformation
    often used in spectroscopic data preprocessing to remove scatter and
    normalize data. The transformation adjusts the mean and standard deviation
    of each row of the input dataset.

    Attributes
    ----------
    norm : bool
        If True, applies normalization (no mean substraction)
        during the SNV transformation.
    """
    def __init__(self, norm: bool = False):
        self.norm = norm

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
        Transforms the input data using a Standard Normal Variate (SNV)
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
            Transformed data array after applying the SNV transformation.
            The output has the same shape as the input array `X`.
        """
        X = self._validate_X(X)  # noqa: N806
        return cast(np.ndarray, _snv(X, norm=self.norm, in_place=False))
