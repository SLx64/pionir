from copy import deepcopy
from functools import singledispatch

import numpy as np

from ..core.collection import SpectrumCollection
from ..core.spectrum import Spectrum
from ..core.typing import SpectrumLike


@singledispatch
def snv(
    data: SpectrumLike,
    norm: bool = False,
    in_place: bool = False
) -> SpectrumLike | None:
    """
    Apply Standard Normal Variate (SNV) transformation to input data.

    This function is a generic implementation allowing for the transformation
    of data depending on its type. The SNV technique standardizes a matrix or
    vector by centering and scaling.

    SNV transformation formula:
        For each row or data instance:
            standardized = (row - mean) / std

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
    mean = 0 if norm else data.mean(axis=data.ndim - 1, keepdims=True)
    std = data.std(axis=data.ndim - 1, keepdims=True)
    if in_place:
        data -= mean
        data /= std
        return None
    return (data - mean) / std


@snv.register(Spectrum)
def _(
    data: Spectrum,
    norm: bool = False,
    in_place: bool = False
) -> Spectrum | None:
    if in_place:
        snv(data.y, norm, in_place=True)
        return None
    new_spectrum = deepcopy(data)
    snv(new_spectrum.y, norm, in_place=True)
    return new_spectrum


@snv.register(SpectrumCollection)
def _(
    data: SpectrumCollection,
    norm: bool = False,
    in_place: bool = False
) -> SpectrumCollection | None:
    if in_place:
        snv(data.y, norm, in_place=True)
        return None
    new_collection = deepcopy(data)
    snv(new_collection.y, norm, in_place=True)
    return new_collection
