from copy import deepcopy
from typing import Any, Callable, TypeVar, cast

import numpy as np

from .base import SpectrumBase

S = TypeVar("S", bound=SpectrumBase)


def apply_transformation(
    data: S,
    *,
    transform_fn: Callable[..., np.ndarray | None],
    in_place: bool,
    **kwargs: Any
) -> S | None:
    """
    Applies a transformation to the spectral data.

    This function applies a given transformation to the spectral data object.
    It can either modify the data in place or return a new object with the
    transformed data, depending on the `in_place` parameter. The
    transformation is applied to the `y` attribute of the `SpectrumBase`
    object. Additional arguments for the transformation can be passed
    through `kwargs`.

    Parameters
    ----------
    data : SpectrumBase
        The input spectral data object to which the transformation is
        to be applied.
    transform_fn : Callable[..., np.ndarray]
        A callable that performs the transformation on the data's `y`
        attribute. It must accept the `y` attribute of the `SpectrumBase`
        object and any additional arguments passed through `kwargs`.
    in_place : bool
        If True, modifies the input `data` object directly. If False, creates
        and returns a new `SpectrumBase` object with the transformed data.
    **kwargs : dict
        Additional keyword arguments to be passed to the `transform` callable.

    Returns
    -------
    SpectrumBase or None
        Returns a new `SpectrumBase` object with the transformed data if
        `in_place` is False. If `in_place` is True, returns None since the
        modification is performed directly on the input `data` object.

    Notes
    -----
    If `in_place` is set to True, the input `SpectrumBase` object will be
    directly modified, and no new object will be created. Ensure that this
    behavior is intentional to avoid unintentionally altering the
    original data.
    """
    y = cast(np.ndarray, transform_fn(data.y, in_place=False, **kwargs))
    if in_place:
        data.y = y
        return None
    new_data = deepcopy(data)
    new_data.y = y
    return new_data
