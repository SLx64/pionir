from copy import deepcopy
from typing import Any, Callable, TypeVar, cast

import numpy as np

from .base import SpectrumBase
from .exceptions import DimensionError, WavelengthError
from .typing import SpectrumLike

S = TypeVar("S", bound=SpectrumBase)


def _extract_x_array(data: SpectrumLike) -> np.ndarray:
    """
    Extracts the x-values from the given data object, which may be of
    various types.

    Parameters
    ----------
    data : SpectrumLike
        The input data object from which x-values are to be extracted.
        Supported types include `Spectrum`, `SpectrumCollection`,
        or `np.ndarray`.

    Returns
    -------
    np.ndarray
        A NumPy array containing the extracted x-values.

    Raises
    ------
    ValueError
        If the input data type is not supported or x-values cannot
        be extracted.
    """
    match data:
        case SpectrumBase():
            return data.x
        case np.ndarray():
            return data
        case _:
            raise ValueError("Failed to extract x-values from data.")


def validate_dimensions_1d(x: np.ndarray, y: np.ndarray) -> None:
    """
    Validates that the dimensions of two 1D arrays match.

    Parameters
    ----------
    x : numpy.ndarray
        The first 1D array to validate.
    y : numpy.ndarray
        The second 1D array to validate.

    Raises
    ------
    DimensionError
        If the dimensions of `x` and `y` do not match.
    """
    if x.shape != y.shape:
        raise DimensionError("Dimensions of x and y must match")


def validate_wavelength(s1: SpectrumLike, s2: SpectrumLike) -> None:
    """
    Validates whether the wavelengths of two spectra match.

    This function extracts the x-axis (wavelength) data from two input spectra
    and ensures that their dimensionality and values are compatible. It first
    checks that both spectra have the same dimensions, and then verifies that
    the wavelengths align within a tolerance. If not, it raises a
    `WavelengthError`.

    Parameters
    ----------
    s1 : SpectrumLike
        The first spectrum to validate.
    s2 : SpectrumLike
        The second spectrum to validate.

    Raises
    ------
    WavelengthError
        If the wavelengths of the two spectra do not match within a specified
        tolerance.
    DimensionError
        If the dimensions of the spectra do not match.
    """
    x1 = _extract_x_array(s1)
    x2 = _extract_x_array(s2)
    validate_dimensions_1d(x1, x2)
    if not np.allclose(x1, x2):
        raise WavelengthError("Wavelengths of spectra must match")


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
