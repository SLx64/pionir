import numpy as np

from .base import SpectrumBase
from .exceptions import DimensionError, WavelengthError
from .typing import SpectrumLike


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
