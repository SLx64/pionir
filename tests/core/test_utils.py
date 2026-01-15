from copy import deepcopy

import numpy as np
import pytest

from pionir.core.exceptions import DimensionError, WavelengthError
from pionir.core.spectrum import Spectrum
from pionir.core.utils import (
    _extract_x_array,
    apply_transformation,
    validate_dimensions_1d,
    validate_wavelength,
)


def test_extract_x_array_spectrum(sample_spectrum):
    x = _extract_x_array(sample_spectrum)
    assert np.array_equal(x, sample_spectrum.x)


def test_extract_x_array_collection(sample_collection):
    x = _extract_x_array(sample_collection)
    assert np.array_equal(x, sample_collection.x)


def test_extract_x_array_ndarray():
    arr = np.array([1, 2, 3])
    x = _extract_x_array(arr)
    assert np.array_equal(x, arr)


def test_extract_x_array_invalid():
    with pytest.raises(
        ValueError,
        match="Failed to extract x-values from data."
    ):
        _extract_x_array("invalid")  # ty:ignore[invalid-argument-type]


def test_validate_dimensions_1d_success():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    validate_dimensions_1d(x, y)  # Should not raise


def test_validate_dimensions_1d_failure():
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    with pytest.raises(
        DimensionError,
        match="Dimensions of x and y must match"
    ):
        validate_dimensions_1d(x, y)


def test_validate_wavelength_success(sample_spectrum):
    s2 = deepcopy(sample_spectrum)
    validate_wavelength(sample_spectrum, s2)  # Should not raise


def test_validate_wavelength_mismatch(sample_spectrum):
    s2 = deepcopy(sample_spectrum)
    s2.x = s2.x + 1
    with pytest.raises(
        WavelengthError,
        match="Wavelengths of spectra must match"
    ):
        validate_wavelength(sample_spectrum, s2)


def test_validate_wavelength_dimension_mismatch(sample_spectrum):
    s2 = Spectrum(x=np.array([1, 2]), y=np.array([3, 4]))
    with pytest.raises(DimensionError):
        validate_wavelength(sample_spectrum, s2)


def test_apply_transformation_in_place(sample_spectrum):
    original_y = sample_spectrum.y.copy()
    def mock_transform(y, in_place=False):
        return y * 2
    
    result = apply_transformation(
        data=sample_spectrum,
        transform_fn=mock_transform,
        in_place=True
    )
    
    assert result is None
    assert np.array_equal(sample_spectrum.y, original_y * 2)


def test_apply_transformation_not_in_place(sample_spectrum):
    original_y = sample_spectrum.y.copy()
    def mock_transform(y, in_place=False):
        return y * 2
    
    result = apply_transformation(
        data=sample_spectrum,
        transform_fn=mock_transform,
        in_place=False
    )
    
    assert result is not sample_spectrum
    assert np.array_equal(result.y, original_y * 2)
    assert np.array_equal(sample_spectrum.y, original_y)


def test_apply_transformation_with_kwargs(sample_spectrum):
    original_y = sample_spectrum.y.copy()
    def mock_transform(y, in_place=False, factor=1):
        return y * factor
    
    apply_transformation(
        data=sample_spectrum,
        transform_fn=mock_transform,
        in_place=True, factor=3
    )
    assert np.array_equal(sample_spectrum.y, original_y * 3)
