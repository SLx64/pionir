import numpy as np
import pytest
from scipy.signal import savgol_filter

from pionir.core.collection import SpectrumCollection
from pionir.core.spectrum import Spectrum
from pionir.smoothing.savgol import SGFTransformer, savgol


def test_savgol_ndarray_1d():
    data = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    window_length = 5
    polyorder = 2
    
    result = savgol(data, window_length=window_length, polyorder=polyorder)
    expected = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    
    assert result is not None
    assert np.allclose(result, expected)


def test_savgol_ndarray_2d():
    data = np.array([
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0]
    ])
    window_length = 5
    polyorder = 2
    
    result = savgol(data, window_length=window_length, polyorder=polyorder)
    expected = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    
    assert result is not None
    assert np.allclose(result, expected)


def test_savgol_deriv():
    data = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    window_length = 5
    polyorder = 2
    deriv = 1
    
    result = savgol(data, window_length=window_length, polyorder=polyorder, deriv=deriv)
    expected = savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=deriv)
    
    assert result is not None
    assert np.allclose(result, expected)


def test_savgol_in_place():
    data = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0], dtype=float)
    data_copy = data.copy()
    window_length = 5
    polyorder = 2
    
    expected = savgol_filter(data_copy, window_length=window_length, polyorder=polyorder)
    result = savgol(data, window_length=window_length, polyorder=polyorder, in_place=True)
    
    assert result is None
    assert np.allclose(data, expected)
    assert not np.array_equal(data, data_copy)


def test_savgol_spectrum(sample_spectrum):
    window_length = 5
    polyorder = 2
    
    if len(sample_spectrum.y) < window_length:
        pytest.skip("Sample spectrum too short for test")

    expected_y = savgol_filter(sample_spectrum.y, window_length=window_length, polyorder=polyorder)
    result = savgol(sample_spectrum, window_length=window_length, polyorder=polyorder)
    
    assert isinstance(result, Spectrum)
    assert np.allclose(result.y, expected_y)
    assert not np.array_equal(result.y, sample_spectrum.y)


def test_savgol_spectrum_in_place(sample_spectrum):
    window_length = 5
    polyorder = 2
    
    if len(sample_spectrum.y) < window_length:
        pytest.skip("Sample spectrum too short for test")

    expected_y = savgol_filter(sample_spectrum.y, window_length=window_length, polyorder=polyorder)
    result = savgol(sample_spectrum, window_length=window_length, polyorder=polyorder, in_place=True)
    
    assert result is None
    assert np.allclose(sample_spectrum.y, expected_y)


def test_savgol_collection(sample_collection):
    window_length = 5
    polyorder = 2
    
    if sample_collection.y.shape[1] < window_length:
        pytest.skip("Sample collection spectra too short for test")

    expected_y = savgol_filter(sample_collection.y, window_length=window_length, polyorder=polyorder)
    result = savgol(sample_collection, window_length=window_length, polyorder=polyorder)
    
    assert isinstance(result, SpectrumCollection)
    assert np.allclose(result.y, expected_y)
    assert not np.array_equal(result.y, sample_collection.y)


def test_savgol_collection_in_place(sample_collection):
    window_length = 5
    polyorder = 2
    
    if sample_collection.y.shape[1] < window_length:
        pytest.skip("Sample collection spectra too short for test")

    expected_y = savgol_filter(sample_collection.y, window_length=window_length, polyorder=polyorder)
    result = savgol(sample_collection, window_length=window_length, polyorder=polyorder, in_place=True)
    
    assert result is None
    assert np.allclose(sample_collection.y, expected_y)


def test_savgol_transformer():
    data = np.array([
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0]
    ])
    window_length = 5
    polyorder = 2
    
    transformer = SGFTransformer(window_length=window_length, polyorder=polyorder)
    transformed = transformer.fit_transform(data)
    
    expected = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    assert np.allclose(transformed, expected)


def test_savgol_not_implemented():
    with pytest.raises(NotImplementedError):
        savgol("not a spectrum", window_length=5, polyorder=2) # type: ignore
