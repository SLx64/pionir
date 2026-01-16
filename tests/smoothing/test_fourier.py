import numpy as np
import pytest

from pionir.core.collection import SpectrumCollection
from pionir.core.spectrum import Spectrum
from pionir.smoothing.fourier import (
    FFTSmoothingTransformer,
    _fftsmoothing,
    fftsmoothing,
)


def test_fftsmoothing_ndarray_1d():
    data = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    sigma = 1.0
    result = _fftsmoothing(data, sigma=sigma)
    assert result is not None
    assert result.shape == data.shape
    assert not np.array_equal(result, data)


def test_fftsmoothing_ndarray_2d():
    data = np.array([
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0]
    ])
    sigma = 1.0
    result = _fftsmoothing(data, sigma=sigma)
    assert result is not None
    assert result.shape == data.shape
    for i in range(data.shape[0]):
        assert not np.array_equal(result[i], data[i])


def test_fftsmoothing_deriv():
    data = np.ones(100)
    deriv1 = _fftsmoothing(data, sigma=1.0, deriv=1)
    assert deriv1 is not None
    assert np.allclose(deriv1, 0, atol=1e-10)

    x = np.linspace(0, 2*np.pi, 100, endpoint=False)
    data = np.sin(x)
    deriv1 = _fftsmoothing(data, sigma=10.0, deriv=1)
    assert np.max(np.abs(deriv1)) > 0.05
    assert np.max(np.abs(deriv1)) < 0.07


def test_fftsmoothing_in_place():
    data = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=float)
    data_copy = data.copy()
    result = _fftsmoothing(data, sigma=1.0, in_place=True)
    assert result is None
    assert not np.array_equal(data, data_copy)


def test_fftsmoothing_spectrum(sample_spectrum):
    result = fftsmoothing(sample_spectrum, sigma=1.0)
    assert isinstance(result, Spectrum)
    assert result.y.shape == sample_spectrum.y.shape
    assert not np.array_equal(result.y, sample_spectrum.y)


def test_fftsmoothing_spectrum_in_place(sample_spectrum):
    original_y = sample_spectrum.y.copy()
    result = fftsmoothing(sample_spectrum, sigma=1.0, in_place=True)
    assert result is None
    assert not np.array_equal(sample_spectrum.y, original_y)


def test_fftsmoothing_collection(sample_collection):
    result = fftsmoothing(sample_collection, sigma=1.0)
    assert isinstance(result, SpectrumCollection)
    assert result.y.shape == sample_collection.y.shape
    assert not np.array_equal(result.y, sample_collection.y)


def test_fftsmoothing_collection_in_place(sample_collection):
    original_y = sample_collection.y.copy()
    result = fftsmoothing(sample_collection, sigma=1.0, in_place=True)
    assert result is None
    assert not np.array_equal(sample_collection.y, original_y)


def test_fftsmoothing_parameter_p():
    data = np.random.rand(100)
    res_p1 = _fftsmoothing(data, sigma=1.0, p=1.0)
    res_p2 = _fftsmoothing(data, sigma=1.0, p=2.0)
    assert res_p1 is not None
    assert res_p2 is not None
    assert not np.array_equal(res_p1, res_p2)


def test_fftsmoothing_higher_deriv():
    n = 1000
    x = np.linspace(0, np.pi, n)
    data = np.sin(x)
    deriv2 = _fftsmoothing(data, sigma=1e6, deriv=2)
    assert deriv2 is not None

    expected = - (np.pi / n)**2 * data
    assert np.allclose(deriv2[10:-10], expected[10:-10], atol=1e-5)


def test_fftsmoothing_transformer():
    data = np.random.rand(5, 100)
    transformer = FFTSmoothingTransformer(sigma=1.0, deriv=1)
    transformed = transformer.fit_transform(data)
    assert transformed.shape == data.shape
    
    expected = _fftsmoothing(data, sigma=1.0, deriv=1)
    assert expected is not None
    assert np.allclose(transformed, expected)


def test_fourier_not_implemented():
    with pytest.raises(NotImplementedError):
        fftsmoothing("not a spectrum", sigma=1.0) # type: ignore
