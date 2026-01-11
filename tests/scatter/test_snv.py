import numpy as np
import pytest

from pionir.core.collection import SpectrumCollection
from pionir.core.spectrum import Spectrum
from pionir.scatter.snv import SNVTransformer, snv


def test_snv_ndarray_1d():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = snv(data)
    assert result is not None
    assert np.isclose(result.mean(), 0.0)
    assert np.isclose(result.std(), 1.0)


def test_snv_ndarray_2d():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = snv(data)
    assert result is not None
    assert np.allclose(result.mean(axis=1), 0.0)
    assert np.allclose(result.std(axis=1), 1.0)


def test_snv_norm_true():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = snv(data, norm=True)
    assert result is not None
    assert np.allclose(result.std(axis=1), 1.0)
    assert not np.allclose(result.mean(axis=1), 0.0)


def test_snv_in_place():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data_copy = data.copy()
    result = snv(data, in_place=True)
    assert result is None
    assert np.allclose(data.std(axis=1), 1.0)
    assert np.allclose(data.mean(axis=1), 0.0)
    assert not np.array_equal(data, data_copy)


def test_snv_spectrum():
    x = np.linspace(0, 10, 100)
    y = np.random.rand(100)
    spec = Spectrum(x, y)
    result = snv(spec)
    assert isinstance(result, Spectrum)
    assert np.isclose(result.y.mean(), 0.0)
    assert np.isclose(result.y.std(), 1.0)
    assert not np.isclose(spec.y.std(), 1.0)


def test_snv_spectrum_in_place():
    x = np.linspace(0, 10, 100)
    y = np.random.rand(100)
    spec = Spectrum(x, y)
    result = snv(spec, in_place=True)
    assert result is None
    assert np.isclose(spec.y.mean(), 0.0)
    assert np.isclose(spec.y.std(), 1.0)


def test_snv_collection():
    x = np.linspace(0, 10, 100)
    y1 = np.random.rand(100)
    y2 = np.random.rand(100)
    s1 = Spectrum(x, y1)
    s2 = Spectrum(x, y2)
    col = SpectrumCollection([s1, s2])
    
    result = snv(col)
    assert isinstance(result, SpectrumCollection)
    assert np.allclose(result.y.mean(axis=1), 0.0)
    assert np.allclose(result.y.std(axis=1), 1.0)
    assert not np.allclose(col.y.std(axis=1), 1.0)


def test_snv_collection_in_place():
    x = np.linspace(0, 10, 100)
    y1 = np.random.rand(100)
    y2 = np.random.rand(100)
    s1 = Spectrum(x, y1)
    s2 = Spectrum(x, y2)
    col = SpectrumCollection([s1, s2])
    
    result = snv(col, in_place=True)
    assert result is None
    assert np.allclose(col.y.mean(axis=1), 0.0)
    assert np.allclose(col.y.std(axis=1), 1.0)


def test_snv_transformer():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    transformer = SNVTransformer()
    transformed = transformer.fit_transform(data)
    assert np.allclose(transformed.mean(axis=1), 0.0)
    assert np.allclose(transformed.std(axis=1), 1.0)


def test_snv_transformer_norm():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    transformer = SNVTransformer(norm=True)
    transformed = transformer.transform(data)
    assert np.allclose(transformed.std(axis=1), 1.0)
    assert not np.allclose(transformed.mean(axis=1), 0.0)


def test_snv_not_implemented():
    with pytest.raises(NotImplementedError):
        snv("not a spectrum")
