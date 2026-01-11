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


def test_snv_spectrum(sample_spectrum):
    result = snv(sample_spectrum)
    assert isinstance(result, Spectrum)
    assert np.isclose(result.y.mean(), 0.0)
    assert np.isclose(result.y.std(), 1.0)
    assert not np.isclose(sample_spectrum.y.std(), 1.0)


def test_snv_spectrum_in_place(sample_spectrum):
    result = snv(sample_spectrum, in_place=True)
    assert result is None
    assert np.isclose(sample_spectrum.y.mean(), 0.0)
    assert np.isclose(sample_spectrum.y.std(), 1.0)


def test_snv_collection(sample_collection):
    result = snv(sample_collection)
    assert isinstance(result, SpectrumCollection)
    assert np.allclose(result.y.mean(axis=1), 0.0)
    assert np.allclose(result.y.std(axis=1), 1.0)
    assert not np.allclose(sample_collection.y.std(axis=1), 1.0)


def test_snv_collection_in_place(sample_collection):
    result = snv(sample_collection, in_place=True)
    assert result is None
    assert np.allclose(sample_collection.y.mean(axis=1), 0.0)
    assert np.allclose(sample_collection.y.std(axis=1), 1.0)


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
