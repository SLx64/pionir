import numpy as np
import pytest

from pionir.core.collection import SpectrumCollection
from pionir.core.spectrum import Spectrum
from pionir.scatter.rnv import RNVTransformer, rnv


def test_rnv_ndarray_1d():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rnv(data)
    assert result is not None
    assert np.isclose(result.mean(), 0.0)
    q1, q3 = np.percentile(result, [25, 75])
    assert np.isclose(q3 - q1, 1.0)


def test_rnv_ndarray_2d():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = rnv(data)
    assert result is not None
    assert np.allclose(result.mean(axis=1), 0.0)
    q1, q3 = np.percentile(result, [25, 75], axis=1)
    assert np.allclose(q3 - q1, 1.0)


def test_rnv_custom_iqr():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    iqr = (10, 90)
    result = rnv(data, iqr=iqr)
    assert result is not None
    assert np.isclose(result.mean(), 0.0)
    q_low, q_high = np.percentile(result, [10, 90])
    assert np.isclose(q_high - q_low, 1.0)


def test_rnv_in_place():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    data_copy = data.copy()
    result = rnv(data, in_place=True)
    assert result is None
    assert np.allclose(data.mean(axis=1), 0.0)
    q1, q3 = np.percentile(data, [25, 75], axis=1)
    assert np.allclose(q3 - q1, 1.0)
    assert not np.array_equal(data, data_copy)


def test_rnv_spectrum(sample_spectrum):
    result = rnv(sample_spectrum)
    assert isinstance(result, Spectrum)
    assert np.isclose(result.y.mean(), 0.0)
    q1, q3 = np.percentile(result.y, [25, 75])
    assert np.isclose(q3 - q1, 1.0)
    assert not np.isclose(sample_spectrum.y.mean(), 0.0)


def test_rnv_spectrum_in_place(sample_spectrum):
    result = rnv(sample_spectrum, in_place=True)
    assert result is None
    assert np.isclose(sample_spectrum.y.mean(), 0.0)
    q1, q3 = np.percentile(sample_spectrum.y, [25, 75])
    assert np.isclose(q3 - q1, 1.0)


def test_rnv_collection(sample_collection):
    result = rnv(sample_collection)
    assert isinstance(result, SpectrumCollection)
    assert np.allclose(result.y.mean(axis=1), 0.0)
    q1, q3 = np.percentile(result.y, [25, 75], axis=1)
    assert np.allclose(q3 - q1, 1.0)
    assert not np.allclose(sample_collection.y.mean(axis=1), 0.0)


def test_rnv_collection_in_place(sample_collection):
    result = rnv(sample_collection, in_place=True)
    assert result is None
    assert np.allclose(sample_collection.y.mean(axis=1), 0.0)
    q1, q3 = np.percentile(sample_collection.y, [25, 75], axis=1)
    assert np.allclose(q3 - q1, 1.0)


def test_rnv_transformer():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    transformer = RNVTransformer(iqr=(10, 90))
    transformed = transformer.fit_transform(data)
    assert np.allclose(transformed.mean(axis=1), 0.0)
    q_low, q_high = np.percentile(transformed, [10, 90], axis=1)
    assert np.allclose(q_high - q_low, 1.0)


def test_rnv_not_implemented():
    with pytest.raises(NotImplementedError):
        rnv("not a spectrum") # type: ignore
