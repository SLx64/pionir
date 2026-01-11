import numpy as np
import pytest

from pionir.core.collection import SpectrumCollection
from pionir.core.exceptions import DimensionError
from pionir.core.spectrum import Spectrum


def test_collection_init(sample_collection, random_spectrum_data):
    x, y, _ = random_spectrum_data
    assert len(sample_collection) == 2
    np.testing.assert_array_equal(sample_collection.x, x)
    assert sample_collection.y.shape == (2, len(x))


def test_collection_append(sample_collection, random_spectrum_data):
    x, y, _ = random_spectrum_data
    s3 = Spectrum(x, y, {"id": 3})
    sample_collection.append(s3)
    assert len(sample_collection) == 3
    assert sample_collection.y.shape == (3, len(x))

def test_collection_getitem(sample_collection):
    s = sample_collection[0]
    assert isinstance(s, Spectrum)
    assert s.metadata["id"] == 1


def test_collection_setitem(sample_collection, random_spectrum_data):
    x, y, _ = random_spectrum_data
    s_new = Spectrum(x, y, {"id": 100})
    sample_collection[0] = s_new
    assert sample_collection[0].metadata["id"] == 100


def test_collection_iter(sample_collection):
    ids = [s.metadata["id"] for s in sample_collection]
    assert ids == [1, 2]


def test_collection_set_data(sample_collection, random_spectrum_data):
    x, y, _ = random_spectrum_data
    new_x = x
    new_y = np.zeros((2, len(x)))
    sample_collection.set_data(new_x, new_y)
    np.testing.assert_array_equal(sample_collection.x, new_x)
    np.testing.assert_array_equal(sample_collection.y, new_y)


def test_collection_dimension_error(sample_collection):
    x_wrong = np.array([1, 2, 3])
    y_wrong = np.array([1, 2, 3])
    s_wrong = Spectrum(x_wrong, y_wrong)
    
    with pytest.raises(DimensionError):
        sample_collection.append(s_wrong)


def test_collection_empty():
    c = SpectrumCollection()
    assert len(c) == 0
    assert c.x is None
    assert c.y is None


def test_collection_set_x_updates_spectra(
    sample_collection, random_spectrum_data
):
    x, _, _ = random_spectrum_data
    new_x = x + 1
    sample_collection.x = new_x
    np.testing.assert_array_equal(sample_collection.x, new_x)
    for s in sample_collection:
        np.testing.assert_array_equal(s.x, new_x)
