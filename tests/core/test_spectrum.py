import numpy as np
import pytest

from pionir.core.exceptions import DimensionError
from pionir.core.spectrum import Spectrum


def test_spectrum_init(random_spectrum_data):
    x, y, metadata = random_spectrum_data
    s = Spectrum(x, y, metadata)
    np.testing.assert_array_equal(s.x, x)
    np.testing.assert_array_equal(s.y, y)
    assert s.metadata["sample"] == "A"


def test_spectrum_setters(random_spectrum_data):
    x, y, _ = random_spectrum_data
    s = Spectrum(x, y)
    
    new_x = x * 2
    s.x = new_x
    np.testing.assert_array_equal(s.x, new_x)
    
    new_y = y * 2
    s.y = new_y
    np.testing.assert_array_equal(s.y, new_y)


def test_spectrum_dimension_error(random_spectrum_data):
    x, y, _ = random_spectrum_data
    s = Spectrum(x, y)
    
    with pytest.raises(DimensionError):
        s.x = np.array([1, 2, 3])
        
    with pytest.raises(DimensionError):
        s.y = np.array([1, 2, 3])


def test_spectrum_set_data(random_spectrum_data):
    x, y, _ = random_spectrum_data
    s = Spectrum(x, y)
    new_x = np.linspace(0, 5, 50)
    new_y = np.cos(new_x)
    s.set_data(new_x, new_y)
    np.testing.assert_array_equal(s.x, new_x)
    np.testing.assert_array_equal(s.y, new_y)


def test_spectrum_len(random_spectrum_data):
    x, y, _ = random_spectrum_data
    s = Spectrum(x, y)
    assert len(s) == len(x)


def test_spectrum_empty_len():
    s = Spectrum([], [])
    assert len(s) == 0
