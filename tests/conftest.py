import numpy as np
import pytest

from pionir.core.collection import SpectrumCollection
from pionir.core.metadata import Metadata
from pionir.core.spectrum import Spectrum


@pytest.fixture(scope="session")
def random_spectrum_data():
    """Precalculate some data for all tests."""
    x = np.linspace(0, 10, 100)
    y = np.random.rand(len(x))
    metadata = Metadata({"sample": "A", "temperature": 25})
    return x, y, metadata


@pytest.fixture
def sample_spectrum(random_spectrum_data):
    x, y, metadata = random_spectrum_data
    return Spectrum(x=x, y=y, metadata=metadata)


@pytest.fixture
def sample_collection(random_spectrum_data):
    x, y, _ = random_spectrum_data
    s1 = Spectrum(x, y, {"id": 1})
    s2 = Spectrum(x, y, {"id": 2})
    return SpectrumCollection([s1, s2])
