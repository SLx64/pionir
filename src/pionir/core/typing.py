import numpy as np

from ..core.collection import SpectrumCollection
from ..core.spectrum import Spectrum

SpectrumLike = np.ndarray | Spectrum | SpectrumCollection