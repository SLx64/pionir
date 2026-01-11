from abc import ABC, abstractmethod

import numpy as np

from pionir.core.metadata import Metadata


class SpectrumBase(ABC):
    """
    Base class for handling spectral data representation and manipulation.

    This class serves as an abstract base for spectral data structures. It
    provides an interface for manipulating spectral data through `x` and `y`
    attributes, as well as a method for setting data.

    Attributes
    ----------
    _metadata : dict
        Dictionary containing metadata information related to the spectrum.
        If not provided during initialization, an empty metadata dictionary
        will be created.
    """
    def __init__(self, metadata: dict | None = None):
        if metadata is None:
            self._metadata = Metadata()
        else:
            self._metadata = Metadata(metadata)

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    @abstractmethod
    def x(self) -> np.ndarray | None:
        pass

    @x.setter
    @abstractmethod
    def x(self, value: np.ndarray | list):
        pass

    @property
    @abstractmethod
    def y(self) -> np.ndarray | None:
        pass

    @y.setter
    @abstractmethod
    def y(self, value: np.ndarray | list):
        pass

    @abstractmethod
    def set_data(self, x: np.ndarray | list, y: np.ndarray | list) -> None:
        pass
