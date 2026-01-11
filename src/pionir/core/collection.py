from typing import Iterator

import numpy as np

from ..core.base import SpectrumBase
from ..core.exceptions import DimensionError
from ..core.metadata import Metadata
from ..core.spectrum import Spectrum


class SpectrumCollection(SpectrumBase):
    """
    Represents a collection of spectrum objects and manages their dimensional
    consistency.

    This class is designed to store and manipulate multiple `Spectrum`
    objects, ensuring that their dimensional attributes (e.g., x and y) are
    compatible.s.

    Attributes
    ----------
    _x : numpy.ndarray or None
        The shared x-axis values of the spectra, or None if not set.
    _y : numpy.ndarray or None
        The aggregated y-axis values of the spectra, or None if not set.
    """
    def __init__(
        self, spectra: list | None = None,
        metadata: Metadata | dict | None = None
    ):
        super().__init__(metadata=metadata)
        self._spectra: list[Spectrum] = []
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._stale: bool = True

        if spectra:
            for s in spectra:
                self.append(s)

    def _recalculate(self):
        """
        Recalculates the internal state based on the current spectra.

        This method calculates and updates the internal `_y` array depending
        on the state of the `_stale` attribute.
        """
        if not len(self._spectra):
            self._x = None
            self._y = None
            return

        self._y = np.array([s.y for s in self._spectra])
        self._stale = False

    def _validate_dim(self, spectrum: Spectrum) -> bool:
        """
        Validates the dimensional compatibility of the given spectrum with
        the current object. This method ensures that the x-axis and y-axis
        dimensions of the input spectrum match or comply with the dimensional
        requirements of the current object.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum object to validate. It should have valid x and y
            attributes to represent the data.

        Returns
        -------
        bool
            True if the dimensions of the spectrum are compatible,
            False otherwise.

        Raises
        ------
        DimensionError
            If the spectrum lacks x or y values, an error is raised
            indicating invalid dimensional data.
        """
        if spectrum.x is None or spectrum.y is None:
            raise DimensionError("Spectrum must have x and y values")
        if self.x is None:
            return True
        if not np.array_equal(spectrum.x, self.x):
            return False
        if len(self._spectra) > 0 and self._y is not None:
            return spectrum.y.shape == self._y.shape[1]
        return True

    @property
    def x(self) -> np.ndarray | None:
        """
        This property retrieves the value of the `_x` attribute if it has
        been set.

        Returns
        -------
        np.ndarray or None
            The value of the `_x` attribute.
        """
        return self._x

    @x.setter
    def x(self, value: np.ndarray | list):
        """
        Sets the x attribute with the given values and validates the
        dimensions if the y attribute is not None.

        Parameters
        ----------
        value : numpy.ndarray or list
            The values to set for the x attribute. Must match the shape of the
            existing y attribute, if defined.
        """
        x = np.asarray(value).copy()
        if self._y is not None and self._y.shape[1] != len(value):
            raise DimensionError(
                "Dimension of x values must match the dimensions of "
                "y values"
            )
        self._x = x
        for spectrum in self._spectra:
            spectrum.x = x

    @property
    def y(self) -> np.ndarray | None:
        """
        This property retrieves the value of the `_y` attribute if it has
        been set. If the cached value is stale, it triggers a recalculation
        before returning the value.

        Returns
        -------
        np.ndarray or None
            The current or recalculated value of y. If no value is available,
            None is returned.
        """
        if self._stale:
            self._recalculate()
        return self._y

    @y.setter
    def y(self, value: np.ndarray | list):
        """
        Sets the y-values for the spectra collection while ensuring the
        given values match the required dimensions of the collection.

        Parameters
        ----------
        value : numpy.ndarray or list
            A 2D array or list containing the y-values to be assigned to
            the spectra collection. The number of rows must match the number
            of spectra in the collection.
        """
        y = np.asarray(value).copy()
        if self.y is None:
            raise DimensionError("Cannot set y values on an empty collection.")
        if y.shape[0] != len(self._spectra):
            raise DimensionError(
                "Number of rows in y must match the number of "
                "spectra in the collection"
            )
        for i, spectrum in enumerate(self._spectra):
            spectrum.y = y[i, :]

        # the new y values already known here and we don't need to recalculate
        self._y = y
        self._stale = False

    def set_data(self, x: np.ndarray | list, y: np.ndarray | list) -> None:
        """
        Sets the data for the spectra collection by assigning x and y values.
        The provided x and y arrays must conform to specific dimensional
        constraints.

        Parameters
        ----------
        x : np.ndarray or list
            The new x-axis data to be assigned to each spectrum in the
            collection. It must be compatible with the y dimensions.
        y : np.ndarray or list
            The new y-axis data to be assigned to the collection. It should be
            a 2D array where the number of rows corresponds to the number of
            spectra in the collection and the number of columns matches the
            length of x.

        Raises
        ------
        DimensionError
            If the dimensions of x and y do not match, or if the number of
            rows in y does not correspond to the number of spectra in the
            collection.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[0] != y.shape[1]:
            raise DimensionError("Dimensions of x and y must match")
        if len(self._spectra) != y.shape[0]:
            raise DimensionError(
                "Number of rows in y must match the number of "
                "spectra in the collection"
            )
        self._x, self._y = None, None
        self.x = x
        self.y = y

    def append(self, spectrum):
        """
        Appends a spectrum to the current collection of spectra.

        This method validates the dimensions of the given spectrum against
        the  existing spectra in the collection. If the dimensions match,
        the spectrum  is appended to the internal list of spectra.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum to be appended. It must have dimensions compatible
            with the existing spectra.

        Raises
        ------
        DimensionError
            Raised when the dimensions of the provided spectrum do not match
            the existing spectra.
        """
        if not self._validate_dim(spectrum):
            raise DimensionError("Spectrum dimensions do not match.")
        if self._x is None:
            self._x = spectrum.x.copy()
        self._spectra.append(spectrum)
        self._stale = True

    def __setitem__(self, index: int, spectrum: Spectrum) -> None:
        """
        Sets the spectrum at the specified index in the spectra container.

        This method replaces the spectrum at the given index with a new
        spectrum,  validating its dimensions first to ensure they match the
        expected dimensions.

        Parameters
        ----------
        index : int
            The index at which the spectrum will be set.

        spectrum : Spectrum
            The spectrum to set at the specified index. Its dimensions
            are validated before being set.

        Raises
        ------
        DimensionError
            If the dimensions of the provided spectrum do not match the
            expected dimensions.
        """
        if not self._validate_dim(spectrum):
            raise DimensionError("Spectrum dimensions do not match.")
        self._spectra[index] = spectrum
        self._stale = True

    def __getitem__(self, index: int) -> Spectrum:
        """
        Retrieve a Spectrum object from the collection by its index.

        Parameters
        ----------
        index : int
            The index of the Spectrum object to retrieve from the collection.

        Returns
        -------
        Spectrum
            The Spectrum object located at the specified index.
        """
        return self._spectra[index]

    def __iter__(self) -> Iterator[Spectrum]:
        """
        Returns an iterator over the spectra in the collection.

        Yields
        ------
        Iterator[Spectrum]
            An iterator that yields `Spectrum` objects from the collection.
        """
        return iter(self._spectra)

    def __len__(self) -> int:
        """
        Returns the number of elements in the collection.

        Returns
        -------
        int
            The number of elements in the collection.
        """
        return len(self._spectra)
