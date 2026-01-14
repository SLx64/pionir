from typing import Iterator

import numpy as np

from .base import SpectrumBase
from .exceptions import DimensionError
from .metadata import Metadata
from .spectrum import Spectrum
from .utils import validate_wavelength


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
            self._recalculate()

    def _recalculate(self):
        """
        Recalculates internal state variables.

        This method triggers recalculations of internal `_x`and `_y`
        attributes. It ensures that dependent internal states are updated
        correctly based on changes in the system.

        Notes
        -----
        This method is internal to the class and operates on
        private attributes to maintain consistency.
        """
        self._recalculate_y()
        self._recalculate_x()

    def _recalculate_y(self):
        """
        Recalculates the internal state based on the current spectra.

        This method calculates and updates the internal `_y` array depending
        on the state of the `_stale` attribute.
        """
        if not len(self._spectra):
            self._y = None
            return

        self._y = np.array([s.y for s in self._spectra])
        self._stale = False

    def _recalculate_x(self):
        """
        Recalculates the internal state based on the current spectra.

        This method calculates and updates the internal `_y` array depending
        on the state of the `_stale` attribute.
        """
        if not len(self._spectra):
            self._x = None
            return

        self._x = self._spectra[0].x
        self._stale = False

    @property
    def x(self) -> np.ndarray:
        """
        This property retrieves the value of the `_x` attribute if it has
        been set.

        Returns
        -------
        np.ndarray or None
            The value of the `_x` attribute.
        """
        if self._x is None:
            return np.array([])
        return self._x

    @x.setter
    def x(self, value: np.ndarray | list):
        """
        Sets the x attribute with the given values and updates the x values
        in all contained spectra. Dimension checks are performed on the
        `Spectrum` objects before setting the new values.

        Parameters
        ----------
        value : numpy.ndarray or list
            The values to set for the x attribute. Must match the shape of the
            existing y attribute, if defined.
        """
        x = np.asarray(value).copy()
        for spectrum in self._spectra:
            spectrum.x = x
        self._x = x

    @property
    def y(self) -> np.ndarray:
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
            self._recalculate_y()
        if self._y is None:
            return np.array([])
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
        if not len(self._spectra):
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

    def average(self) -> Spectrum:
        """
        Computes the average of the spectra stored in the collection.

        If there is only one spectrum in the collection, it returns a copy
        of that spectrum's data. If there are multiple spectra, the mean of
        the `y` values across all spectra is computed.

        Raises
        ------
        ValueError
            If the collection is empty or contains spectra with no
            valid data.

        Returns
        -------
         Spectrum
            The averaged spectrum, containing the averaged `y` values and
            the corresponding `x` values.
        """
        if len(self._spectra) == 0:
            raise ValueError("Cannot average an empty collection.")
        if len(self._spectra) == 1:
            s = self._spectra[0]
            return Spectrum(
                x=s.x.copy(),
                y=s.y.copy()
            )
        return Spectrum(
            x=self.x.copy(),
            y=self.y.mean(axis=0)
        )

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
        if len(self._spectra):
            validate_wavelength(spectrum, self)
        else:
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
        WavelengthError
            If the wavelengths of the two spectra do not match within
            a specified tolerance.
        DimensionError
            If the dimensions of the spectra do not match.
        KeyError
            If the index is out of bounds.
        """
        validate_wavelength(spectrum, self)
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
