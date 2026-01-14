import numpy as np

from ..core.base import SpectrumBase
from ..core.exceptions import DimensionError
from ..core.metadata import Metadata


class Spectrum(SpectrumBase):
    """
    Represents a spectrum consisting of x and y data along
    with optional metadata.

    This class provides a way to manage and manipulate spectral data,
    including ensuring that the dimensions of x and y are consistent with each
    other. The class can handle metadata associated with the spectrum and
    provides methods for setting and accessing the data.

    Attributes
    ----------
    _x : numpy.ndarray or None
        The x-values of the spectrum. Must have the same shape as the y-values.
    _y : numpy.ndarray or None
        The y-values of the spectrum. Must have the same shape as the x-values.
    _metadata : dict or None
        Optional metadata associated with the spectrum.
    """
    def __init__(
        self,
        x: np.ndarray | list,
        y: np.ndarray | list,
        metadata: Metadata | dict | None = None
    ):
        super().__init__(metadata=metadata)
        self._x: np.ndarray = np.asarray(x)
        self._y: np.ndarray = np.asarray(y)
        self._check_dimensions(self._x, self._y)

    @staticmethod
    def _check_dimensions(x: np.ndarray, y: np.ndarray) -> None:
        if x.shape != y.shape:
            raise DimensionError("Dimensions of x and y must match")

    @property
    def x(self) -> np.ndarray:
        """
        Gets the value of the x attribute.

        Returns
        -------
        np.ndarray or None
            The value of the `_x` attribute. If `_x` is not set,
            it returns None.
        """
        return self._x

    @x.setter
    def x(self, value: np.ndarray | list):
        """
        Sets the value of the 'x' property.

        Validates and assigns the provided value to the 'x' property.
        It ensures that if '_y' is set, the shape of the input value must
        match the shape of '_y'.

        Parameters
        ----------
        value : np.ndarray or list
            The new value to assign to the 'x' property. If provided as a list,
            it will be converted into a NumPy ndarray.

        Raises
        ------
        DimensionError
            If the shape of `value` does not match the shape of '_y'
            (when '_y' is not None).
        """
        value = np.asarray(value)
        self._check_dimensions(self._y, value)
        self._x = value

    @property
    def y(self) -> np.ndarray | None:
        """
        Gets the value of the y attribute.

        Returns
        -------
        np.ndarray or None
            The value of the `_y` attribute. If `_y` is not set,
            it returns None.
        """
        return self._y

    @y.setter
    def y(self, value: np.ndarray | list):
        """
        Setter method for the `y` attribute. Ensures that the provided value
        is a valid array-like object and matches the dimensionality of the `x`
        attribute, if `x` is already set.

        Parameters
        ----------
        value : np.ndarray or list
            The new value to assign to the `y` attribute. Must be a list or
            NumPy array. If `x` is set, the first dimension of `value` must
            match the first dimension of `x`.

        Raises
        ------
        DimensionError
            If the dimensions of `x` and the provided `value` do not match.
        """
        value = np.asarray(value)
        self._check_dimensions(self._x, value)
        self._y = value

    def set_data(self,  x: np.ndarray | list, y: np.ndarray | list) -> None:
        """
        Sets the x and y data for the instance. The method ensures the input
        data arrays are converted to NumPy arrays and that their dimensions
        match.

        Parameters
        ----------
        x : numpy.ndarray or list
            The input data for x. Must align in dimensions with the y data.
        y : numpy.ndarray or list
            The input data for y. Must align in dimensions with the x data.

        Raises
        ------
        DimensionError
            Raised when the dimensions of x and y do not match.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        self._check_dimensions(x, y)
        self._x = x
        self._y = y

    def __len__(self):
        """
        Calculates the length of the object based on the `_x` attribute.

        Returns
        -------
        int
            The length of `_x` if `_x` and `_y` are not `None`, otherwise 0.
        """
        return len(self._x)
