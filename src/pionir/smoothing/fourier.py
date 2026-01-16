from functools import singledispatch
from typing import Any, Self, cast

import numpy as np
from scipy.signal.windows import general_gaussian

from ..core.collection import SpectrumCollection
from ..core.spectrum import Spectrum
from ..core.typing import SpectrumLike
from ..core.utils import apply_transformation
from ..sklearn.utils import StatelessTransformer


def _fftsmoothing(
    data: np.ndarray,
    sigma: float,
    p: float = 1,
    deriv: int = 0,
    in_place: bool = False
) -> np.ndarray | None:
    """
    Applies Fourier transform-based smoothing and optionally calculates
    derivatives of the input data.

    This function performs smoothing by multiplying the Fourier transform of
    the data with a General Gaussian window. It can also compute arbitrary
    derivatives in the frequency domain.

    Parameters
    ----------
    data : np.ndarray
        The input data to be smoothed. Can be a 1D or 2D array. If 2D, the
        transformation is applied along the last axis (axis=-1).
    sigma : float
        The standard deviation of the General Gaussian window, controlling the
        degree of smoothing.
    p : float, default 1
        The shape parameter of the General Gaussian window. p=1 corresponds to
        a Gaussian window, while p=2 corresponds to a window that is flatter in
        the middle.
    deriv : int, default 0
        The order of the derivative to compute. 0 means no derivative (only
        smoothing).
    in_place : bool, default False
        If True, the transformation is performed in-place on the input array,
        and the function returns None. If False, a new array is returned.

    Returns
    -------
    np.ndarray or None
        The transformed data as a new array if `in_place` is False. Otherwise,
        returns None.
    """
    tmp = np.concatenate([data, np.flip(data, axis=-1)], axis=-1)

    length = tmp.shape[-1]
    window = np.roll(general_gaussian(length, p, sigma), length // 2)

    f_tmp = np.fft.fft(tmp, axis=-1)
    f_tmp *= window

    if deriv > 0:
        freqs = np.fft.fftfreq(length)
        f_tmp *= (1j * 2 * np.pi * freqs) ** deriv

    tmp_f = np.real(np.fft.ifft(f_tmp, axis=-1))[..., :data.shape[-1]]

    if in_place:
        data[:] = tmp_f
        return None
    return tmp_f


@singledispatch
def fftsmoothing(
    data: SpectrumLike,
    sigma: float,
    p: float = 1,
    deriv: int = 0,
    in_place: bool = False
) -> SpectrumLike | None:
    """
    Apply Fourier smoothing and optional derivation to the given data.

    This function is a generic implementation allowing for the transformation
    of data depending on its type.

    Parameters
    ----------
    data : SpectrumLike
        The input data to be smoothed. Typically, this is a multidimensional
        data structure or array representing signals or spectra.
    sigma : float
        The standard deviation of the General Gaussian window.
    p : float, default 1
        The shape parameter of the General Gaussian window.
    deriv : int, default 0
        The order of the derivative to compute.
    in_place : bool, default False
        If True, the transformation is performed in-place. Defaults to False.

    Returns
    -------
    SpectrumLike or None
        The transformed data. If `in_place` is False, it returns a new instance
        of the transformed data. If `in_place` is True, it returns None.

    Raises
    ------
    NotImplementedError
        Raised if no handler for the input data type is registered.
    """
    raise NotImplementedError


@fftsmoothing.register(np.ndarray)
def _(
    data: np.ndarray,
    sigma: float,
    p: float = 1,
    deriv: int = 0,
    in_place: bool = False
) -> np.ndarray | None:
    return _fftsmoothing(data, sigma, p, deriv, in_place)


@fftsmoothing.register(Spectrum)
def _(
    data: Spectrum,
    sigma: float,
    p: float = 1,
    deriv: int = 0,
    in_place: bool = False
) -> Spectrum | None:
    return apply_transformation(
        data=data,
        transform_fn=_fftsmoothing,
        in_place=in_place,
        sigma=sigma,
        p=p,
        deriv=deriv
    )


@fftsmoothing.register(SpectrumCollection)
def _(
    data: SpectrumCollection,
    sigma: float,
    p: float = 1,
    deriv: int = 0,
    in_place: bool = False
) -> SpectrumCollection | None:
    return apply_transformation(
        data=data,
        transform_fn=_fftsmoothing,
        in_place=in_place,
        sigma=sigma,
        p=p,
        deriv=deriv
    )


class FFTSmoothingTransformer(StatelessTransformer):
    """
    Fourier transform smoothing and derivation Transformer.

    Attributes
    ----------
    sigma : float
        The standard deviation of the General Gaussian window.
    p : float
        The shape parameter of the General Gaussian window.
    deriv : int
        The order of the derivative to compute.
    """
    def __init__(self, sigma: float, p: float = 1, deriv: int = 0):
        self.sigma = sigma
        self.p = p
        self.deriv = deriv

    def fit(self, X: np.ndarray, y: Any | None = None) -> Self:  # noqa: N803
        """
        Fits the model to the provided data. Has no effect on the model in
        this case.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit the model.
        y : Any or None, optional
            Optional target variable.

        Returns
        -------
        Self
            The instance of the fitted model.
        """
        self._validate_X(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """
        Transforms the data using Fourier transform smoothing.

        Parameters
        ----------
        X : np.ndarray
            Input data array to be transformed. It is expected to be a
            2D array where rows represent samples and columns represent
            features.

        Returns
        -------
        np.ndarray
            Transformed data array after applying the Savitzky-Golay filter.
            The output has the same shape as the input array `X`.
        """
        X = self._validate_X(X)  # noqa: N806
        return cast(np.ndarray, _fftsmoothing(
            X,
            sigma=self.sigma,
            p=self.p,
            deriv=self.deriv,
            in_place=False
        ))
