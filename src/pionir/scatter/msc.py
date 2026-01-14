from copy import deepcopy
from functools import singledispatch
from typing import Any, Self, cast

import numpy as np

from ..core.collection import SpectrumCollection
from ..core.spectrum import Spectrum
from ..core.typing import SpectrumLike
from ..sklearn.utils import StatelessTransformer


def _extract_reference_array(data: SpectrumLike) -> np.ndarray:
    """
    Extracts and returns the reference array from a SpectrumLike object.

    For a `SpectrumCollection` object, the function computes the average of
    the collection and returns the `y` attribute of the resulting spectrum.

    Parameters
    ----------
    data : SpectrumLike
        The input data from which the reference array is extracted.
        Must be an instance of `Spectrum`, `SpectrumCollection`, or
        a `numpy.ndarray`.

    Returns
    -------
    numpy.ndarray
        The extracted reference array from the input `SpectrumLike` object.

    Raises
    ------
    ValueError
        If the provided input is not a valid `SpectrumLike` object.
    """
    match data:
        case Spectrum():
            return data.y
        case SpectrumCollection():
            return data.average().y
        case np.ndarray():
            return data
        case _:
            raise ValueError("Reference data must be a SpectrumLike object.")


@singledispatch
def msc(
    data: SpectrumLike,
    reference: SpectrumLike | None,
    in_place: bool = False
) -> SpectrumLike | None:
    """
    Apply Multiplicative Scatter Correction (MSC) to spectral data.

    This  method normalizes spectral data by adjusting for scattering effects,
    using either a provided reference spectrum or calculating the mean
    spectrum if no reference is supplied.

    Parameters
    ----------
    data : SpectrumLike
        The data to be transformed. The type of `data` will depend on its
        registered handlers.

    reference : SpectrumLike
        The reference spectrum to be used for MSC normalization.
        If None, the mean spectrum of the input data will
        be used as the reference if `data` is a SpectrumCollection.

    in_place : bool, optional
        If True, the input data will be modified in place.
        If False, a new corrected dataset will be created and returned.

    Returns
    -------
    SpectrumLike or None
        The corrected spectral data, returned as `SpectrumLike`.
        If `in_place` is True, the function returns None
        as the input dataset is modified directly.
    """
    raise NotImplementedError


@msc.register(np.ndarray)
def _(
    data: np.ndarray,
    reference: SpectrumLike,
    in_place: bool = False
) -> np.ndarray | None:
    reference = _extract_reference_array(reference)
    if reference.shape[-1] != data.shape[-1]:
        raise ValueError(f"Data and reference must have the same length. "
                         f"({data.shape[-1]} != {reference.shape[-1]})")

    # Fit each row of data against reference
    # data is (N, samples) or (samples,)
    # reference is (samples,)
    if data.ndim == 1:
        fit = np.polyfit(reference, data, 1)
        if in_place:
            data -= fit[1]
            data /= fit[0]
            return None
        return (data - fit[1]) / fit[0]
    
    # For 2D data, we fit each row
    if in_place:
        for i in range(len(data)):
            fit = np.polyfit(reference, data[i], 1)
            data[i] -= fit[1]
            data[i] /= fit[0]
        return None
    
    corrected = np.empty_like(data)
    for i in range(len(data)):
        fit = np.polyfit(reference, data[i], 1)
        corrected[i] = (data[i] - fit[1]) / fit[0]
    return corrected


@msc.register(Spectrum)
def _(
    data: Spectrum,
    reference: SpectrumLike,
    in_place: bool = False
) -> Spectrum | None:
    if reference is None:
        raise ValueError("Reference must be provided for a single Spectrum.")
    if in_place:
        corrected_y = cast(np.ndarray, msc(data.y, reference, in_place=False))
        data.y = corrected_y
        return None
    new_spectrum = deepcopy(data)
    msc(new_spectrum, reference, in_place=True)
    return new_spectrum


@msc.register(SpectrumCollection)
def _(
    data: SpectrumCollection,
    reference: SpectrumLike | None,
    in_place: bool = False
) -> SpectrumCollection | None:
    if reference is None:
        reference = data.average().y
    
    if in_place:
        # data.y returns a copy, so we must set it back after in-place modification
        # OR we can just use the corrected array from non-in-place call
        corrected_y = cast(np.ndarray, msc(data.y, reference, in_place=False))
        data.y = corrected_y
        return None
    new_collection = deepcopy(data)
    msc(new_collection, reference, in_place=True)
    return new_collection


class MSCTransformer(StatelessTransformer):
    """
    A transformer for applying Multiplicative Scatter Correction (MSC)
    to spectral data.

    Attributes
    ----------
    reference : SpectrumLike
        The reference spectrum to which the input spectra are aligned.
    """
    def __init__(self, reference: SpectrumLike):
        self.reference = reference

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
        Transforms the input data using a Multiplicative Scatter
        Correction (MSC) transformation.

        Parameters
        ----------
        X : np.ndarray
            Input data array to be transformed. It is expected to be a
            2D array where rows represent samples and columns represent
            features.

        Returns
        -------
        np.ndarray
            Transformed data array after applying the MSC transformation.
            The output has the same shape as the input array `X`.
        """
        X = self._validate_X(X)  # noqa: N806
        return cast(np.ndarray, msc(X, self.reference, in_place=False))
