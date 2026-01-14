import numpy as np
import pytest

from pionir.core.collection import SpectrumCollection
from pionir.core.spectrum import Spectrum
from pionir.scatter.msc import MSCTransformer, _extract_reference_array, msc


def test_extract_reference_array(sample_spectrum, sample_collection):
    arr = np.array([1, 2, 3])
    assert np.array_equal(_extract_reference_array(arr), arr)

    assert np.array_equal(_extract_reference_array(sample_spectrum), sample_spectrum.y)

    expected_ref = sample_collection.average().y
    assert np.array_equal(_extract_reference_array(sample_collection), expected_ref)

    with pytest.raises(ValueError, match="Reference data must be a SpectrumLike object."):
        _extract_reference_array("invalid")  # type: ignore


def test_msc_ndarray_1d():
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data = reference * 2.0 + 0.5  # Linear transformation
    
    result = msc(data, reference)
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, reference)


def test_msc_ndarray_2d():
    reference = np.array([1.0, 2.0, 3.0])
    data = np.array([
        reference * 1.5 + 0.2,
        reference * 0.8 + 0.1
    ])
    
    result = msc(data, reference)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == data.shape
    assert np.allclose(result[0], reference)
    assert np.allclose(result[1], reference)


def test_msc_in_place():
    reference = np.array([1.0, 2.0, 3.0])
    data = reference * 1.5 + 0.2
    
    result = msc(data, reference, in_place=True)
    
    assert result is None
    assert np.allclose(data, reference)


def test_msc_spectrum(sample_spectrum):
    reference = sample_spectrum.y.copy()
    sample_spectrum.y = reference * 1.2 + 0.3
    
    result = msc(sample_spectrum, reference)
    
    assert isinstance(result, Spectrum)
    assert np.allclose(result.y, reference)
    assert not np.allclose(sample_spectrum.y, reference)


def test_msc_spectrum_in_place(sample_spectrum):
    reference = sample_spectrum.y.copy()
    sample_spectrum.y = reference * 1.2 + 0.3
    
    result = msc(sample_spectrum, reference, in_place=True)
    
    assert result is None
    assert np.allclose(sample_spectrum.y, reference)


def test_msc_collection(sample_collection):
    reference = sample_collection.average().y

    sample_collection.y = np.array([
        reference * 1.1 + 0.05 for _ in range(len(sample_collection))
    ])

    result = msc(sample_collection, reference)
    
    assert isinstance(result, SpectrumCollection)
    for s in result:
        assert np.allclose(s.y, reference)


def test_msc_collection_in_place(sample_collection):
    reference = sample_collection.average().y
    for s in sample_collection:
        s.y = reference * 1.1 + 0.05
        
    result = msc(sample_collection, reference, in_place=True)
    
    assert result is None
    for s in sample_collection:
        assert np.allclose(s.y, reference)


def test_msc_collection_no_reference(sample_collection):
    result = msc(sample_collection, reference=None)
    
    assert isinstance(result, SpectrumCollection)
    expected_ref = sample_collection.average().y
    assert np.allclose(result.average().y, expected_ref)


def test_msc_spectrum_no_reference(sample_spectrum):
    with pytest.raises(ValueError, match="Reference must be provided for a single Spectrum."):
        msc(sample_spectrum, reference=None)


def test_msc_transformer():
    reference = np.array([1.0, 2.0, 3.0])
    data = np.array([
        reference * 1.5 + 0.2,
        reference * 0.8 + 0.1
    ])
    
    transformer = MSCTransformer(reference=reference)
    transformed = transformer.fit_transform(data)
    
    assert np.allclose(transformed[0], reference)
    assert np.allclose(transformed[1], reference)


def test_msc_mismatched_length():
    reference = np.array([1.0, 2.0, 3.0])
    data = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError, match="Data and reference must have the same length."):
        msc(data, reference)


def test_msc_not_implemented():
    with pytest.raises(NotImplementedError):
        msc("invalid", None) # type: ignore
