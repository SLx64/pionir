import pytest

from pionir.core.exceptions import DimensionError


def test_dimension_error_is_exception():
    err = DimensionError("Dimensions of x and y must match")
    assert isinstance(err, Exception)
    assert str(err) == "Dimensions of x and y must match"
