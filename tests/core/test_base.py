import pytest

from pionir.core.base import SpectrumBase


def test_cannot_instantiate_abstract_spectrum_base():
    with pytest.raises(TypeError):
        # noinspection PyAbstractClass
        SpectrumBase()
