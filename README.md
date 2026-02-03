# PioNIR

PioNIR is a Python library designed for the processing and analysis of near-infrared (NIR) spectra. It offers a suite of chemometric tools for preprocessing, modeling, and evaluation, facilitating the development of spectroscopy workflows. The library tightly integrates with the scientific Python ecosystem, with [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [scikit-learn](https://scikit-learn.org/stable/).

## Installation

The latest release is available on PyPI and can be installed via pip.

```shell
pip install pionir
```

For development purposes, use [uv](https://docs.astral.sh/uv/) to manage the installation.

```shell
uv sync --dev
```

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, [ty](https://docs.astral.sh/ty/) for type checking, and [pytest](https://docs.pytest.org/) for testing.

```shell
uvx ruff check src
uvx ty check src
uv run pytest
```

## Features

### Data Representation

PioNIR provides specialized `Spectrum` and `SpectrumCollection` classes to handle NIR spectra. These classes serve as the foundation for handling individual measurements and larger datasets within the library.

```python
from pionir import Spectrum, SpectrumCollection, Metadata

# x, y: NumPy ndarrays
s = Spectrum(x=..., y=..., metadata=...)  # Single spectrum
c = SpectrumCollection([s])  # Collection of spectra
```

### Unified Interface with Single Dispatch

PioNIR uses a single dispatch approach, allowing the same function interface to handle both individual `Spectrum` objects and `SpectrumCollection` instances as well as plain NumPy `ndarray`.

```python
from pionir import Spectrum, SpectrumCollection
from pionir.scatter import snv

s = Spectrum(...)
c = SpectrumCollection(...)

# The same function handles both single spectra and collections
s_snv = snv(s)
c_snv = snv(c)
```

### scikit-learn Integration

The library is designed for compatibility with scikit-learn, providing transformers that can be directly incorporated into pipelines.

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from pionir import SpectrumCollection
from pionir.sklearn.transformer import SNVTransformer

c = SpectrumCollection([...])

pcr_pipeline = Pipeline([
    ("snv", SNVTransformer()),
    ("pca", PCA(n_components=5)),
    ("regression", LinearRegression())
])

pcr_pipeline.fit(c.y, ...)
pcr_pipeline.predict(...)
```
