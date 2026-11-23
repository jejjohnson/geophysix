# geophysix

[![Tests](https://github.com/jejjohnson/geophysix/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geophysix/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/geophysix/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/geophysix/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/geophysix/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/geophysix/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/geophysix/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/geophysix/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/geophysix/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/geophysix)
[![PyPI version](https://img.shields.io/pypi/v/geophysix.svg)](https://pypi.org/project/geophysix/)
[![Python versions](https://img.shields.io/pypi/pyversions/geophysix.svg)](https://pypi.org/project/geophysix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)


Physical constants, Earth geometry and atmospheric state, on any array library.

geophysix is the GeoStack's library of closed-form physical facts about the Earth, its air and its sea water: constants, the ellipsoid and the sphere, rotation, thermodynamics, the wind, the sea-water equation of state and column units. It is GeoNumerics tier 0 and imports nothing from the rest of the stack.

The formulas are written in the [Python array API standard](https://data-apis.org/array-api/latest/): NumPy in gives NumPy out, and JAX, torch, CuPy or dask arrays give the same back, on the same device. An xarray DataArray gives a DataArray back, aligned by dimension name. No array library but NumPy is imported unless you pass its arrays.

## Install

```bash
pip install geophysix
```

```bash
uv add geophysix
```

The `jax` and `xarray` extras add those libraries: `pip install "geophysix[jax,xarray]"`.

## Status

Wave 0: the package skeleton and its exception hierarchy. The public modules `constants`, `earth`, `rotation`, `thermo`, `seawater` and `columns` are planned and built by the feature issues.

## Design

The design is in [geostack](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/geophysix.md).
