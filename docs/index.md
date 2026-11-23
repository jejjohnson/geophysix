# geophysix

> Physical constants, Earth geometry and atmospheric state, on any array library.

geophysix is the GeoStack's library of closed-form physical facts about the Earth, its air and its sea water: constants, the ellipsoid and the sphere, rotation, thermodynamics, the wind, the sea-water equation of state and column units. It is GeoNumerics tier 0 and imports nothing from the rest of the stack.

The formulas are written in the [Python array API standard](https://data-apis.org/array-api/latest/): NumPy in gives NumPy out, and JAX, torch, CuPy or dask arrays give the same back, on the same device. JAX is never imported at module level, so the same functions run under `jit`, `vmap` and `grad` when you pass JAX arrays and need nothing else installed when you do not.

## Installation

::::{tab-set}
:::{tab-item} uv

```bash
uv add geophysix
```
:::
:::{tab-item} pip

```bash
pip install geophysix
```
:::
:::{tab-item} From source

```bash
git clone https://github.com/jejjohnson/geophysix.git
cd geophysix
uv run poe install
```
:::
::::

## Status

Wave 0: the package skeleton and the exception hierarchy ([`GeophysixError`](xref:api#geophysix.GeophysixError)). The public modules are planned; see the [API reference](xref:api#geophysix) for the list.

## Design rules

1. **A closed-form fact at a point.** Anything on a grid or a field belongs above, in gridax or diagnostix.
2. **SI units everywhere.** Degrees versus radians is an explicit argument.
3. **Constants enter in the input's dtype**, so float32 stays float32.
4. **Fail with a typed exception.** Everything raised derives from [`GeophysixError`](xref:api#geophysix.GeophysixError), which is a `ValueError`.
