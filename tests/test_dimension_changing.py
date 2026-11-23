"""DataArrays through the two functions that change the length of a dimension."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _backends import needs_jax

from geophysix import GeophysixError
from geophysix.earth import cell_area, sphere
from geophysix.thermo import hypsometric_thickness


# three levels and three columns, so that a transposed axis is not caught by its length
P = np.asarray([1e5, 8e4, 5e4])
T = np.asarray([[290.0, 295.0, 300.0], [280.0, 270.0, 260.0], [250.0, 240.0, 230.0]])
EXPECTED = hypsometric_thickness(P, T.T)


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


@pytest.fixture
def pressure(xr):
    return xr.DataArray(P, {"level": [1000, 800, 500]}, ("level",), name="p")


@pytest.fixture
def temperature(xr):
    coords = {
        "level": [1000, 800, 500],
        "x": ("x", [1, 2, 3], {"units": "km"}),
        "z": ("level", [0.0, 2e3, 5e3]),
    }
    attrs = {"units": "K"}
    return xr.DataArray(T, coords, ("level", "x"), name="tv", attrs=attrs)


# --- hypsometric_thickness ---


def test_the_order_of_the_dimensions_does_not_matter(pressure, temperature) -> None:
    for t in (temperature, temperature.transpose("x", "level")):
        out = hypsometric_thickness(pressure, t, dim="level")
        assert out.dims == ("x", "level")
        np.testing.assert_array_equal(out.values, EXPECTED)


def test_only_the_coordinates_of_the_levels_go(pressure, temperature) -> None:
    out = hypsometric_thickness(pressure, temperature, dim="level")
    assert set(out.coords) == {"x"}
    assert out.x.attrs == {"units": "km"}
    assert out.name is None
    assert out.attrs == {}


def test_levels_in_another_order_raise(pressure, temperature) -> None:
    flipped = temperature.isel(level=slice(None, None, -1))
    with pytest.raises(ValueError, match="level"):
        hypsometric_thickness(pressure, flipped, dim="level")


def test_thickness_stays_lazy_and_float32(pressure, temperature) -> None:
    da = pytest.importorskip("dask.array")
    chunked = temperature.astype(np.float32).chunk({"x": 1, "level": 2})
    out = hypsometric_thickness(pressure.astype(np.float32), chunked, dim="level")
    assert isinstance(out.data, da.Array)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.values, EXPECTED, rtol=1e-5)


def test_thickness_needs_dim_with_dataarrays_and_only_there(
    pressure, temperature
) -> None:
    with pytest.raises(GeophysixError, match="DataArrays need 'dim'"):
        hypsometric_thickness(pressure, temperature)
    with pytest.raises(GeophysixError, match="plain arrays have none"):
        hypsometric_thickness(P, T.T, dim="level")
    with pytest.raises(GeophysixError, match="'pressure' must be a DataArray"):
        hypsometric_thickness(P, temperature, dim="level")
    with pytest.raises(GeophysixError, match="'pressure' has no dimension 'x'"):
        hypsometric_thickness(pressure, temperature, dim="x")


# --- cell_area ---


@pytest.fixture
def edges(xr):
    lat = xr.DataArray(np.linspace(-90.0, 90.0, 19), dims="lat_edge", name="lat")
    lon = xr.DataArray(np.linspace(0.0, 360.0, 37), dims="lon_edge", name="lon")
    return lat, lon


def test_areas_of_the_sphere(edges) -> None:
    out = cell_area(*edges, dims=("lat", "lon"), earth=sphere(2.0))
    assert out.sizes == {"lat": 18, "lon": 36}
    assert out.name is None
    assert math.isclose(float(out.sum()), 16.0 * math.pi, rel_tol=1e-12)
    plain = cell_area(edges[0].values, edges[1].values, earth=sphere(2.0))
    np.testing.assert_array_equal(out.values, plain)


def test_radians_pass_through(edges) -> None:
    lat, lon = (np.deg2rad(edge) for edge in edges)
    out = cell_area(lat, lon, dims=("lat", "lon"), degrees=False)
    np.testing.assert_allclose(out, cell_area(*edges, dims=("lat", "lon")), rtol=1e-12)


def test_a_result_may_take_the_names_of_the_edges(edges) -> None:
    lat, lon = edges[0].rename(lat_edge="lat"), edges[1].rename(lon_edge="lon")
    lat = lat.assign_coords(lat=lat.values)
    out = cell_area(lat, lon, dims=("lat", "lon"))
    assert out.sizes == {"lat": 18, "lon": 36}
    assert set(out.coords) == set()


@pytest.mark.parametrize("dims", [None, "ab", ("lat", "lat"), ("lat",)])
def test_areas_need_two_names(edges, dims) -> None:
    with pytest.raises(GeophysixError, match="DataArrays need 'dims'"):
        cell_area(*edges, dims=dims)


def test_areas_refuse_what_cannot_be_named(xr, edges) -> None:
    lat, lon = edges
    with pytest.raises(GeophysixError, match="plain arrays have none"):
        cell_area(lat.values, lon.values, dims=("lat", "lon"))
    with pytest.raises(GeophysixError, match="'lat_edges' must be a DataArray"):
        cell_area(lat.values, lon, dims=("lat", "lon"))
    with pytest.raises(GeophysixError, match="share the dimension 'edge'"):
        cell_area(
            lat.rename(lat_edge="edge"),
            lon.rename(lon_edge="edge"),
            dims=("lat", "lon"),
        )
    square = xr.DataArray(np.ones((2, 2)), dims=("a", "b"))
    with pytest.raises(GeophysixError, match="'lat_edges' must have one dimension"):
        cell_area(square, lon, dims=("lat", "lon"))


# --- plain arrays ---


@needs_jax
def test_plain_jax_arrays_still_transform() -> None:
    import jax
    import jax.numpy as jnp

    p, t = jnp.asarray(P), jnp.asarray(T.T)
    out = jax.jit(hypsometric_thickness)(p, t)
    np.testing.assert_allclose(out, EXPECTED, rtol=1e-14)
    area = jax.jit(cell_area)(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0]))
    assert area.shape == (1, 1)
