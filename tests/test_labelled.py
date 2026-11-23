"""DataArrays through geophysix: labels kept, aligned by name, never guessed."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from _backends import needs_jax

from geophysix import BackendError, GeophysixError
from geophysix._src.labelled import elementwise, is_dataarray
from geophysix.columns import convert
from geophysix.constants import CH4
from geophysix.earth import ecef_to_geodetic, geodetic_to_ecef, sphere
from geophysix.rotation import beta_plane, coriolis
from geophysix.thermo import number_density
from geophysix.wind import wind_components, wind_speed


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


@pytest.fixture
def temperature(xr):
    return xr.DataArray(
        np.asarray([[280.0, 290.0], [250.0, 260.0]], dtype=np.float32),
        dims=("level", "x"),
        coords={
            "level": [1000, 800],
            "x": [1, 2],
            "height": ("level", [100.0, 2000.0]),
        },
        name="t",
        attrs={"units": "K"},
    )


@pytest.fixture
def pressure(xr):
    values = np.asarray([1e5, 8e4], dtype=np.float32)
    return xr.DataArray(values, {"level": [1000, 800]}, ("level",), name="p")


def test_plain_arrays_pass_through() -> None:
    out = coriolis(np.asarray([0.0, 30.0]))
    assert type(out) is np.ndarray
    assert not is_dataarray(out)


def test_a_square_grid_aligns_by_name(pressure, temperature) -> None:
    out = number_density(pressure, temperature)
    expected = number_density(pressure.values[:, None], temperature.values)
    assert out.dims == ("level", "x")
    np.testing.assert_array_equal(out.values, expected)
    # by position, as geophysix broadcast before, the answer is 25 % off
    by_position = number_density(pressure.values, temperature.values)
    assert np.max(np.abs(by_position / expected - 1.0)) > 0.2


def test_coordinates_stay_and_the_name_and_attributes_go(pressure, temperature) -> None:
    out = number_density(pressure, temperature)
    assert set(out.coords) == {"level", "x", "height"}
    assert out.name is None
    assert out.attrs == {}
    assert out.dtype == np.float32


def test_dimensions_broadcast(xr) -> None:
    lat = xr.DataArray([10.0, 20.0], dims="lat")
    lon = xr.DataArray([0.0, 5.0, 10.0], dims="lon")
    x, _, z = geodetic_to_ecef(lon, lat)
    assert x.sizes == {"lon": 3, "lat": 2}
    # z does not depend on the longitude
    assert z.sizes == {"lon": 3, "lat": 2}


def test_every_output_has_the_dimensions_of_all_the_inputs(xr) -> None:
    x = xr.DataArray([6.4e6, 6.5e6], dims="x")
    z = xr.DataArray([0.0, 1e5, 2e5], dims="z")
    for out in ecef_to_geodetic(x, 0.0, z):
        assert out.sizes == {"x": 2, "z": 3}
    # the conversion to mass does not read the pressure
    out = convert(x, "mol_m2", "kg_m2", gas=CH4, pressure=z)
    assert out.sizes == {"x": 2, "z": 3}
    np.testing.assert_array_equal(out.isel(z=0), x * CH4.molar_mass)


def test_mismatched_coordinates_raise(pressure, temperature) -> None:
    with pytest.raises(ValueError, match="level"):
        number_density(pressure.assign_coords(level=[1000, 850]), temperature)


@pytest.mark.parametrize("plain", [np.ones(2), [1e5, 8e4], (1e5, 8e4)])
def test_an_array_without_names_is_not_guessed(temperature, plain) -> None:
    with pytest.raises(GeophysixError, match="'pressure' has dimensions but no names"):
        number_density(plain, temperature)


def test_keywords_align_and_options_pass_through(pressure, temperature) -> None:
    column = temperature * 10.0
    out = convert(column, "ppm_m", "mol_m2", pressure=pressure, temperature=temperature)
    expected = convert(
        column.values,
        "ppm_m",
        "mol_m2",
        pressure=pressure.values[:, None],
        temperature=temperature.values,
    )
    np.testing.assert_array_equal(out.values, expected)


def test_parameters_may_be_labelled(xr) -> None:
    y = xr.DataArray([0.0, 1e5, 2e5], dims="y")
    f0 = xr.DataArray([1e-4, 1.2e-4], dims="case")
    out = beta_plane(y, f0=f0, beta=2e-11, y0=0.0)
    assert set(out.dims) == {"y", "case"}
    np.testing.assert_allclose(out.sel(case=0), [1e-4, 1.02e-4, 1.04e-4], rtol=1e-12)


def test_records_numbers_and_flags_pass_through(xr) -> None:
    lat = xr.DataArray([0.0, np.pi / 2], dims="lat")
    out = coriolis(lat, earth=sphere(6.4e6, rotation_rate=1e-4), degrees=False)
    np.testing.assert_allclose(out.values, [0.0, 2e-4], rtol=1e-15)
    assert wind_speed(lat, np.float64(1.0)).dims == ("lat",)


def test_several_outputs_are_a_tuple_of_dataarrays(xr) -> None:
    speed = xr.DataArray([5.0, 2.0], {"x": [10, 20]}, ("x",), name="speed")
    u, v = wind_components(speed, 270.0)
    assert is_dataarray(u) and is_dataarray(v)
    assert u.x.values.tolist() == [10, 20]
    assert u.name is None and v.name is None
    lon, lat, height = ecef_to_geodetic(*geodetic_to_ecef(speed, speed, 10.0 * speed))
    np.testing.assert_allclose(lon, speed, rtol=1e-12)
    np.testing.assert_allclose(lat, speed, rtol=1e-12)
    np.testing.assert_allclose(height, 10.0 * speed, atol=1e-8)


def test_dask_stays_lazy(xr, pressure, temperature) -> None:
    da = pytest.importorskip("dask.array")
    out = number_density(pressure, temperature.chunk({"x": 1}))
    assert isinstance(out.data, da.Array)
    np.testing.assert_array_equal(out, number_density(pressure, temperature))


@needs_jax
def test_a_dataarray_of_jax_holds_jax(xr) -> None:
    import jax
    import jax.numpy as jnp

    out = coriolis(xr.DataArray(jnp.arange(3.0), dims="lat"))
    assert isinstance(out.data, jax.Array)


@needs_jax
def test_transforms_go_through_the_decorator() -> None:
    import jax
    import jax.numpy as jnp

    p, t = jnp.asarray([1e5, 8e4]), jnp.asarray([280.0, 250.0])
    eager = number_density(p, t)
    np.testing.assert_allclose(jax.jit(number_density)(p, t), eager, rtol=1e-14)
    np.testing.assert_allclose(jax.vmap(number_density)(p, t), eager, rtol=1e-14)
    slope = jax.grad(number_density)(1e5, 280.0)
    np.testing.assert_allclose(slope, eager[0] / 1e5, rtol=1e-14)


def test_a_dataset_is_not_an_array(xr) -> None:
    with pytest.raises(BackendError, match="Dataset"):
        coriolis(xr.Dataset({"lat": ("x", [1.0])}))


def test_a_dataarray_in_a_record_is_refused(xr) -> None:
    rate = xr.DataArray([1e-4], dims="x")
    with pytest.raises(BackendError, match="DataArray"):
        coriolis(1.0, earth=sphere(6.4e6, rotation_rate=rate))


def test_the_decorator_keeps_the_signature_and_docstring() -> None:
    @elementwise()
    def affine(values, slope):
        """Slope · values."""
        return slope * values

    assert affine.__name__ == "affine"
    assert affine.__doc__ == "Slope · values."
    assert list(inspect.signature(affine).parameters) == ["values", "slope"]


def test_a_missing_argument_is_reported_as_python_does(xr) -> None:
    with pytest.raises(TypeError, match="temperature"):
        number_density(1e5)
    with pytest.raises(TypeError, match="temperature"):
        number_density(xr.DataArray([1e5], dims="x"))
