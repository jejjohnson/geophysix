# API Reference

The public API is the set of public modules below, plus the exceptions that `geophysix.__all__` exports. Private code lives in `geophysix._src`. The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Module | Contents | Status |
|---|---|---|
| [`geophysix.constants`](constants.md) | CODATA 2018 constants, standard gravity, dry air and trace gases | available |
| [`geophysix.earth`](earth.md) | The ellipsoid and the sphere: records, coordinates, radii, cell areas, distances | available |
| [`geophysix.rotation`](rotation.md) | The Coriolis parameter, β and the β-plane | available |
| [`geophysix.thermo`](thermo.md) | The ideal gas, virtual and potential temperature, the Exner function, hypsometric thickness | available |
| [`geophysix.seawater`](seawater.md) | The linear equation of state, buoyancy and reduced gravity | available |
| [`geophysix.columns`](columns.md) | Column units, the dry-air column and mixing ratios | available |
| [Type aliases](typing.md) | `ArrayLike` | available |
| [Exceptions](exceptions.md) | `GeophysixError`, `EarthError`, `UnitError`, `RangeError` | available |

## Package overview

::: geophysix
    options:
      members: false
      show_root_heading: false
      show_source: false
