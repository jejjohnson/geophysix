# Changelog

## [0.0.10](https://github.com/jejjohnson/geophysix/compare/v0.0.9...v0.0.10) (2026-11-26)


### Bug Fixes

* **deps:** pipekit at pipekit-array-v0.3.0, the release with the areas of every package ([#81](https://github.com/jejjohnson/geophysix/issues/81)) ([da48828](https://github.com/jejjohnson/geophysix/commit/da4882833548bc6979ab557ab5947bf2f2190bd4))

## [0.0.9](https://github.com/jejjohnson/geophysix/compare/v0.0.8...v0.0.9) (2026-11-25)


### Bug Fixes

* **deps:** pipekit-array 0.2.0 and its areas ([#78](https://github.com/jejjohnson/geophysix/issues/78)) ([c63e60d](https://github.com/jejjohnson/geophysix/commit/c63e60d53fe0e7b2420f7240d344d563124b7605))

## [0.0.8](https://github.com/jejjohnson/geophysix/compare/v0.0.7...v0.0.8) (2026-11-25)


### Bug Fixes

* **deps:** pipekit-array 0.1.5, the tag that diagnostix pins ([#73](https://github.com/jejjohnson/geophysix/issues/73)) ([44883ca](https://github.com/jejjohnson/geophysix/commit/44883cad5be33876d2e4f81c03c563a8156fabf5))
* **deps:** pipekit-array 0.1.6, the newest tag of pipekit ([#76](https://github.com/jejjohnson/geophysix/issues/76)) ([5fcc282](https://github.com/jejjohnson/geophysix/commit/5fcc282ea18f70101b2cf7c6aaec2b17fb44981a))

## [0.0.7](https://github.com/jejjohnson/geophysix/compare/v0.0.6...v0.0.7) (2026-11-24)


### Bug Fixes

* **deps:** pipekit-array 0.1.4, the tag that gridax and geotoolz pin ([#70](https://github.com/jejjohnson/geophysix/issues/70)) ([350b461](https://github.com/jejjohnson/geophysix/commit/350b46175169845c0e6bb4e2f4c510b06e7ed2b3))

## [0.0.6](https://github.com/jejjohnson/geophysix/compare/v0.0.5...v0.0.6) (2026-11-23)


### Bug Fixes

* **deps:** pipekit-array 0.1.3, one tag for the stack ([#69](https://github.com/jejjohnson/geophysix/issues/69)) ([3f9f670](https://github.com/jejjohnson/geophysix/commit/3f9f67082b409b43fbdec32d09f4e2eae5853245))
* **xarray:** a DataArray's coordinates keep their attributes ([02e96b0](https://github.com/jejjohnson/geophysix/commit/02e96b0b55fb742bbd2adbadb643a10cc910c36c)), closes [#66](https://github.com/jejjohnson/geophysix/issues/66)

## [0.0.5](https://github.com/jejjohnson/geophysix/compare/v0.0.4...v0.0.5) (2026-11-23)


### Features

* **backend:** a DataArray in gives a DataArray out, aligned by dimension name ([#62](https://github.com/jejjohnson/geophysix/issues/62)) ([3d28b92](https://github.com/jejjohnson/geophysix/commit/3d28b92b59ebc2027b7aca67926ac37ed1fdf3ef)), closes [#58](https://github.com/jejjohnson/geophysix/issues/58)
* **backend:** an array keeps its library, device and gradient, and an unknown one raises ([#60](https://github.com/jejjohnson/geophysix/issues/60)) ([2ea1d05](https://github.com/jejjohnson/geophysix/commit/2ea1d05f4a9617d241c73f77b3f293f0e9857f31)), closes [#56](https://github.com/jejjohnson/geophysix/issues/56)
* **earth,thermo:** cell_area and hypsometric_thickness take and return DataArrays ([#63](https://github.com/jejjohnson/geophysix/issues/63)) ([97a1439](https://github.com/jejjohnson/geophysix/commit/97a14397d239b15ca8d481473120cddab67602ec)), closes [#59](https://github.com/jejjohnson/geophysix/issues/59)

## [0.0.4](https://github.com/jejjohnson/geophysix/compare/v0.0.3...v0.0.4) (2026-11-21)


### Features

* **constants:** the ISA sea-level temperature, standard_temperature = 288.15 K ([#47](https://github.com/jejjohnson/geophysix/issues/47)) ([74dc7a4](https://github.com/jejjohnson/geophysix/commit/74dc7a45545db1b0b33b43fc8960258543745d33)), closes [#42](https://github.com/jejjohnson/geophysix/issues/42)
* **earth:** the radius of a parallel and the length of a degree of latitude and longitude ([#48](https://github.com/jejjohnson/geophysix/issues/48)) ([350b4a4](https://github.com/jejjohnson/geophysix/commit/350b4a44f7fca887596974ebdd7e6c8ad5047f4d)), closes [#43](https://github.com/jejjohnson/geophysix/issues/43)
* **thermo:** temperature from potential temperature, and vertical velocity from pressure velocity ([#49](https://github.com/jejjohnson/geophysix/issues/49)) ([94bf770](https://github.com/jejjohnson/geophysix/commit/94bf770cb1dc1b6146255ff77dbd456f056590c4)), closes [#44](https://github.com/jejjohnson/geophysix/issues/44) [#45](https://github.com/jejjohnson/geophysix/issues/45)
* **wind:** wind speed, direction and components, in a new module ([#50](https://github.com/jejjohnson/geophysix/issues/50)) ([2a57ad3](https://github.com/jejjohnson/geophysix/commit/2a57ad3a249ccbf6e6bcdfc2e0d3ed1a9f0dd36e)), closes [#46](https://github.com/jejjohnson/geophysix/issues/46)

## [0.0.3](https://github.com/jejjohnson/geophysix/compare/v0.0.2...v0.0.3) (2026-11-20)


### Bug Fixes

* **constants:** importing geophysix never imports JAX; records register when JAX loads ([#39](https://github.com/jejjohnson/geophysix/issues/39)) ([af11e49](https://github.com/jejjohnson/geophysix/commit/af11e4931e8d43f2fe883571fe96b423dbdb428c))

## [0.0.2](https://github.com/jejjohnson/geophysix/compare/v0.0.1...v0.0.2) (2026-11-20)


### Features

* **constants:** the Constants, Gas and Ellipsoid records, with STANDARD, WGS84, GRS80 and SPHERE ([6342235](https://github.com/jejjohnson/geophysix/commit/6342235303d1b5dbd473959ce78ddf1bcdb3beb9)), closes [#15](https://github.com/jejjohnson/geophysix/issues/15)
* **earth:** geodetic to Earth-centred coordinates and back, and the radii of curvature ([cf53f51](https://github.com/jejjohnson/geophysix/commit/cf53f518e9e4e54525e0bb4e142affb4af6ee2f6)), closes [#17](https://github.com/jejjohnson/geophysix/issues/17)
* **earth:** great-circle distance and initial azimuth on a sphere of given radius ([96e5077](https://github.com/jejjohnson/geophysix/commit/96e50777af59b4921d654bc4ea2f7afcad4b9d3d))
* **earth:** great-circle distance and initial azimuth on a sphere of given radius ([95fb810](https://github.com/jejjohnson/geophysix/commit/95fb8109702956a65f740dddabcb514ff6bd6b4a)), closes [#19](https://github.com/jejjohnson/geophysix/issues/19)
* **earth:** the area of a latitude-longitude cell on the ellipsoid and on the sphere ([98e7bbc](https://github.com/jejjohnson/geophysix/commit/98e7bbc4d5f7dd7fcb535f68691428504e953c55)), closes [#18](https://github.com/jejjohnson/geophysix/issues/18)
* **rotation, thermo, columns:** the Coriolis parameter, number density, column units and the dry-air column ([#35](https://github.com/jejjohnson/geophysix/issues/35)) ([3ec06fc](https://github.com/jejjohnson/geophysix/commit/3ec06fcc365c92fb6bcb0edd864f33f5914c1214)), closes [#28](https://github.com/jejjohnson/geophysix/issues/28) [#29](https://github.com/jejjohnson/geophysix/issues/29) [#30](https://github.com/jejjohnson/geophysix/issues/30) [#31](https://github.com/jejjohnson/geophysix/issues/31)
* **seawater, thermo:** the linear equation of state, buoyancy, reduced gravity, and dry-air thermodynamics ([cfb7aa2](https://github.com/jejjohnson/geophysix/commit/cfb7aa245ae1aa9b8a0bd30746c96d989e0d2cbc))


### Bug Fixes

* **constants:** the record helper imports JAX by name, so ty resolves it without JAX installed ([dc287e5](https://github.com/jejjohnson/geophysix/commit/dc287e5c08731e9539313d304ecb027dc6d4ab06))

## Changelog
