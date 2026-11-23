---
name: add-physical-function
description: Add a physical function to a public module of the package — where it goes in _src/<module>/, its record argument with a default, a Google-style docstring, the oracle test, a jit / vmap / grad smoke test that skips without JAX, the parity call, the API page and the capability index. Use when asked to add, port or implement a constant, a formula, a conversion or a new module in src/geophysix.
---

# Add a physical function

Read "1. Public functions" and "The public API" under "The contracts" in `AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` for the quantity and its synonyms, and the "Reuse before you write" table in `AGENTS.md`.
- A combination of existing names needs no new function.

## 2. Pick its home

Private code goes in `src/geophysix/_src/<module>/<topic>.py`; the public module (`constants`, `earth`, `rotation`, `thermo`, `wind`, `seawater`, `columns`) re-exports it. The ownership test decides the module: a closed-form fact at a point belongs here, anything on a grid or a field belongs above (gridax, diagnostix). A new public module is created by its first feature issue. A new exception goes in `exceptions.py`, derives from `GeophysixError` and joins `tests/test_exceptions.py`.

## 3. Write it

- Open with `xp, (x, y) = floating_inputs(x, y, params=(...))` from `pipekit_array.functions`, and call only functions of the array API standard on `xp`. Never `import jax` at module level.
- Decorate an elementwise function with `@elementwise()` from `pipekit_array.functions`, so that a DataArray in gives a DataArray out.
- Constants enter in the input's dtype: `c = xp.asarray(C, dtype=x.dtype)`; float32 in gives float32 out.
- Parameters that belong together are a frozen dataclass record passed as a keyword argument with a default (`*, ellipsoid: Ellipsoid = WGS84`).
- SI units. Degrees versus radians is an explicit argument (`degrees: bool = False`), never guessed.
- The poles and `e = 0` return the limit; a branch uses the double `where` so its gradient is finite.
- Comments above the line with the Unicode equation above each mathematical line; descriptive names; one operation per line.
- Google-style docstring in the `python-style` layout: a one-line summary (it becomes the capability-index entry), the defining equation in Unicode (never `$…$`), `Args:` with meaning, symbol and unit, `Returns:`, `Raises:` and an `Examples:` block of rounded values.

## 4. Export and document

- Add the name to the public module's `__all__` (sorted); a new module joins `SUBMODULES` in `tests/test_public_api.py`.
- API page `docs/api/<module>.md` with a `::: geophysix.<module>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/test_<module>.py`, one class per function:

- **Oracle.** Against pyproj, gsw or a CODATA value, through the helpers in `tests/_oracles.py`, with the tolerance and its source in a comment; skips when the oracle package is missing.
- **Property.** A `hypothesis` property (symmetry, monotonicity, a round trip, a limit).
- **Transforms smoke test.** `jit`, `vmap` and `grad` run and give finite results; skips without JAX (`needs_jax` in `tests/_backends.py`).
- **Parity.** A `Call` in `tests/_calls.py`, with its tolerances: `tests/test_array_kinds.py` runs it on every array kind and fails when a public function has none. A loose tolerance says why in a comment.
- The poles, `e = 0` and each invalid argument (`RangeError`, `UnitError` with `match=`).

## 6. Verify

```bash
uv run pytest --no-cov tests/test_<module>.py src/geophysix/_src/<module> tests/test_public_api.py tests/test_capabilities.py -v
uv run poe capabilities
uv run poe test-cov
```

then the `pre-pr-check` skill.
