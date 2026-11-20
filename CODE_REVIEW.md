# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (a constant that promotes float32, degrees read as radians, a pole that gives `nan`, a docstring that promises what the code does not do) or a re-implementation of something the package already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import order, `__all__` ordering, doctest output, the coverage gate); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented Earth radius, gravity, haversine, Coriolis or ideal-gas formula is a **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`); the library never imports gridax, diagnostix, pipekit, geotoolz or any model.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

- **Array namespace.** A function computes in the namespace of its first array argument (`geophysix._src.backend.array_namespace`); NumPy in gives NumPy out; JAX is never imported at module level.
- **Constants** enter in the input's dtype (`xp.asarray(c, dtype=x.dtype)`); float32 in gives float32 out.
- **Units.** SI everywhere; degrees versus radians by an explicit argument, never by the magnitude of the input.
- **Records** are frozen dataclasses of floats, passed as keyword arguments with defaults.
- **Limits.** The poles and `e = 0` return the correct limit, not `nan`; a gradient through a branch uses the double `where`.
- **Pure.** Return a new array and never mutate the input.
- **Exceptions** derive from `GeophysixError` (a `ValueError`): `EarthError`, `UnitError`, `RangeError`; errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: in the module's `__all__` and the top-level `__all__`; a new module in `SUBMODULES` (`tests/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (rounded values, not 17-digit floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the top-level name (`xref:api#geophysix.<Name>`); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against an oracle (pyproj, gsw, CODATA or a hand computation), a `hypothesis` property, a `jit` / `vmap` / `grad` smoke test that skips without JAX, and a NumPy / JAX parity test; the poles, `e = 0` and the float32 dtype have their own tests.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the statistic's own sampling distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `dataclasses` with `slots=True` for data containers (`frozen=True` when immutable); `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line.
- New CLIs and scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging`.
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

Each example below runs on NumPy alone.

### Constants take the input's dtype

```python
import numpy as np

OMEGA = 7.292115e-5


# ❌ A float64 constant promotes a float32 input to float64 under NumPy's
#    value-based rules for arrays, and it breaks float32 under JAX
def coriolis_bad(lat_rad):
    return 2.0 * np.asarray(OMEGA, dtype=np.float64) * np.sin(lat_rad)


# ✅ The constant enters in the input's dtype
def coriolis(lat_rad):
    omega = np.asarray(OMEGA, dtype=lat_rad.dtype)
    return 2.0 * omega * np.sin(lat_rad)


lat = np.asarray([0.0, 0.5], dtype=np.float32)
assert coriolis_bad(lat).dtype == np.float64
assert coriolis(lat).dtype == np.float32
```

### Degrees or radians is an argument, never a guess

```python
import numpy as np


# ❌ Guesses the unit from the magnitude: 1.5 degrees is read as radians
def sin_lat_bad(lat):
    return np.sin(np.deg2rad(lat) if np.max(np.abs(lat)) > np.pi else lat)


# ✅ The caller says which, and the default is documented
def sin_lat(lat, *, degrees: bool = False):
    return np.sin(np.deg2rad(lat) if degrees else lat)


assert np.isclose(sin_lat(np.asarray(30.0), degrees=True), 0.5)
```

### The limits are values, not `nan`

```python
import numpy as np


# ❌ The sphere (e = 0) divides by zero, and the poles give nan
def n_bad(a, e, lat_rad):
    return a / np.sqrt(1.0 - e**2 * np.sin(lat_rad) ** 2) * np.cos(lat_rad) / np.cos(lat_rad)


# ✅ The expression is finite at e = 0 and at the poles
def prime_vertical_radius(a, e, lat_rad):
    return a / np.sqrt(1.0 - e**2 * np.sin(lat_rad) ** 2)


assert prime_vertical_radius(6371.0, 0.0, np.pi / 2) == 6371.0
```

### Gradients through a branch use the double `where`

```python
import numpy as np


# ✅ The inner where keeps the unused branch finite, so its gradient is too;
#    a single outer where would still differentiate through the 0 / 0
def sinc(x):
    small = np.abs(x) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0, np.sin(safe) / safe)


assert sinc(np.asarray(0.0)) == 1.0
```

### Doctest output the code really prints

```python
# ❌ Expected output copied from a calculator: the doctest fails on the
#    17th digit, or passes only on one platform
#     >>> coriolis(np.asarray(0.5))
#     4.4e-05

# ✅ Round in the example, and paste what it prints
#     >>> round(float(coriolis(np.asarray(0.5))), 8)
#     6.992e-05
import numpy as np

lat = np.asarray(0.5)
assert round(float(2.0 * 7.292115e-5 * np.sin(lat)), 8) == 6.992e-05
```


---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
