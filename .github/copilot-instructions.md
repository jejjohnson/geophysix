# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the single source of truth for every coding agent working here (the module map, "reuse before you write", the contracts, the tests that enforce them, commands, the pre-commit checklist, the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/geophysix/`: the GeoStack's library of closed-form physical facts about the Earth, its air and its sea water. It is GeoNumerics tier 0 and imports nothing from the GeoStack but `pipekit_array.functions` (never gridax, diagnostix, geotoolz or a model). Planned public modules: `constants`, `earth`, `rotation`, `thermo`, `seawater`, `columns`; private code in `_src/<module>/`. Search [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a helper.
- Ownership test: a closed-form fact at a point belongs here; anything on a grid or a field belongs above.
- Keep the contracts in `AGENTS.md`:
  - **array rules**: NumPy in gives NumPy out; JAX is never imported at module level; a function opens with `floating_inputs` (`pipekit_array.functions`) and calls only functions of the array API standard; an unknown array raises `BackendError`; constants enter through `like`, in the input's dtype and on its device, never promoting float32;
  - **records** are frozen dataclasses of floats passed as keyword arguments with defaults;
  - **units** are SI, and degrees versus radians is an explicit argument;
  - every exception derives from `GeophysixError`, which is a `ValueError`, except those of the array backend (`AGENTS.md`, "Exceptions").

The rules for every project:

- Every docstring `Examples:` block runs (`--doctest-modules`): the expected output must be what the code prints.
- Before committing, from the repo root: `uv run poe test`, `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `uv run poe typecheck`; `uv run poe capabilities` after a public API change. Lint the whole repo, never a subdirectory.
- When you suggest a change, propose the code; when you fix a bug, add the test that reproduces it. Don't nitpick what ruff and the formatter handle.
- House style ("House style" in `AGENTS.md`): the maintainer's `python-style` for Python (comments above the line, a Unicode equation above each mathematical line, descriptive names), `eman-style-math` for maths (Unicode in code and docstrings, macros in MyST and MkDocs pages), `cyclopts` and `loguru` for new CLIs and scripts, Markdown never hard-wrapped, Mermaid rather than ASCII diagrams.
- Partial test runs need `--no-cov` (the 90 % coverage gate fails a subset).
- Never push or merge to `main` unless told to; never replace an existing PR description, only append to it.
- No AI attribution: no `Co-Authored-By:` trailer for an AI and no "Generated with …" line, unless asked. PR descriptions open with why, then a small before/after table; one `Closes #N` line per issue.
- Step-by-step recipes (add a physical function, guide page or notebook; pre-PR check; review) are plain Markdown in `.claude/skills/<name>/SKILL.md`, and the two review checklists in `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows [`CODE_REVIEW.md`](../CODE_REVIEW.md).
