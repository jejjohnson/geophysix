---
name: geophysix-review
description: Review a change or pull request in this repository against its own rules, CODE_REVIEW.md, the contracts in AGENTS.md, reuse of what the package already provides, and units and conventions (SI units, degrees versus radians, dtype promotion, poles and e = 0 limits, gradients). Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note which modules, tests and docs it touches.
2. **Reuse**: run the `reuse-reviewer` subagent on the diff: a hand-written Earth radius, gravity, haversine, Coriolis parameter or ideal-gas relation is the main way a package drifts from itself.
3. **Units and conventions**: run the `correctness-reviewer` subagent in parallel with step 2: SI units, degrees versus radians, dtype promotion by constants, poles and `e = 0` limits, double-`where` gradients, doctest accuracy, test tolerances.
4. **Contracts**: for each module touched, the rules under "The contracts" in `AGENTS.md`: the function opens with `floating_inputs`, calls only the array API standard, and JAX is never imported at module level; constants enter through `like`, in the input's dtype and on its device; records are frozen dataclasses passed as keyword arguments with defaults; private code sits in `_src/<module>/` and the public module only re-exports; the layer rule holds (no import of gridax, diagnostix, pipekit, geotoolz or a model); exceptions derive from `GeophysixError`; the function has its oracle, property, `jit` / `vmap` / `grad` tests, and its `Call` in `tests/_calls.py`.
5. **Checklist**: the rest of `CODE_REVIEW.md` (public API, API page and `docs/api/index.md` row, capability index, MyST prose, tests and tiers), skipping what ruff, ty or the tests already enforce.
6. **House style**: the `python-style` review checklist if the skill is installed, otherwise "House style" in `AGENTS.md`: docstring layout with a Unicode equation, comments above the line, descriptive names, maths in the `eman-style-math` notation, `cyclopts` and `loguru` in scripts.
7. **Verify claims**: for anything you flag as a bug, run it (`uv run python -c "..."`) and report what it printed.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with priority, file, lines and a concrete change, summary).
