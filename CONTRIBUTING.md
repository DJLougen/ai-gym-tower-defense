# Contributing

Thanks for considering a contribution! This project aims to be a reliable
sandbox for AI research, so we value correctness and reproducibility above
flashy features.

## Getting set up

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode plus pytest and ruff.

## Before you push

Run the smoke tests and the linter:

```bash
pytest
ruff check .
```

Both should be green. If you change gameplay-affecting code, include:

1. A unit test (or extend `tests/test_smoke.py`).
2. A GIF before/after for visual changes (render with `evaluate.py --gif`).
3. A note in the README if you changed the observation or action space.

## Project conventions

- **Pure-Python engine.** The engine has no dependency on NumPy, PyTorch, or
  pygame. Keep it that way so the game is trivially portable.
- **Deterministic by default.** Seed every RNG. If a function needs randomness,
  it takes a `seed` or an `rng` argument.
- **Observations are float32, actions are int64.** Match the existing spaces.
- **No hidden global state.** `GameConfig` is frozen; everything tunable is a
  field.

## Feature ideas we'd welcome

- More maps / procedural map generation.
- Additional tower or enemy archetypes (keep them in `config.py`).
- Multi-agent / adversarial modes.
- JAX / sb3-compatible wrappers.
- Notebook walkthroughs.

## Reporting bugs

Please include:

- Python + OS versions.
- The smallest script that reproduces the issue.
- The full traceback.

Thank you!
