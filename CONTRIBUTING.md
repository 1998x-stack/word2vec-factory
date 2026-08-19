# Contributing

Thanks for your interest in `word2vec-factory`!

This is an educational, from-scratch implementation. We value clarity and
reproducibility over framework features.

## Getting started

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -e .
pip install pytest ruff black
```

## Development commands

```bash
make lint     # ruff check + black --check
make format   # black formatting
make test     # pytest
```

## Pull request checklist

- Code passes `make lint`.
- All tests pass (`make test`).
- New behavior ships with tests in `tests/`.
- Docs stay in English; code *docstrings* stay Chinese to match the codebase.
- No unrelated refactoring (especially in `trainer/engine.py` — it is a
  deliberately readable monolith).