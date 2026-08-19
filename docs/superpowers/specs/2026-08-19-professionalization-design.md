# word2vec-factory Professionalization — Design Spec

**Date:** 2026-08-19
**Status:** Approved (design review), pending implementation plan

## Goal

Enhance the `word2vec-factory` repository professionally — code and docs — while
preserving its **educational / from-scratch** nature. Balance chosen: focused,
readable code-quality work (tests, lint/format, bug fixes) followed by a
documentation expansion. Not a production framework overhaul.

## Decisions (from brainstorming)

- **Primary goal:** both code and docs, leaning **code quality first**.
- **Rigor level:** **balanced** — focused readable tests on the tricky math + one
  end-to-end smoke test; `ruff` (lint + format, isort) config; fix dead config and
  the `cuda:7` bug; no CI.
- **Docs scope:** README overhaul + a `docs/` mini-site (architecture, config
  reference, ablation guide, evaluation, from-scratch notes).
- **Language:** English for README and `docs/` prose; code docstrings stay
  **Chinese** (current style). Content follows repo context `从零实现与消融`.

---

## Section 1 — Code quality & correctness (balanced)

### Real bug fixes
- `pick_device()` in `w2v_factory/utils.py`: remove hardcoded `cuda:7`.
  `"auto"` → `torch.device("cuda")`; explicit `cpu`/`cuda:N` respected as-is.
- **Dead config keys** (no code reads them) removed from `config.py` dataclasses and
  `configs/base.yaml`:
  - `ModelCfg.hs_use_huffman` (HS is always Huffman in this implementation)
  - `TrainCfg.num_workers` (no DataLoader is used)
  - `RunCfg.save_every_steps` (no checkpointing exists)
  - `EvalCfg.topk` (unused; the eval CLI has its own `--topk`)
- **Centralize subsampling coercion:** `subsample._coerce_t` is the single source of
  truth for coercing `None`/float/int/string `subsample_t`. `config.py` delegates to it
  instead of reimplementing the logic.

### Tooling (no CI, per balanced choice)

- Add `[tool.ruff]` config to `pyproject.toml` (lint + format; isort is built into
  ruff). Optionally add `[tool.black]`. Use plain `ruff check .` / `ruff format .`.
- Add a `Makefile` with `make lint`, `make format`, `make test`.
- Add `scripts/lint.sh`.
- Normalize type hints on unttyped functions (`ablate.py`: `parse_args`,
  `set_by_path`, `main`) and any `no-qa`-free missing-annotation spots that ruff flags.
- Add `pytest` as a dev/optional dependency group in `pyproject.toml`.

### Tests (`tests/`, pytest)

Focused, readable, on the tricky math + one smoke test:

- `test_huffman.py` — inner-node count = V−1; per-word path validity; prefix-code
  uniqueness; determinism.
- `test_subsample.py` — `t` coercion (None/float/int/string variants); discard-prob
  bounds and zeros-when-disabled behavior.
- `test_sampler.py` — AliasSampler approximately matches target unigram^0.75
  distribution; internal `q` sums to 1.
- `test_metrics.py` — analogy accuracy on a small synthetic embeddings set.
- `test_vocab.py` — `min_count` / `max_vocab` / ordering behavior.
- `test_config.py` — YAML `INCLUDE` merging + subsample coercion through `load_cfg`.
- `test_engine_smoke.py` — end-to-end on a tiny corpus: skip-gram+NS and CBOW+HS;
  assert loss is finite and vector files (`embeddings.txt`, `embeddings.npy`) written.

---

## Section 2 — Docs

English README + `docs/` prose in English; code docstrings stay Chinese.

- **`README.md`** (rewrite): overview, features, install, quickstart, train/eval/ablate
  usage, project layout, pointers to `docs/`.
- **`docs/architecture.md`** — factory-registry design, module responsibilities, data
  flow (raw text → vocab → subsampling → sampler → batch → loss → update), HS & NS math.
- **`docs/config-reference.md`** — every config key, default, meaning.
- **`docs/ablation.md`** — how the grid runs and how to read outputs.
- **`docs/evaluation.md`** — Google Analogy / 3CosAdd details.
- **`docs/from-scratch.md`** — algorithm derivations (skipgram/CBOW, Huffman HS,
  negative sampling, subsampling).

---

## Section 3 — Housekeeping

- **`.gitignore`** — `__pycache__/`, `*.pyc`, `runs/`, `.ruff_cache/`, `.pytest_cache/`,
  `*.egg-info/`, etc.
- **Remove committed `__pycache__/`** from git (untrack + prune).
- **`LICENSE`** — MIT.
- **`CONTRIBUTING.md`** — brief contribution guide.

---

## Out of scope (YAGNI)

- No CI/CD, no packaging/distribution changes beyond the existing editable install.
- No new algorithms or features (no checkpointing, no dataloader workers).
- No rewriting `trainer/engine.py` as a new framework — it stays a readable monolith to
  preserve teaching clarity.
- No Chinese docs prose; code docstrings not translated to English.