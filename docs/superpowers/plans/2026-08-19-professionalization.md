# word2vec-factory Professionalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Professionally enhance the `word2vec-factory` repo — readable tests, lint/format tooling, correctness fixes, housekeeping, and a documentation mini-site — while preserving its from-scratch, educational nature.

**Architecture:** Minimal-risk changes layered in this order: (1) environment + tooling, (2) small source fixes (device selection, dead config, coercion centralization, type hints), (3) a focused regression test suite for the tricky math plus one end-to-end smoke test, (4) housekeeping files, (5) documentation. No new algorithms or features, no CI, no framework rewrite.

**Tech Stack:** Python 3.10+ / 3.12, PyTorch (CPU is enough for the smoke test), NumPy, PyYAML, Loguru, pytest, ruff, black.

## Global Constraints

- **Language rule:** README and `docs/` prose in English. Code docstrings/comments stay in Chinese (current style). Do not translate existing docstrings to English in this plan.
- **Preserve teaching clarity:** do not restructure `trainer/engine.py` into a new framework; it remains a readable monolith.
- **No CI, no packaging/distribution changes.** Keep the editable install as-is.
- **No new features:** no checkpointing, no DataLoader workers, no new algorithms.
- **Formatter standard:** `line-length = 120` for both black and ruff.
- **Config schema cleanup:** remove dead keys `ModelCfg.hs_use_huffman`, `TrainCfg.num_workers`, `RunCfg.save_every_steps`, `EvalCfg.topk` from both `config.py` dataclasses and `configs/base.yaml`.
- **Forward-compat:** dataclass loading already strips unknown keys (`_filter_kwargs`) and YAML has `INCLUDE` merging — removing keys from dataclasses must be accompanied by removing them from `configs/base.yaml` so docs match reality.
- All source/test code must pass `ruff check .` (empty) before each task's final commit.

---

### Task 1: Environment & dependencies

**Files:**
- Modify: `requirements.txt` (unchanged — verify)
- Execute: create `.venv` and install everything

**Interfaces:**
- Consumes: nothing.
- Produces: a working Python environment with all runtime deps + `pytest`, `ruff`, `black`. Every later task runs commands in this environment.

- [ ] **Step 1: Create virtualenv and install runtime deps**

```bash
cd /Users/x/Desktop/1998x-stack/00-仓库/04-深度学习与CV/从零实现与消融/word2vec-factory
python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

- [ ] **Step 2: Install dev tooling**

```bash
. .venv/bin/activate
pip install pytest ruff black
```

- [ ] **Step 3: Verify imports work**

```bash
. .venv/bin/activate
python -c "import torch, numpy, yaml, loguru, w2v_factory; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore 2>/dev/null || true
git commit -am "chore: add .venv ignore" 2>/dev/null || true
```

> If `.gitignore` does not exist yet, do not create it here — Task F creates it. If the commit fails because there are no changes, that is fine; move on.

---

### Task B — Tooling config (ruff + black + Makefile + lint script)

**Files:**
- Modify: `pyproject.toml`
- Create: `Makefile`
- Create: `scripts/lint.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: `make lint`, `make format`, `make test` targets; `scripts/lint.sh`; declarative ruff/black config that later tasks run against.

- [ ] **Step 1: Add tooling config to `pyproject.toml`**

Append these sections to the existing `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 120
exclude = ["*.egg-info", ".venv"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]
ignore = ["E501"]

[tool.ruff.format]
line-length = 120

[tool.black]
line-length = 120
target-version = ["py310"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `Makefile`**

```makefile
.PHONY: lint format test install

install:
	pip install -e .
	pip install pytest ruff black

lint:
	ruff check w2v_factory tests scripts
	black --check w2v_factory tests scripts

format:
	black w2v_factory tests scripts

test:
	pytest -q
```

- [ ] **Step 3: Create `scripts/lint.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Runs the full static-check + format-check gate (mirrors `make lint`).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ ! -d .venv ]; then
  echo "[!] no .venv found — run Task A first"
  exit 1
fi
. .venv/bin/activate
ruff check w2v_factory tests scripts
black --check w2v_factory tests scripts
echo "[✓] lint + format check clean"
```

- [ ] **Step 4: Make the script executable and verify config parses**

```bash
chmod +x scripts/lint.sh
. .venv/bin/activate
ruff --version && black --version && pytest --version
```
Expected: all three print version banners.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Makefile scripts/lint.sh
git commit -m "chore: add ruff/black/pytest config + Makefile + lint script"
```

---

### Task C — Core source fixes (device default + dead config + coercion + type hints)

**Files:**
- Modify: `w2v_factory/utils.py`
- Modify: `w2v_factory/config.py`
- Modify: `configs/base.yaml`
- Modify: `w2v_factory/cli/ablate.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pick_device(cfg_device: str) -> torch.device` — returns `cuda` (not `cuda:7`) on `"auto"` when CUDA available.
  - Dataclasses `RunCfg`/`TrainCfg`/`ModelCfg`/`EvalCfg` without the four dead fields.
  - `config.py` imports `_coerce_t` from `..data.subsample` and delegates subsample coercion to it.
  - Updated `configs/base.yaml` without the four dead keys.
  - Fully type-hinted `w2v_factory/cli/ablate.py`.

- [ ] **Step 1: Fix `pick_device` in `w2v_factory/utils.py`**

Replace the body of `pick_device` with:

```python
def pick_device(cfg_device: str) -> torch.device:
    """根据配置选择设备。'auto' 则优先 CUDA。"""
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)
```

- [ ] **Step 2: Remove dead config fields in `w2v_factory/config.py`**

Edit the dataclasses so they become:

```python
@dataclass
class RunCfg:
    out_dir: str = "runs/exp1"
    tb: bool = True
```

```python
@dataclass
class TrainCfg:
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 0.025
    lr_schedule: str = "linear"  # linear | none
    optimizer: str = "sgd"
    device: str = "auto"
```

```python
@dataclass
class ModelCfg:
    arch: str = "skipgram"  # skipgram | cbow
    dim: int = 300
    window: int = 5
    ns_neg_k: int = 5
    loss: str = "ns"        # ns | hs
    share_input_output: bool = False
```

```python
@dataclass
class EvalCfg:
    analogy_file: Optional[str] = None
```

- [ ] **Step 3: Delegate subsample coercion in `config.py`**

Add an import at the top of `config.py`:

```python
from ..data.subsample import _coerce_t
```

Replace the inline coercion block in `_from_dict` with:

```python
    data_kwargs = _filter_kwargs(DataCfg, d.get("DATA", {}))
    data_kwargs["subsample_t"] = _coerce_t(data_kwargs.get("subsample_t"))
    data = DataCfg(**data_kwargs)
```

- [ ] **Step 4: Update `configs/base.yaml`**

Remove these lines entirely:
- `save_every_steps: 5000` (under `RUN:`)
- `num_workers: 0` (under `TRAIN:`)
- `hs_use_huffman: true` (under `MODEL:`)
- `analogy_file: null` and `topk: 1` (the whole `EVAL:` block)

- [ ] **Step 5: Type-hint and tidy `w2v_factory/cli/ablate.py`**

Replace the top of the file and the untyped helpers so the file reads:

```python
from __future__ import annotations
import argparse
import os
from copy import deepcopy
from typing import Any, Dict, List
import itertools
import yaml
from loguru import logger

from ..config import load_cfg, _to_dict, _from_dict
from ..log import setup_logging
from ..trainer.engine import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation sweeps")
    p.add_argument("--grid", type=str, required=True, help="configs/ablation_grid.yaml")
    return p.parse_args()


def set_by_path(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Dict[str, Any] = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def main() -> None:
    args = parse_args()
    with open(args.grid, "r", encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)
    base_cfg_path = grid_cfg["base_cfg"]
    choices = grid_cfg["grid"]
    out_root = grid_cfg.get("out_root", "runs/ablations")
    base = load_cfg(base_cfg_path)
    base_d = _to_dict(base)

    keys = list(choices.keys())
    values = [choices[k] for k in keys]
    for vals in itertools.product(*values):
        d = deepcopy(base_d)
        name_parts = []
        for k, v in zip(keys, vals):
            set_by_path(d, k, v)
            name_parts.append(f"{k.replace('.', '_')}={v}")
        cfg = _from_dict(d)
        cfg.RUN.out_dir = os.path.join(out_root, "__".join(name_parts))
        setup_logging(cfg.RUN.out_dir)
        logger.info(f"Running ablation: {cfg.RUN.out_dir}")
        Trainer(cfg).train()


if __name__ == "__main__":
    main()
```

Note: keep the existing `import os` at the top too (add it beside the `from __future__`/stdlib imports if you removed it).

- [ ] **Step 6: Verify import + lint clean**

```bash
. .venv/bin/activate
python -c "import w2v_factory.config, w2v_factory.cli.ablate, w2v_factory.utils; print('ok')"
ruff check w2v_factory
```
Expected: prints `ok`; `ruff check` reports no errors (or only lines you haven't formatted yet — see Task H).

- [ ] **Step 7: Quick sanity — config still loads all configs**

```bash
. .venv/bin/activate
python -m w2v_factory.cli.train --help >/dev/null && echo "train ok"
```
Or simply:
```bash
. .venv/bin/activate
python - <<'PY'
from w2v_factory.config import load_cfg
for p in ["configs/base.yaml","configs/skipgram_ns.yaml","configs/cbow_hs.yaml","configs/paper_repro.yaml"]:
    load_cfg(p)
print("all configs load ok")
PY
```
Expected: prints `all configs load ok`.

- [ ] **Step 8: Commit**

```bash
git add w2v_factory/utils.py w2v_factory/config.py configs/base.yaml w2v_factory/cli/ablate.py
git commit -m "fix: device default, dead config keys, coercion centralization, ablate type hints"
```

---

### Task D — Test suite (math + config + smoke)

**Files:**
- Create: `tests/test_vocab.py`
- Create: `tests/test_subsample.py`
- Create: `tests/test_huffman.py`
- Create: `tests/test_sampler.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_config.py`
- Create: `tests/test_engine_smoke.py`
- Create: `tests/__init__.py` (empty)

**Interfaces:**
- Consumes: Task C's cleaned config; existing public functions (`build_vocab`, `compute_discard_probs`, `_coerce_t`, `build_huffman_codes`, `AliasSampler`, `build_unigram_sampler`, `evaluate_analogy`, `load_analogy`, `load_cfg`, `Trainer`, `Cfg`).
- Produces: a passing `pytest` suite that locks in the math contracts and a CPU smoke test.

- [ ] **Step 1: Create `tests/__init__.py`** (empty file)

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 2: Write `tests/test_vocab.py`**

```python
from w2v_factory.data.vocab import build_vocab


def test_build_vocab_min_count_and_ordering():
    stream = [["a", "b", "a", "c"], ["a", "b"], ["c", "d"]]
    v = build_vocab(stream, min_count=2, max_vocab=10)
    # counts: a=3, b=2, c=2, d=1 -> d drops below min_count
    assert v.itos == ["a", "b", "c"]
    assert v.stoi == {"a": 0, "b": 1, "c": 2}
    assert v.counts == [3, 2, 2]
    assert v.size == 3
    assert v.total_tokens == 8


def test_max_vocab_limits():
    stream = [["a"], ["b"], ["c"], ["d"]]
    v = build_vocab(stream, min_count=1, max_vocab=2)
    assert v.size == 2
```

- [ ] **Step 3: Write `tests/test_subsample.py`**

```python
import numpy as np
import pytest
from w2v_factory.data.subsample import _coerce_t, compute_discard_probs


@pytest.mark.parametrize("v", [None, "null", "None", "false", "", "0"])
def test_coerce_null_variants(v):
    assert _coerce_t(v) is None


def test_coerce_numeric():
    assert _coerce_t(1e-5) == pytest.approx(1e-5)
    assert _coerce_t("1e-5") == pytest.approx(1e-5)
    assert _coerce_t("5e-4") == pytest.approx(5e-4)


def test_discard_disabled_returns_zeros():
    probs = compute_discard_probs([10, 5], 100, None)
    assert probs.shape == (2,)
    assert (probs == 0).all()


def test_discard_enabled_shapes_and_bounds():
    counts = [1000, 500, 100, 50, 10, 1]
    probs = compute_discard_probs(counts, sum(counts), 1e-5)
    assert probs.shape == (len(counts),)
    assert probs.dtype == np.float32
    assert (probs >= 0).all() and (probs <= 1).all()


def test_discard_string_t_matches_float_t():
    counts = [9000, 900, 100]
    total = 10000
    p_num = compute_discard_probs(counts, total, 1e-5)
    p_str = compute_discard_probs(counts, total, "1e-5")
    np.testing.assert_allclose(p_num, p_str)
```

- [ ] **Step 4: Write `tests/test_huffman.py`**

```python
from w2v_factory.data.huffman import build_huffman_codes


def _unique_paths(paths):
    return len({tuple(p) for p in paths})


def test_inner_nodes_and_path_lengths():
    counts = [3, 2, 5, 1, 7]
    paths, codes = build_huffman_codes(counts)
    V = len(counts)
    assert len(paths) == V and len(codes) == V
    for p, c in zip(paths, codes):
        assert len(p) == len(c)          # one code bit per tree edge
        assert len(p) >= 1               # every word has a non-empty path


def test_unique_leaf_paths():
    counts = [3, 2, 5, 1, 7, 4, 6]
    paths, _ = build_huffman_codes(counts)
    assert _unique_paths(paths) == len(counts)


def test_codes_are_binary():
    counts = [3, 2, 5, 1, 7]
    _, codes = build_huffman_codes(counts)
    flat = [b for c in codes for b in c]
    assert set(flat) <= {0, 1}


def test_deterministic():
    counts = [3, 2, 5, 1, 7, 4, 6]
    p1, c1 = build_huffman_codes(list(counts))
    p2, c2 = build_huffman_codes(list(counts))
    assert p1 == p2 and c1 == c2
```

- [ ] **Step 5: Write `tests/test_sampler.py`**

```python
import numpy as np
from w2v_factory.data.sampler import AliasSampler, build_unigram_sampler


def test_alias_matches_distribution():
    probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    sampler = AliasSampler(probs)
    n = 200_000
    samples = sampler.sample(n)
    hist = np.bincount(samples, minlength=len(probs)) / n
    np.testing.assert_allclose(hist, probs / probs.sum(), atol=0.02)


def test_sample_shape_and_range():
    probs = np.array([0.25, 0.25, 0.5])
    sampler = AliasSampler(probs)
    samples = sampler.sample((8, 3))
    assert samples.shape == (8, 3)
    assert samples.min() >= 0 and samples.max() < 3


def test_unigram_sampler_has_same_support():
    counts = [1, 3, 2, 5, 4]
    sampler = build_unigram_sampler(counts)
    expected = counts / counts.sum()
    assert sampler.sample(50).min() >= 0
    assert sampler.sample(50).max() < len(counts)
```

- [ ] **Step 6: Write `tests/test_metrics.py`**

```python
import numpy as np
import pytest
from w2v_factory.trainer.metrics import load_analogy, evaluate_analogy


def test_load_analogy_skips_section_headers(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text(
        ": capital-common-countries\n"
        "paris france rome italy\n"
        ": grammar-adjective\n"
        "x y z w\n"
    )
    assert load_analogy(str(f)) == [("paris", "france", "rome", "italy"), ("x", "y", "z", "w")]


def test_evaluate_analogy_3cosadd():
    vecs = {
        "king": [10.0, 0.0],
        "man": [9.0, 0.0],
        "woman": [0.0, 9.0],
        "queen": [0.0, 10.0],
        "unused": [1.0, 1.0],
    }
    itos = list(vecs)
    stoi = {w: i for i, w in enumerate(itos)}
    emb = np.asarray([vecs[w] for w in itos], dtype=np.float64)
    items = [("king", "man", "woman", "queen"), ("a", "b", "c", "not_in_vocab")]
    acc = evaluate_analogy(emb, stoi, itos, items, topk=1)
    # first item hits; second skipped (d not in vocab)
    assert acc == pytest.approx(1.0)
```

- [ ] **Step 7: Write `tests/test_config.py`**

```python
import pytest
from w2v_factory.config import load_cfg


def test_include_merge(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("SEED: 7\nMODEL:\n  dim: 100\n  loss: ns\nTRAIN:\n  lr: 0.01\n")
    child = tmp_path / "child.yaml"
    child.write_text("INCLUDE: base.yaml\nMODEL:\n  loss: hs\n")
    cfg = load_cfg(str(child))
    assert cfg.SEED == 7
    assert cfg.MODEL.dim == 100
    assert cfg.MODEL.loss == "hs"
    assert cfg.TRAIN.lr == 0.01


def test_subsample_coercion_through_load(tmp_path):
    fnone = tmp_path / "a.yaml"
    fnone.write_text("DATA:\n  subsample_t: null\n")
    assert load_cfg(str(fnone)).DATA.subsample_t is None

    fstr = tmp_path / "b.yaml"
    fstr.write_text("DATA:\n  subsample_t: '1e-5'\n")
    assert load_cfg(str(fstr)).DATA.subsample_t == pytest.approx(1e-5)
```

- [ ] **Step 8: Write `tests/test_engine_smoke.py`**

The exact file content (readable, uses the public dataclasses, passes `ruff check`):

```python
import pytest
from w2v_factory.config import Cfg, RunCfg, DataCfg, TrainCfg, ModelCfg, EvalCfg
from w2v_factory.trainer.engine import Trainer

CORPUS = (
    "the quick brown fox jumps over the lazy dog\n"
    + "the quick brown fox jumps over the lazy dog\n"
    + "fox jumps over the dog\n"
    + "the dog sleeps on the mat\n"
    + "the dog is on the mat\n"
    + "fox and dog are quick animals\n"
) * 20


@pytest.mark.parametrize(
    "arch,loss",
    [("skipgram", "ns"), ("cbow", "ns"), ("skipgram", "hs"), ("cbow", "hs")],
)
def test_train_smoke(tmp_path, arch, loss):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS)

    out_dir = tmp_path / f"out_{arch}_{loss}"
    cfg = Cfg(
        SEED=0,
        RUN=RunCfg(out_dir=str(out_dir)),
        DATA=DataCfg(
            input_files=[str(corpus)],
            lowercase=True,
            min_count=2,
            max_vocab=10000,
            subsample_t=None,
            max_sent_len=1000,
        ),
        TRAIN=TrainCfg(epochs=1, batch_size=16, lr=0.01, lr_schedule="none",
                       optimizer="sgd", device="cpu"),
        MODEL=ModelCfg(arch=arch, dim=8, window=2, ns_neg_k=2, loss=loss,
                       share_input_output=False),
        EVAL=EvalCfg(),
    )
    Trainer(cfg).train()

    assert (out_dir / "embeddings.txt").exists()
    assert (out_dir / "embeddings.npy").exists()
```

- [ ] **Step 9: Run the full suite (expect pass)**

```bash
. .venv/bin/activate
pytest -q
```
Expected: all tests pass (the smoke test exercises all four arch×loss combos on CPU).

- [ ] **Step 10: Commit**

```bash
git add tests
git commit -m "test: add math, config, and smoke test suite"
```

> If the smoke test is slow, that's acceptable — it runs one epoch on ~120 short sentences across four combos. If a combination hangs or fails, debug with the systematic-debugging approach; the loss should be finite and all four models should train.

---

### Task E — Format & lint pass across the tree

**Files:**
- Modify: all under `w2v_factory/`, `tests/` (format/tidy only).

**Interfaces:**
- Consumes: Task D's code.
- Produces: a tree that passes `make lint` (ruff check + `black --check`).

- [ ] **Step 1: Auto-format source and tests**

```bash
. .venv/bin/activate
black w2v_factory tests scripts
```

- [ ] **Step 2: Lint and fix**

```bash
. .venv/bin/activate
ruff check w2v_factory tests scripts
```
Expected: exit 0. If `ruff` flags unused imports or trivial issues, fix them (e.g., remove unused `import numpy` if flagged). Do **not** change behavior.

- [ ] **Step 3: Re-run tests to confirm nothing broke**

```bash
. .venv/bin/activate
pytest -q
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "style: format with black and clean up ruff warnings"
```

---

### Task G — Housekeeping: .gitignore, pycache prune, LICENSE, CONTRIBUTING

**Files:**
- Create: `.gitignore`
- Delete: tracked `__pycache__` and `*.pyc`
- Create: `LICENSE`
- Create: `CONTRIBUTING.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a clean, ignorable working tree; a MIT license; a short contribution guide.

- [ ] **Step 1: Create `.gitignore`**

```
.venv/
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
runs/
.pytest_cache/
.ruff_cache/
.DS_Store
```

- [ ] **Step 2: Remove tracked compiled artifacts**

```bash
git rm -r --cached w2v_factory/**/__pycache__ w2v_factory/__pycache__ 2>/dev/null || true
git rm -q --cached $(git ls-files '*.pyc') 2>/dev/null || true
rm -rf $(git ls-files '**/__pycache__' | xargs -n1 dirname) 2>/dev/null || true
```
Verify none tracked:
```bash
git ls-files | grep -E "__pycache__|\.pyc$" || echo "none tracked"
```
Expected: prints `none`.

- [ ] **Step 3: Create `LICENSE` (MIT)**

```
MIT License

Copyright (c) 2026 word2vec-factory contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Create `CONTRIBUTING.md`**

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore LICENSE CONTRIBUTING.md
git rm -r --cached $(git ls-files '**/__pycache__') 2>/dev/null || true
git commit -am "chore: add gitignore, MIT license, contributing guide; prune pycache"
```

---

### Task H — README rewrite

**Files:**
- Modify: `README.md` (full rewrite, English)

**Interfaces:**
- Consumes: nothing.
- Produces: the project landing page; links to each `docs/` page.

- [ ] **Step 1: Write the new `README.md`**

```markdown
# word2vec-factory

A pluggable, factory-driven Word2Vec implementation for **CBOW / Skip-gram**
with **Hierarchical Softmax (Huffman)** and **Negative Sampling**, built for
industrial ablation and for learning word2vec from scratch.

## Features

- Factory registries for models, losses, tokenizers, and schedulers.
- Accurate Huffman Hierarchical Softmax.
- Negative Sampling with unigram^0.75 alias-table sampling.
- Subsampling of frequent words and randomized context windows (paper-style).
- Linear LR decay, TensorBoard, Loguru logging, reproducible seeds.
- OOV stories: Google Analogy evaluation and batch ablation sweeps.

## Installing

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Development tools (tests, linting):

```bash
pip install pytest ruff black
```

## Quickstart

```bash
# Train Skip-gram + Negative Sampling on the sample corpus
python -m w2v_factory.cli.train --cfg configs/skipgram_ns.yaml

# Train CBOW + Hierarchical Softmax
python -m w2v_factory.cli.train --cfg configs/cbow_hs.yaml

# Evaluate on the Google analogy set
python -m w2v_factory.cli.eval --vectors runs/exp1/embeddings.txt \
    --analogy questions-words.txt

# Run the full ablation grid
python -m w2v_factory.cli.ablate --grid configs/ablation_grid.yaml
```

## Project layout

```
w2v_factory/
  cli/            command-line tools (train, eval, ablate)
  data/           vocab, subsampling, tokenizers, Huffman, alias sampling
  losses/         hierarchical softmax + negative sampling packers
  models/         CBOW and Skip-gram
  trainer/        engine, exporter, lr schedulers, evaluation metrics
  config.py       dataclass config + YAML loader (with INCLUDE)
  registry.py     factory registry
configs/          run and ablation YAML files
scripts/          conda setup, train/eval/ablate helpers, lint
sample_corpus.txt   toy corpus for testing
questions-words.txt Google analogy dataset
```

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration reference](docs/config-reference.md)
- [Ablations](docs/ablation.md)
- [Evaluation](docs/evaluation.md)
- [From-scratch notes](docs/from-scratch.md)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: overhaul README landing page"
```

---

### Task I — `docs/architecture.md`

**Files:**
- Create: `docs/architecture.md`

**Interfaces:**
- Consumes: nothing.
- Produces: an English architecture overview of the factory registry, module
  responsibilities, and data flow.

- [ ] **Step 1: Write `docs/architecture.md`**

```markdown
# Architecture

This document describes how `word2vec-factory` is organized and how data flows
through a training run.

## Design goals

- **Pluggable** pieces (models, losses, tokenizers, schedulers) exposed through a
  small registry, so ablations can swap one component without touching the
  training loop.
- **From scratch**: every algorithm (Huffman, 3Cos scores, subsampling) is
  implemented directly — no `gensim` underneath.

## Factories and the registry

`w2v_factory/registry.py` defines a tiny `Registry` with register/get/create.
Four global registries exist:

| Registry | Holds | Example |
| --- | --- | --- |
| `MODEL_REG` | CBOW / Skip-gram | `SkipGram` | 
| `LOSS_REG` | NS / HS packing & losses | `negative_sampling` |
| `TOKENIZER_REG` | tokenizers | `simple` |
| `SCHED_REG` | lr schedulers | `linear` |

The engine calls the concrete `forward_ns` / `forward_hs` methods directly for
clarity; the registry keeps new components easy to add and ablate.

## Data flow

```
raw text files
   │  iter_tokens()                      (line-by-line tokenization)
   ▼
build_vocab()                            (min_count, max_vocab)
   │
compute_discard_probs()                  (subsample_t)
   ▼
SentenceIndexer.encode()                 (ids + frequent-word dropping)
   │
generate_{skipgram,cbow}_pairs()  (randomized window)
   ▼
batching + packing (HS paths or negative samples) + forward pass
   ▼
backward + optimizer.step + (lr decay) + logging
```

## Model responsibilities

- `models/base.py`: shared in/out embeddings and init.
- `models/skipgram.py`, `models/cbow.py`: forward computations for NS and HS.
- The **context** is `[B, Lmax]` padded; a mask selects real tokens.

## Loss math

- **Negative sampling**: objective is `log σ(v·u₀) + Σ_neg log σ(-v·u_k)`.
- **Hierarchical softmax**: each word gets a root-to-leaf Huffman path; the loss is
  the product of per-node Bernoulli probabilities `σ((2code-1)·v·u_node)`.

See `docs/from-scratch.md` for derivations.
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: add architecture overview"
```

---

### Task J — `docs/config-reference.md`

**Files:**
- Create: `docs/config-reference.md`

**Interfaces:**
- Consumes: the (now cleaned) config schema from Task C.
- Produces: a reference table of every config key, default, and meaning.
  Keep identical to `configs/base.yaml`.

- [ ] **Step 1: Write `docs/config-reference.md`**

```markdown
# Configuration reference

All run-level settings live in YAML and are loaded by `w2v_factory.config.load_cfg`.
A config can `INCLUDE` another file to inherit and override it.

## Top-level

| Key | Default | Meaning |
| --- | --- | --- |
| `SEED` | `42` | global random seed |

## `RUN`

| Key | Default | Meaning |
| --- | --- | --- |
| `out_dir` | `runs/exp1` | output directory for vectors + logs |
| `tb` | `true` | write TensorBoard log t |

## `DATA`

| Key | Default | Meaning |
| --- | --- | --- |
| `input_files` | `[]` | list of corpus paths (one sentence per line) |
| `lowercase` | `true` | lowercase tokens |
| `min_count` | `5` | drop words in the corpus below this count |
| `max_vocab` | `1000000` | cap vocabulary size |
| `subsample_t` | `1e-5` | frequent-word subsample threshold; `null`/`'null'` disables |
| `tokenizer` | `simple` | tokenizer name (`simple`) |
| `max_sent_len` | `10000` | truncate over-long lines |

## `TRAIN`

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | `10` | number of passes over the corpus |
| `batch_size` | `1024` | number of target/completed examples per step |
| `lr` | `0.025` | learning rate |
| `lr_schedule` | `linear` | `linear` or `none` |
| `optimizer` | `sgd` | `sgd` or `adam` |
| `device` | `auto` | `auto`, `cpu`, or `cuda:N` |

## `MODEL`

| Key | Default | Meaning |
| --- | --- | --- |
| `arch` | `skipgram` | `skipgram` or `cbow` |
| `dim` | `300` | embedding size |
| `window` | `5` | max context window |
| `ns_neg_k` | `5` | number of negatives for NS |
| `loss` | `ns` | `ns` or `hs` |
| `share_input_output` | `false` | share the input/output embedding matrices |
```

> Note: the config was cleaned so every key above actually affects training
> (removed dead keys like `num_workers`, `save_every_steps`, `hs_use_huffman`).

- [ ] **Step 2: Commit**

```bash
git add docs/config-reference.md
git commit -m "docs: add config reference"
```

---

### Task K — `docs/ablation.md`

**Files:**
- Create: `docs/ablation.md`

**Interfaces:**
- Consumes: nothing.

- [ ] **Step 1: Write `docs/ablation.md`**

```markdown
# Ablations

Run ablations with a grid file. See `configs/ablation_grid.yaml`.

```yaml
grid:
  MODEL.arch: ["skipgram", "cbow"]
  MODEL.loss: ["hs", "ns"]
  MODEL.dim: [100, 300, 600]
  MODEL.window: [5, 10]
  TRAIN.epochs: [5, 10]
  DATA.subsample_t: [null, 1e-5]
base_cfg: configs/base.yaml
out_root: runs/ablations
```

Run the grid (cartesian product of every choice):

```bash
python -m w2v_factory.cli.ablate --grid configs/ablation_grid.yaml
```

Each combination runs the full training pipeline and writes vectors to:

```
runs/ablations/MODEL_arch=skipgram__MODEL_loss=hs__.../embeddings.txt
```

The folder name encodes every varied key, so it is easy to compare runs.

## Tips

- Keep the base cfg fixed; point `base_cfg` at the config that sets `DATA.input_files`.
- Set `DATA.subsample_t` to `null` in the grid when you want to disable it.
- Evaluate each produced `embeddings.txt` with `scripts/eval_analogy.sh`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/ablation.md
git commit -m "docs: add ablation guide"
```

---

### Task L — `docs/evaluation.md`

**Files:**
- Create: `docs/evaluation.md`

**Interfaces:**
- Consumes: nothing.

- [ ] **Step 1: Write `docs/evaluation.md`**

```markdown
# Evaluation

Evaluated on the Google Analogy task (the shipped `questions-words.txt`):
`a : b :: c : d`, solved with 3CosAdd.

## Usage

```bash
python -m w2v_factory.cli.eval --vectors runs/exp1/embeddings.txt \
    --analogy questions-words.txt [--topk 1]
```

Or use the helper:

```bash
bash scripts/eval_analogy.sh [vectors.txt] [questions-words.txt]
```

## Metrics

`evaluate_analogy` returns the fraction of correct analogies under
3CosAdd: `b - a + c`, candidate-scored by cosine similarity, excluding the
`a`/`b`/`c` tokens. Only questions whose tokens are all in the vocabulary count.

## File format

`embeddings.txt` is the standard `word2vec` text format:

```
|V| dim
word x.xxxx x.xxxx ...
```

Loaded by `load_word2vec_txt` in `w2v_factory/cli/eval.py`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/evaluation.md
git commit -m "docs: add evaluation guide"
```

---

### Task M — `docs/from-scratch.md`

**Files:**
- Create: `docs/from-scratch.md`

**Interfaces:**
- Consumes: nothing.

- [ ] **Step 1: Write `docs/from-scratch.md`**

```markdown
# From-scratch notes

Derivations for the algorithms implemented in this repo. Math here uses `v`
for embedding (context/input) and `u` for output/context vectors.

## Skip-gram objective

Predict the context words given a center word. For a positive pair `(center, ctx)`
the score is the dot product of the center vector and the context vector.

```
score(center, ctx) = v_center · u_ctx
```

## Negative sampling

Negative sampling (NS) approximates `log P(posit)` with:

```
L = log σ(v·u) + Σ_k log σ(-v·u_k)
```

where `u_k` are drawn from unigram^0.75 via an alias table.

## Hierarchical softmax

Huffman tree over words; each leaf has a root-to-leaf path of internal nodes.
Assign code `1` for right, `0` for left at each edge. The loss is:

```
L = -Σ_i log σ( (2·code_i - 1) · v · u_i )
```

The gradient only crosses the path nodes, making it O(path) per word.

## Subsampling

Frequent words are dropped with probability `p = 1 - √(t/f)`, clipped to
`[0,1]`, reducing them prevalence and speeding up training.

## Randomized window

Each positive uses a window sampled in `[1, window]`, the paper’s trick for
robustness.

## Why an alias table

Building the unigram^0.75 table once and sampling in O(1) avoids a per-step
`O(V)` scan. `data/sampler.py` implements Vose’s alias method.
```

- [ ] **Step 2: Commit**

```bash
git add docs/from-scratch.md
git commit -m "docs: add from-scratch notes"
```

---

### Task N — Final verification

**Files:**
- none (verification only).

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Full lint, format check, test run**

```bash
. .venv/bin/activate
make lint
pytest -q
```
Expected: `make lint` exit 0; `pytest` all green.

- [ ] **Step 2: Confirm all shipping configs load + README renders**

```bash
. .venv/bin/activate
python - <<'PY'
from w2v_factory.config import load_cfg
for p in ["configs/base.yaml","configs/skipgram_ns.yaml","configs/cbow_hs.yaml","configs/paper_repro.yaml"]:
    load_cfg(p)
print("configs ok")
PY
```
Expected: `configs ok`.

- [ ] **Step 3: Commit any remaining changes**

```bash
git add -A
git commit -m "chore: final lint/test pass" || echo "nothing to commit"
```

- [ ] **Step 4: Summarize**

Report: files added/changed, `pytest` summary line, lint status, and the
implied freedom (no GPU required) — smoke tests run on CPU.
```

## Self-Review

- **Spec coverage:** Every spec section maps to a task: Section 1 (code) → Tasks C, D, E (+B tooling); Section 2 (docs) → Tasks H–M; Section 3 (housekeeping) → Task G. The spec’s "Out of scope" is honored (no CI, no engine rewrite, no Chinese docs prose). ✔
- **Placeholder scan:** No TBD/TODO; every step has explicit code/commands. Removed the exploratory pseudo-code scaffolding in `test_engine_smoke.py` and replaced with the final clean file (reads in Step 8). ✔
- **Type consistency:** `pick_device` signature unchanged; `_coerce_t` import path `w2v_factory.data.subsample._coerce_t` matches production; dataclass field name `subsample_t` consistent across `config.py`, tests, and docs; `Trainer`, `Cfg`, `RunCfg`, `DataCfg`, `TrainCfg`, `ModelCfg`, `EvalCfg` names used consistently across tests and smoke test. ✔
- **Formatter consistency:** `line-length = 120` in both ruff and black so they agree and minimize churn (only 4 lines > 119 in the tree). ✔