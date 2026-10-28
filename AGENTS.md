# AGENTS.md — Word2Vec Factory agent contract

Repository-level instructions for Codex, OpenCode, Claude Code and other coding agents. Read [README.md](README.md) and [docs/README.md](docs/README.md) for project scope. This document complements [CONTRIBUTING.md](CONTRIBUTING.md); verified code and tests take precedence over obsolete historical plans.

## Mission

Maintain a **research-grade, factory-mode, from-scratch PyTorch Word2Vec laboratory**. Prioritize mathematical correctness, clean experiment isolation, reproducibility and auditability. Avoid turning this repository into an unrelated general-purpose NLP framework.

**Paper provenance is a release gate:**
- [arXiv:1301.3781](https://arxiv.org/abs/1301.3781): CBOW, Skip-gram, Huffman HS, random context distances, analogy tasks and scaling studies.
- [arXiv:1310.4546](https://arxiv.org/abs/1310.4546): Negative Sampling, word subsampling and related extensions.
- Never claim to have reproduced DistBelief, original Google News/LDC corpora, NNLM/RNNLM or Sentence Completion without code, licensed data and measured results.
- Never fabricate performance claims, charts, evidence, paper results or test outcomes.

## Module boundaries

| Directory / file | Owns |
| --- | --- |
| `w2v_factory/config.py`, `configs/` | Strict configuration schema, YAML inheritance, reproducible inputs |
| `w2v_factory/registry.py` | Model/loss/tokenizer/scheduler factories |
| `w2v_factory/models/` | CBOW and Skip-gram forward passes |
| `w2v_factory/data/` | Streaming tokenization, vocab, Huffman, sampling, windows |
| `w2v_factory/losses/` | Negative draw and hierarchical-softmax batch preparation |
| `w2v_factory/trainer/` | Shared training loop, SGD/LR, checkpoint, metrics and manifests |
| `w2v_factory/experiments/` | OFAT/grid design, category-aware analogy evaluation, reporting |
| `w2v_factory/cli/` | Training, evaluation, ablation, reports and corpus budgeting |
| `tests/` | Deterministic regression, numerical and end-to-end tests |
| `docs/` | Paper traceability, methodology, implementation constraints and status |

Use `MODEL_REG`, `LOSS_REG`, `TOKENIZER_REG`, `SCHED_REG` and `EXPERIMENT_REG` at their existing dispatch points. New ablations should be config-driven; do **not** copy training loops or silently create different mathematical implementations per experiment.

## Mathematical invariants — protect with tests

1. **Skip-gram** input is center word, positive target is the **context word**, including Huffman paths for HS. Preserve `tests/test_training_correctness.py`.
2. **CBOW** mean pooling must mask padding and divide by real context length. The `sum` intervention is a distinct ablation, not the paper's default.
3. **HS** uses V−1 internal Huffman output nodes; paths/codes, signs and vectorized gradients must match a scalar reference.
4. **NS** uses the configured unigram exponent, independent RNG and rejection of positive/negative collisions. Any change to sampling distribution is a protocol change.
5. **Context windows** keep fixed versus uniform-sampled radius distinct; sampled distances, sentence boundaries and token progress must be correct.
6. **3CosAdd** computes `q = v_b - v_a + v_c` from **raw input vectors**, then cosine-ranks candidates, excluding input words `a`, `b` and `c`. Never normalize the three offset terms independently before subtraction. Report per-category `total / eligible / correct / coverage`.
7. **Training** preserves bounded streaming, corpus SHA-256 identity, monotonic token-progress LR, separated RNG streams, and epoch-boundary checkpoint/resume checks. Reproducible CPU continuation does not imply cross-device bit-for-bit reproducibility.
8. **Science** separates implementation, automated verification, and **real-corpus measurement**. The bundled fairy-tale corpus is a smoke-test input, not original paper data.

## Working procedure

1. **Inspect:** verify branch, latest PR, impacted modules, current docs and existing tests. Use a new feature branch; do not silently push directly to `main` or merge a PR.
2. **Scope:** state the paper section, hypothesis, independent intervention, confounders, dataset assumptions and objective acceptance checks. Minimize unrelated refactors.
3. **Implement:** extend the correct factory / schema / config / test boundary. Update checkpoint signature semantics and manifests when training behavior changes. Keep Python >=3.10 and readable PyTorch/NumPy code.
4. **Verify:** add scalar/gradient/property regression tests where appropriate, then run the suite. Test the four CBOW/SG × HS/NS combinations when changing shared logic. Check errors for filtered corpora, malformed configs, one-word vocab, OOV and negative collisions.
5. **Budget:** always use `--dry-run` for ablation plans before training. Check seeds, `max_runs`, tokenizer, corpus hash, actual active factors and output isolation. Do **not** launch large sweeps as routine validation without explicit user request.
6. **Document:** update relevant config reference, paper-to-code matrix, evaluation protocol, README / documentation index. Report whether each requirement is implemented, tested, measured or pending.
7. **Submit:** make small reviewable commits. Use a PR and state exact CI / local checks and incomplete tests. Do not claim success before evidence is available.

## Verification commands (from repository root)

```bash
python -m pip install -e '.[viz]'
python -m pip install pytest ruff black

python -m compileall -q w2v_factory tests
python -m pytest -q

python -m w2v_factory.cli.ablate \
  --grid configs/paper1301_ablations.yaml --dry-run

python -m w2v_factory.cli.ablate \
  --grid configs/post2013_extensions.yaml --dry-run

# Optional style gate. If pre-existing files fail, report separately.
make lint
```

Do not run `python -m w2v_factory.cli.ablate` without `--dry-run` as an automatic CI or review step. Data-heavy training requires explicit project scope, licensed corpora and sufficient compute.

## Experiment design and evidence rules

- `one_factor` experiments vary one active control at a time against the same baseline; historical `profiles` may change multiple factors and cannot establish individual causal effects.
- `grid` combinations require explicit cost bounding. Avoid declaring NS-specific `ns_neg_k`/`ns_power` as meaningful HS ablations.
- Track each seed and configuration signature; report OOV coverage and benchmark SHA-256. Show all seed results before aggregating; three paired seeds are descriptive, not automatic statistical significance.
- Preserve logs and artifacts (`run_manifest.json`, `experiment.json`, `evaluation.json`, `events.jsonl`, `metrics.jsonl`, `summary.csv`). Never invent run output.
- Different word coverage, device, corpus, token budget or number of SGD steps may confound quality/speed comparisons. PCA is a diagnostic, **not** an accuracy benchmark.

## File hygiene and documentation

- Executable modules belong to `w2v_factory/`, YAML to `configs/`, tests to `tests/`, helper scripts to `scripts/` and current docs to `docs/`.
- `docs/superpowers/` is historical planning with superseded assumptions; never treat it as an active implementation contract.
- Avoid duplicate documentation. Register new current docs in [docs/README.md](docs/README.md) and new experiments in [configs/README.md](configs/README.md).
- Preserve stable paths where possible. If moving files, update all internal links and references in source/config/scripts.
- Chinese and English documentation are both allowed. Keep Python names, CLI options and configuration keys in English and technical descriptions consistent.
- Never commit `.venv/`, `runs/`, `data/`, training weights, checkpoint files, user data, secrets or machine-specific paths. Software MIT licensing does not grant a license to all third-party training data.

## Ready-to-submit checklist

- [ ] Paper attribution and experimental hypotheses are accurate.
- [ ] Factories, configs, code and docs describe identical behavior.
- [ ] Math/gradient/edge-case tests and relevant smoke tests pass, or blockers are reported.
- [ ] Experiment plan inspected with `--dry-run`; no unapproved expensive execution.
- [ ] Documentation links resolve and new docs are indexed.
- [ ] Results use real evidence, clear OOV/coverage, provenance and uncertainty.
- [ ] Commit and PR are reviewable; CI status and remaining gaps are explicit.
