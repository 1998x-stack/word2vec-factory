# Controlled experiment protocol (Factory Mode)

## Objective

Build falsifiable, reproducible evidence around word2vec architectures, not a leaderboard
of cherry-picked runs. Experiments are classified as `original_1301`,
`followup_1310` or `engineering` and carry an explicit hypothesis.

## Architecture and extensibility

```text
configs/experiments/*.yaml
  -> load_suite() (schema, provenance, budgets)
  -> DESIGN_REG: one_factor | factorial
  -> build_plan() (immutable Case identities, seed-matched controls)
  -> Trainer(cfg)
      -> MODEL_REG: cbow | skipgram
      -> OPTIMIZER_REG: sgd | adam
      -> dataset tokenizer factory & LR scheduler
      -> NS or HS training forward
  -> evaluate by category / coverage
  -> JSONL events + per-run manifests + CSV/JSON/Markdown summaries
  -> publication-style PNG / SVG charts
```

The standard `Cfg`, streaming corpus fingerprint, optimizer and RNG checkpoints
are shared with the legacy CLI. The old grid runner is preserved for compatibility;
new experiment results use `cli.paper_ablate`.

## How to run

```bash
pip install -e '.[viz]'
python -m pytest -q
python -m w2v_factory.cli.paper_ablate --suite configs/experiments/smoke.yaml
# Inspect runs/paper-smoke/plan.json: this is dry-run.
python -m w2v_factory.cli.paper_ablate --suite configs/experiments/smoke.yaml --run
python -m w2v_factory.cli.paper_ablate --suite configs/experiments/smoke.yaml --report-only
```

Run original and follow-up suites after providing a real, legal corpus of suitable size.
Set `DATA.input_files`, tokenizer, vocab threshold, `max_sent_len` and fixed evaluation
before experiments; the bundled sample text cannot support claims about 2013 benchmark scores.

## Primary metrics (always report denominators)

- `accuracy_covered = correct / eligible`, where eligible means all 4 analogy terms in vocabulary;
  `null` when zero questions covered.
- `coverage = eligible / total`; `accuracy_all = correct / total`.
- Semantic/syntactic and each category: total, eligible, correct, coverage and both accuracies.
- Cost: end-to-end training wall time (includes vocabulary, export and manifest), examples/s
  and in-vocabulary source tokens/s. Distinguish classifications and source token counts.
- Resource profile: vocabulary size, dimension, actual examples, optimizer steps,
  peak batch buffer, device and environment info; compare under matched conditions.

## Statistical design

Use at least 3 paired seeds (prefer 5+ on publication-scale work).
For OFAT, compare a treatment with its *same-suite same-seed* control.
Compute paired delta `d_s = metric_treatment_s - metric_control_s` then report
mean, SD and a confidence interval across seeds. A single seed gives no variance estimate.
For factorial experiments, fit treatment, interaction and dataset effects only after
checking that all cells have the same random seeds and fixed corpus/evaluation protocol.

Quality vs compute: report (a) same data / same training policy and (b) matched
wall-time or token-budget curves. Architecture and data size may affect number
of training examples; equal epochs do not imply equal updates or FLOPs.
Use small screening runs to identify active factors; avoid executing a huge
uncontrolled Cartesian product.

## Reproducibility/known limitations

- Case IDs/hash include model/data/train/eval settings and seed, excluding output directory;
  existing completed output is reused only when hash and artifacts agree.
- Each run has `experiment_metadata.json`, `run_manifest.json`, `embeddings.npy`,
  `embeddings.txt`, `case_result.json` and optionally `analogy_metrics.json`.
- Suite root has `plan.json`, `events.jsonl`, `aggregate.json`, `summary.csv`,
  `summary.json`, `summary.md` and `plots/*.png|svg` if Matplotlib is installed.
- Training has TensorBoard scalars and Loguru file logs from the original engine.
- Histories from different GPU kernels, PyTorch versions and machines are not guaranteed bitwise identical.
- Recovered incomplete training requires the existing epoch-checkpoint mechanism;
  the ablation runner does not auto-resume half-written cases.
- Dataset fingerprint checking catches drift across epochs; generated dataset
  prefixes must be created reproducibly and recorded with their source digest.
- Leakage from training text into analogy evaluation should be documented. Do not
  treat analogy-score optimization as a held-out modern downstream validation.

## Acceptance gates

1. Every plan has a stable ID/hash and validates all dotted fields and budgets.
2. One-factor controls are identical except the varied axis and same seed; factorial
   cells cover the product once.
3. CBOW fixed context and Skip-gram random context match documented protocol;
   HS gradients/targets and NS positive exclusions retain regression coverage.
4. Every completed run has a readable manifest and summary, no invented metrics.
5. Analogies exclude query terms, report missing-word coverage, and split semantic vs syntactic.
6. Non-finite embeddings and malformed analogy rows fail loudly.
7. Each plot uses measurements from `case_result.json`, never synthetic demonstration results.
8. Continuous integration runs compile, lint and pytest including deterministic plan/eval tests.
