# word2vec-factory

A pluggable, factory-driven Word2Vec implementation for **CBOW / Skip-gram**
with **Hierarchical Softmax (Huffman)** and **Negative Sampling**, built for
industrial ablation and for learning word2vec from scratch.

## Features

- Factory registries for models, losses, tokenizers, and schedulers.
- Accurate Huffman Hierarchical Softmax.
- Negative Sampling with unigram^0.75 alias-table sampling.
- Subsampling of frequent words and randomized context windows (paper-style).
- Bounded-memory corpus streaming, token-progress linear LR decay, isolated deterministic RNG streams, atomic epoch checkpoints/resume, TensorBoard, and Loguru logging.
- Google Analogy evaluation and batch ablation sweeps.

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
  trainer/        streaming engine, token-progress planner, exporter, lr schedulers, metrics
  config.py       dataclass config + YAML loader (with INCLUDE)
  registry.py     factory registry
configs/          run and ablation YAML files
scripts/          conda setup, train/eval/ablate helpers, lint
sample_corpus.txt   toy corpus for testing
questions-words.txt Google analogy dataset
```

## Documentation

- [Architecture](docs/architecture.md)
- [Streaming training and token progress](docs/training-plan.md)
- [Checkpoint and resume](docs/checkpoint-resume.md)
- [Configuration reference](docs/config-reference.md)
- [Ablations](docs/ablation.md)
- [Evaluation](docs/evaluation.md)
- [From-scratch notes](docs/from-scratch.md)

## Paper-grounded factory ablations

[Original 1301 paper-to-code traceability](docs/papers/1301.3781-traceability.md) · [1310 extensions](docs/papers/1310.4546-extensions.md) · [controlled experiment protocol](docs/experiments/protocol.md) · [stage gates](docs/experiments/roadmap.md).

```bash
pip install -e '.[viz]'
python -m w2v_factory.cli.paper_ablate --suite configs/experiments/smoke.yaml   # plan only
python -m w2v_factory.cli.paper_ablate --suite configs/experiments/smoke.yaml --run
python -m w2v_factory.cli.paper_ablate --suite configs/experiments/smoke.yaml --report-only
```

Experiments use real Factory registries for model/optimizer and experimental design, plus
seed-matched one-factor controls or explicit factorial studies. The runner writes
structured JSONL, category/coverage-aware analogy metrics, CSV/JSON/Markdown
summaries, seed-level bootstrap summaries and optional PNG/SVG plots. **Smoke scores are not the paper's scores**.
The original 1301 paper predates the negative-sampling/subsampling extensions in 1310;
they have separate configurations and evidence maps.

## Run artifacts

A successful training run writes `embeddings.txt`, `embeddings.npy`, and
`run_manifest.json`. The manifest records the resolved configuration, token-progress plan, RNG stream
assignments, selected device, vocabulary statistics, per-epoch streaming
counters, actual optimizer steps, and peak buffered examples.
