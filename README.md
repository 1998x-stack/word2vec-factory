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