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
pass 1: raw text files
   │  iter_tokens()                      (line-by-line)
   ▼
build_vocab()                            (counts + token-stream SHA-256)
   │
   ├── compute_discard_probs()
   └── build_training_plan()             (token-progress denominator)

passes 2..E+1: reread raw text each epoch
   │
   ▼
SentenceIndexer.encode_with_positions()  (epoch-specific subsampling)
   │
   ▼
generate_*_pairs_with_progress()         (epoch-specific window RNG)
   │
   ▼
bounded batch buffer                     (<= TRAIN.batch_size examples)
   │
   ├── token-progress LR
   └── HS paths / isolated NS RNG
   ▼
forward → backward → optimizer.step()
```

## Model responsibilities

- `models/base.py`: shared in/out embeddings and init.
- `models/skipgram.py`, `models/cbow.py`: forward computations for NS and HS.
- The **context** is `[B, Lmax]` padded; a mask selects real tokens.

## Loss math

- **Negative sampling**: objective is `log σ(v·u₀) + Σ_neg log σ(-v·u_k)`.
- **Hierarchical softmax**: each word gets a root-to-leaf Huffman path; the loss is
  the product of per-node Bernoulli probabilities `σ((2code−1)·v·u_node)`. Only the
  `V-1` internal Huffman nodes receive output embeddings; path IDs are compact in
  `[0, V-2]`.

See `docs/from-scratch.md` for derivations.

## Training control and reproducibility

Training randomness is split into independent streams for per-epoch subsampling,
negative sampling, and per-epoch context windows. Linear LR decay follows
in-vocabulary source-token progress, so randomized pair cardinality never changes
the schedule denominator. The engine rereads the corpus per epoch and keeps only
one sentence plus a bounded example batch. Each epoch also recomputes the
token-stream fingerprint to detect corpus/tokenizer drift. Successful runs write
`run_manifest.json` with token progress, stream statistics, RNG assignments,
and peak buffered examples.

See [training-plan.md](training-plan.md) for the invariants and rationale.
