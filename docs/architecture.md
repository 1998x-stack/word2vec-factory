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
generate_{skipgram,cbow}_pairs()         (randomized window)
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
  the product of per-node Bernoulli probabilities `σ((2code−1)·v·u_node)`.

See `docs/from-scratch.md` for derivations.