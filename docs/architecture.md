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
build_epoch_plans()                     (exact examples + optimizer steps)
   │
generate_{skipgram,cbow}_pairs()         (per-epoch deterministic window RNG)
   ▼
batching + packing (HS paths or isolated negative-sampling RNG) + forward pass
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
  the product of per-node Bernoulli probabilities `σ((2code−1)·v·u_node)`. Only the
  `V-1` internal Huffman nodes receive output embeddings; path IDs are compact in
  `[0, V-2]`.

See `docs/from-scratch.md` for derivations.

## Training control and reproducibility

Training randomness is split into independent streams for subsampling, negative
sampling, and per-epoch context windows. The context-window stream is replayed
during planning so the linear LR scheduler receives an exact optimizer-step
count. Successful runs write `run_manifest.json` with the resolved config,
epoch plans, RNG stream assignments, and planned/actual step counts.

See [training-plan.md](training-plan.md) for the invariants and rationale.
