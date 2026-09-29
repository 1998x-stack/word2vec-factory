# Streaming training and token-progress scheduling

Word2Vec training uses randomized subsampling, context windows, and negative
sampling. Large corpora also make it important that the training loop does not
materialize all encoded sentences or positive pairs in memory.

## Corpus passes and memory model

A run performs:

1. one streaming pass to build the vocabulary and word counts;
2. one independent streaming training pass per epoch.

During a training pass, only the current tokenized sentence, the current example
batch, vocabulary structures, and model state are resident. The corpus is not
stored as `list[list[int]]`.

Ignoring model/vocabulary state, corpus-side peak memory is therefore bounded by
approximately `O(max_sent_len + batch_size * context_width)`, rather than
`O(corpus_tokens)`.

The vocabulary `Counter`, embedding tables, Huffman tree or alias sampler still
scale with vocabulary size.

## Token-progress LR schedule

A streaming trainer cannot know the exact randomized Skip-gram pair count without
performing another pass or replaying all window draws. Instead, linear LR decay is
driven by deterministic **source-token progress**.

`build_training_plan()` uses the sum of retained vocabulary counts as the number
of trainable source tokens per epoch. Across `E` epochs:

```
total_progress_tokens = sum(vocab.counts) * E
```

Each retained training target/center carries its ordinal in that source-token
stream. Tokens removed by subsampling still advance the ordinal; OOV tokens do
not. A batch uses the progress at the start of the batch to set its LR. At the
end of each epoch the scheduler advances to the exact epoch boundary, including
tokens that produced no training example.

This makes LR independent of random window width, negative-sampling retries, and
subsampling outcomes.

## RNG topology

The trainer derives independent NumPy generators from the configured root
`SEED`:

| Stream | Purpose | Lifetime |
| --- | --- | --- |
| 1, keyed by epoch | frequent-word subsampling | recreated per epoch |
| 2 | negative sampling | full training run |
| 3, keyed by epoch | context-window sampling | recreated per epoch |

The same epoch seed is replayable, while different epochs receive independent
subsampling/window streams. Extra negative-sampling draws cannot perturb positive
pair generation.

Python, legacy NumPy, and PyTorch global RNGs are still seeded for model
initialization and compatibility. A fixed seed does not promise bit-for-bit
identity across PyTorch versions, devices, or kernels.

## Drift detection

The vocabulary pass records both the expected in-vocabulary token mass and an
incremental SHA-256 fingerprint of the exact tokenized sentence stream. Every
training epoch recomputes both while streaming. Count mismatches can fail early;
the final fingerprint also catches equal-count substitutions, reordering, OOV
changes, and tokenizer-output changes without storing the corpus.

## Run manifest

Every successful run writes `run_manifest.json` beside the vectors. It records:

- the fully resolved dataclass configuration;
- the token-progress training plan;
- RNG stream assignments;
- corpus-pass count, streaming memory mode, and corpus SHA-256;
- per-epoch source, retained-token, example, and optimizer-step counters;
- peak buffered examples;
- selected device and final LR.

This artifact makes ablation runs inspectable without reconstructing behavior
from logs.
