# Training correctness review (2026-09-22)

## Scope and confirmed findings

1. **Skip-gram + HS target mismatch (fixed in this PR).** The old training loop discarded the context component of each `(center, context)` pair and packed the Huffman code for the center. Correct Skip-gram uses the center embedding as input and the **context word** as the prediction target. The regression test captures IDs passed into the HS packer.
2. **Negative-sampling false negatives (fixed in this PR).** A single redraw after a positive/negative collision does not ensure the redraw differs from the positive. Batched rejection sampling now retries until every row excludes its positive. A one-word vocabulary with a forbidden positive is rejected.
3. **HS Python-loop overhead (partially addressed).** Skip-gram HS computes all path scores in one masked tensor operation. CBOW HS still has a per-example loop; vectorizing it is a separate optimization, with numerical and gradient equivalence tests required.
4. **Training-data memory (partially addressed).** The old engine allocated all epoch pairs; the revised engine produces pairs lazily in batches. The encoded corpus is still materialized in RAM, so this is not a streaming-corpus implementation.

## Validation gates

- `python -m compileall -q w2v_factory tests`
- `python -m pytest -q` (including existing CBOW/Skip-gram x NS/HS smoke tests)
- Assert Skip-gram HS uses **context** Huffman targets while its model inputs remain centers.
- Assert repeated negative-sampling collisions are eliminated, including a collision on the first redraw.
- Compare vectorized HS loss and input/output gradients against a per-row scalar reference, including padded paths.
- Test empty/filtered corpus and one-word vocabulary error reporting.
- For model-quality regressions, compare the same corpus, seed, vocabulary, window, optimizer, loss, training steps, and evaluation protocol across implementations; report OOV and benchmark-coverage counts.

## Behavioral change and remaining work

Skip-gram NS previously globally shuffled the materialized epoch pairs. The bounded generator now emits pairs in corpus order with random local context windows. This reduces peak pair-list memory but changes the sample-order distribution. Do not attribute changes in final accuracy solely to the correctness fix without controlling this difference.

Future work should be isolated into independently tested PRs: true streaming corpus ingestion and memory benchmarking; exact step/token accounting for linear LR decay; config schema/relative INCLUDE path validation; vectorized CBOW HS; CPU/CUDA numerical reproducibility policy and experiments; checkpoint/resume and export metadata. A fixed seed alone does not guarantee bit-for-bit identical results across PyTorch versions, platforms and CPU/GPU backends (see PyTorch's reproducibility notes).
