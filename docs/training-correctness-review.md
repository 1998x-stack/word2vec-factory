# Training correctness review (2026-09-22)

## Scope and confirmed findings

1. **Skip-gram + HS target mismatch (fixed in this PR).** The old training loop discarded the context component of each `(center, context)` pair and packed the Huffman code for the center. Correct Skip-gram uses the center embedding as input and the **context word** as the prediction target. The regression test captures IDs passed into the HS packer.
2. **Negative-sampling false negatives (fixed in this PR).** A single redraw after a positive/negative collision does not ensure the redraw differs from the positive. Batched rejection sampling now retries until every row excludes its positive. A one-word vocabulary with a forbidden positive is rejected.
3. **HS Python-loop overhead (addressed 2026-09-29).** Skip-gram and CBOW HS now compute path scores with masked tensor operations. CBOW has a scalar-reference regression test that checks both loss and gradients.
4. **HS output-table memory (addressed 2026-09-29).** Huffman paths now use compact internal-node IDs, so HS allocates `V-1` output vectors instead of `2V-1`.
5. **Training-data memory (addressed 2026-09-29).** After the vocabulary pass, every epoch rereads the corpus line-by-line. Encoded sentences and positive pairs are no longer materialized across the corpus; only the current sentence and a bounded example batch are retained.
6. **LR progress accounting (addressed 2026-09-29).** The intermediate exact pair-step planner was replaced by source-token progress. Linear LR now has a deterministic denominator independent of randomized Skip-gram windows, subsampling outcomes, or negative-sampling retries.
7. **NumPy RNG coupling (addressed 2026-09-29).** Subsampling, negative sampling, and per-epoch context windows use independent deterministic streams. Subsampling is recreated per epoch, so epochs no longer reuse one frozen subsampled corpus.
8. **Corpus drift during training (addressed 2026-09-29).** Each streaming epoch recounts in-vocabulary source tokens and compares them with the vocabulary pass. A changed corpus/tokenizer fails explicitly instead of silently invalidating LR progress.

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

Future work should be isolated into independently tested PRs: peak-RSS benchmarking on large corpora; CPU/CUDA numerical reproducibility policy and experiments; checkpoint/resume with RNG/progress restoration; faster evaluation for large vocabularies. A fixed seed alone does not guarantee bit-for-bit identical results across PyTorch versions, platforms and CPU/GPU backends (see PyTorch's reproducibility notes).
