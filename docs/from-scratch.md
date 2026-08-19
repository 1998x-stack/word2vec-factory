# From-scratch notes

Derivations for the algorithms implemented in this repo. Math here uses `v`
for the input/center vector and `u` for the output/context vector.

## Skip-gram objective

Predict the context words given a center word. For a positive pair
`(center, ctx)` the score is the dot product of the two vectors:

```
score(center, ctx) = v_center · u_ctx
```

## Negative sampling

Negative sampling (NS) approximates the logistic-binary objective over a
positive pair plus `K` negatives:

```
L = log σ(v·u) + Σ_k log σ(-v·u_k)
```

where `u_k` are drawn from a unigram^0.75 distribution via an alias table.

## Hierarchical softmax

A Huffman tree is built over the words; each leaf has a root-to-leaf path of
internal nodes. At each edge assign code `1` for right, `0` for left. The loss
from the path nodes `u_i` is:

```
L = -Σ_i log σ( (2·code_i - 1) · v · u_i )
```

The gradient only crosses the path's nodes, so the cost is O(path) per word.

## Subsampling

Frequent words are dropped with probability `p = 1 - √(t/f)`, clipped to
`[0, 1]`. This reduces their prevalence and speeds up training.

## Randomized window

Each positive uses a window sampled uniformly in `[1, window]` — the paper's
trick for robustness against different context sizes.

## Why an alias table

Building the unigram^0.75 table once and sampling in O(1) avoids a per-step
`O(V)` scan. `data/sampler.py` implements Vose's alias method.