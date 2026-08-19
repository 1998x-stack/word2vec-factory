# Evaluation

Evaluated on the Google Analogy task (the shipped `questions-words.txt`):
`a : b :: c : d`, solved with 3CosAdd.

## Usage

```bash
python -m w2v_factory.cli.eval --vectors runs/exp1/embeddings.txt \
    --analogy questions-words.txt [--topk 1]
```

Or use the helper:

```bash
bash scripts/eval_analogy.sh [vectors.txt] [questions-words.txt]
```

## Metric

`evaluate_analogy` returns the fraction of correct analogies under 3CosAdd:
the query is `b - a + c`, candidate words are scored by cosine similarity,
excluding the `a`/`b`/`c` tokens. Only questions whose tokens are all in the
vocabulary count toward the denominator.

## File format

`embeddings.txt` is the standard `word2vec` text format:

```text
|V| dim
word x.xxxx x.xxxx ...
```

Loaded by `load_word2vec_txt` in `w2v_factory/cli/eval.py`.