# word2vec-factory

A pluggable, factory-mode Word2Vec implementation for **CBOW/Skip-gram** with
**Hierarchical Softmax (Huffman)** and **Negative Sampling**, designed for **industrial ablation**.

## Highlights
- Factory registry for models, losses, datasets, schedulers.
- Accurate **Huffman Hierarchical Softmax**.
- **Negative Sampling** with alias-table (unigram^0.75).
- Subsampling of frequent words, randomized context window.
- Linear LR decay, TensorBoard, Loguru, reproducible seeds.
- Google Analogy evaluation & batch ablations.

## Quickstart
```bash
pip install -e .
python -m w2v_factory.cli.train --cfg configs/skipgram_ns.yaml
python -m w2v_factory.cli.eval --vectors runs/exp1/embeddings.txt --analogy path/to/questions-words.txt
```
