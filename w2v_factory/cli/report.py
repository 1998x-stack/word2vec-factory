"""Ablation report generation; plots require the optional viz extra."""
from __future__ import annotations

import argparse

from ..experiments.summary import generate_plots, project_embeddings, write_table


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize and visualize Word2Vec ablations")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--vectors", type=str, default=None, help="Optional embeddings.txt for PCA")
    args = p.parse_args()
    print(write_table(args.root))
    for path in generate_plots(args.root):
        print(path)
    if args.vectors:
        from pathlib import Path

        print(project_embeddings(args.vectors, Path(args.root) / "embedding_pca.png"))


if __name__ == "__main__":
    main()
