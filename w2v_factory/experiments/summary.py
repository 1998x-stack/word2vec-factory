"""Durable experiment tables, optional charts and embedding projections."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect(root: str | Path) -> list[dict]:
    root = Path(root)
    result = []
    for path in sorted(root.glob("*/experiment.json")):
        item = _load_json(path)
        if item.get("status") != "completed":
            continue
        manifest = path.parent / "run_manifest.json"
        if not manifest.is_file():
            continue
        data = _load_json(manifest)
        ev = item.get("evaluation") or {}
        overall = ev.get("overall") or {}
        groups = ev.get("groups") or {}
        runtime = data.get("runtime", {})
        row = {
            "run_id": item["run_id"], "variant": item["variant"], "seed": item["seed"],
            "arch": data["config"]["MODEL"]["arch"],
            "loss": data["config"]["MODEL"]["loss"],
            "dim": data["config"]["MODEL"]["dim"],
            "window": data["config"]["MODEL"]["window"],
            "window_sampling": data["config"]["MODEL"].get("window_sampling", "uniform"),
            "pooling": data["config"]["MODEL"].get("cbow_pooling", "mean"),
            "subsample_t": data["config"]["DATA"].get("subsample_t"),
            "epochs": data["config"]["TRAIN"]["epochs"],
            "time_s": item.get("wall_time_s"),
            "raw_tokens": runtime.get("raw_tokens"),
            "examples": runtime.get("actual_examples"),
            "optimizer_steps": runtime.get("actual_optimizer_steps"),
            "vocab_size": runtime.get("vocab_size"),
            "accuracy": overall.get("accuracy"),
            "coverage": overall.get("coverage"),
            "eligible": overall.get("eligible"),
            "correct": overall.get("correct"),
            "semantic_accuracy": groups.get("semantic", {}).get("accuracy"),
            "syntactic_accuracy": groups.get("syntactic", {}).get("accuracy"),
        }
        if row["time_s"] and row["raw_tokens"] and row["epochs"]:
            # Inclusive of vocab-building overhead, not pure training throughput.
            row["tokens_per_s_end_to_end"] = row["raw_tokens"] * row["epochs"] / row["time_s"]
        else:
            row["tokens_per_s_end_to_end"] = None
        result.append(row)
    return result


def write_table(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    target = root / "summary.csv"
    if not rows:
        return target  # no synthetic results
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def generate_plots(root: str | Path) -> list[Path]:
    """Matplotlib is an optional dependency: pip install -e '.[viz]'."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(root)
    rows = collect(root)
    outputs = []
    quality = [r for r in rows if r["accuracy"] is not None and r["coverage"] is not None]
    if quality:
        by_variant = defaultdict(list)
        for row in quality:
            by_variant[row["variant"]].append(row)
        labels, means, sems = [], [], []
        for label, items in sorted(by_variant.items()):
            values = np.array([r["accuracy"] for r in items], dtype=float)
            labels.append(label)
            means.append(float(values.mean()))
            sems.append(float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(max(9, len(labels)*0.55), 5))
        ax.errorbar(range(len(labels)), means, yerr=sems, fmt="o", capsize=3)
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("3CosAdd top-1 (eligible questions only)")
        ax.set_title("Word2Vec ablations: mean ± SE across seeds; compare coverage")
        fig.tight_layout()
        dest = root / "analogy_accuracy.png"
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        outputs.append(dest)
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in quality:
            if row["tokens_per_s_end_to_end"] is not None:
                ax.scatter(row["tokens_per_s_end_to_end"], row["accuracy"], alpha=0.7)
        ax.set_xlabel("End-to-end training tokens/second (includes vocabulary pass)")
        ax.set_ylabel("3CosAdd accuracy on eligible questions")
        ax.set_title("Quality versus measured speed (fixed device required)")
        fig.tight_layout()
        dest = root / "quality_speed.png"
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        outputs.append(dest)
    # Original Table 2 interaction: vocabulary/training tokens × dimension.
    sizes = {r["raw_tokens"] for r in quality if r["raw_tokens"]}
    dims = {r["dim"] for r in quality}
    if len(sizes) > 1 and len(dims) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        for dim in sorted(dims):
            groups = defaultdict(list)
            for row in quality:
                if row["dim"] == dim and row["raw_tokens"] is not None:
                    groups[row["raw_tokens"]].append(float(row["accuracy"]))
            xs = sorted(groups)
            ys = [float(np.mean(groups[n])) for n in xs]
            se = [float(np.std(groups[n], ddof=1) / np.sqrt(len(groups[n])))
                  if len(groups[n]) > 1 else 0.0 for n in xs]
            ax.errorbar(xs, ys, yerr=se, marker="o", capsize=3, label=f"dim={dim}")
        ax.set_xscale("log")
        ax.set_xlabel("Train corpus tokens (nested prefixes; vocabulary may vary)")
        ax.set_ylabel("3CosAdd eligible accuracy")
        ax.set_title("Dimension × corpus-size scaling (mean ± seed SE)")
        ax.legend()
        fig.tight_layout()
        dest = root / "table2_dimension_tokens.png"
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        outputs.append(dest)
    return outputs


def project_embeddings(vectors: str | Path, output: str | Path, limit: int = 40) -> Path:
    """Small vocabulary diagnostic; no analogy accuracy inferred from PCA clusters."""
    if limit < 4 or limit > 200:
        raise ValueError("limit must be between 4 and 200")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    from ..cli.eval import load_word2vec_txt

    words, emb, _ = load_word2vec_txt(str(vectors))
    selected = min(limit, len(words))
    if selected < 4:
        raise ValueError("Projection needs at least four vocabulary words")
    xy = PCA(n_components=2).fit_transform(emb[:selected])
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(xy[:, 0], xy[:, 1], s=12)
    for i, word in enumerate(words[:selected]):
        ax.annotate(word, (xy[i, 0], xy[i, 1]), fontsize=8)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Exploratory projection; not a quantitative quality metric")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
