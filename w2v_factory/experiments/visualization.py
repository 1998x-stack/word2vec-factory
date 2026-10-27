"""Optional publication-style plots generated from measured results, not demo data."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def render_charts(rows: list[dict], out_root: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Charts require pip install '.[viz]'") from exc

    charts = out_root / "plots"
    charts.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    measured = [r for r in rows if r.get("accuracy_covered") is not None]
    if measured:
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        categories = sorted({(r["architecture"], r["objective"]) for r in measured})
        for group in categories:
            subset = [r for r in measured if (r["architecture"], r["objective"]) == group]
            ax.scatter(
                [r["train_seconds"] for r in subset],
                [100 * r["accuracy_covered"] for r in subset],
                label="/".join(group),
                alpha=0.8,
            )
        ax.set(xlabel="End-to-end training wall time (s)", ylabel="Analogy accuracy on covered questions (%)",
               title="Word2Vec: quality vs measured cost")
        ax.legend()
        ax.grid(alpha=0.2)
        for suffix in ("png", "svg"):
            path = charts / f"quality_cost.{suffix}"
            fig.savefig(path, dpi=160)
            outputs.append(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        for group in categories:
            subset = [r for r in measured if (r["architecture"], r["objective"]) == group]
            by_dim: dict[int, list[float]] = defaultdict(list)
            for r in subset:
                by_dim[int(r["dimension"])].append(100 * r["accuracy_covered"])
            dims = sorted(by_dim)
            if dims:
                ax.plot(dims, [sum(by_dim[d]) / len(by_dim[d]) for d in dims], marker="o", label="/".join(group))
        ax.set(xlabel="Embedding dimension", ylabel="Accuracy, covered questions (%)",
               title="Dimension / architecture trends (descriptive means)")
        ax.legend()
        ax.grid(alpha=0.2)
        for suffix in ("png", "svg"):
            path = charts / f"dimension_accuracy.{suffix}"
            fig.savefig(path, dpi=160)
            outputs.append(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        labels = [r["case_id"][:38] for r in measured]
        ax.bar(range(len(measured)), [100 * r["coverage"] for r in measured])
        ax.set_xticks(range(len(measured)), labels, rotation=90)
        ax.set(ylabel="Four-word analogy vocabulary coverage (%)",
               title="Coverage must accompany covered-only accuracy", ylim=(0, 100))
        for suffix in ("png", "svg"):
            path = charts / f"coverage.{suffix}"
            fig.savefig(path, dpi=160)
            outputs.append(path)
        plt.close(fig)
    return outputs
