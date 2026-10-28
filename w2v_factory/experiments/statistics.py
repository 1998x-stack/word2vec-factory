"""Seed-level estimates for reproducible experiments; descriptive, not causal proof."""
from __future__ import annotations

import numpy as np

METRICS = (
    "accuracy_covered", "semantic_accuracy", "syntactic_accuracy",
    "accuracy_all", "coverage", "train_seconds", "tokens_per_second",
)


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean()), float(data.std(ddof=1)) if len(data) >= 2 else None


def _bootstrap_interval(values: list[float], *, seed: int = 7001, replicates: int = 4000):
    """Percentile CI of seed-level means; require >=3 independent matched seeds."""
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    sampled = rng.integers(0, len(data), size=(replicates, len(data)))
    means = data[sampled].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def aggregate_seed_results(rows: list[dict]) -> list[dict]:
    """Within each suite/arm summarize runs; for OFAT use same-seed control deltas.

    Cross-suite results are not pooled since they can use different corpora or
    experimental protocols. The bootstrap is exploratory for small n.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    control: dict[tuple[str, int], dict] = {}
    for row in rows:
        groups.setdefault((row["suite"], row["arm"]), []).append(row)
        if row["arm"] == "control":
            control[(row["suite"], row["seed"])] = row

    estimates = []
    for (suite, arm), records in sorted(groups.items()):
        measures = {}
        for metric in METRICS:
            values = [float(r[metric]) for r in records if r.get(metric) is not None]
            mean, sd = _mean_sd(values)
            paired = [
                float(r[metric]) - float(control[(suite, r["seed"])][metric])
                for r in records if
                (suite, r["seed"]) in control
                and r.get(metric) is not None
                and control[(suite, r["seed"])].get(metric) is not None
            ]
            delta_mean, delta_sd = _mean_sd(paired)
            measures[metric] = {
                "n": len(values), "mean": mean, "sd_sample": sd,
                "ci95_bootstrap": _bootstrap_interval(values),
                "paired_n": len(paired) if arm != "control" else 0,
                "paired_delta_mean": delta_mean if arm != "control" else None,
                "paired_delta_sd_sample": delta_sd if arm != "control" else None,
                "paired_delta_ci95_bootstrap": (
                    _bootstrap_interval(paired) if arm != "control" else None
                ),
            }
        estimates.append({
            "suite": suite,
            "arm": arm,
            "source": records[0]["source"],
            "seeds": sorted(r["seed"] for r in records),
            "metrics": measures,
        })
    return estimates
