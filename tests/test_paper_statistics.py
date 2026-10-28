import pytest

from w2v_factory.experiments.statistics import aggregate_seed_results


def test_matched_seed_deltas_and_uncertainty():
    rows = []
    for seed, baseline, improved in [(1, 0.2, 0.3), (2, 0.3, 0.4), (3, 0.4, 0.5)]:
        for arm, score in [("control", baseline), ("MODEL_dim-1", improved)]:
            rows.append({
                "suite": "dimension", "arm": arm, "seed": seed,
                "source": "original_1301", "accuracy_covered": score,
                "train_seconds": 10.0,
            })
    items = aggregate_seed_results(rows)
    result = next(x for x in items if x["arm"] == "MODEL_dim-1")
    acc = result["metrics"]["accuracy_covered"]
    assert acc["n"] == 3
    assert acc["mean"] == pytest.approx(0.4)
    assert acc["paired_n"] == 3
    assert acc["paired_delta_mean"] == pytest.approx(0.1)
    assert acc["paired_delta_ci95_bootstrap"] is not None
    assert "ci95_bootstrap" in acc


def test_single_run_does_not_invent_confidence_interval():
    x = aggregate_seed_results([{
        "suite": "q", "arm": "control", "seed": 42,
        "source": "engineering", "accuracy_covered": None,
        "train_seconds": 1.0,
    }])[0]
    assert x["metrics"]["train_seconds"]["ci95_bootstrap"] is None
    assert x["metrics"]["accuracy_covered"]["mean"] is None
