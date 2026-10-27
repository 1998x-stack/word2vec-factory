"""End-to-end test of the actual paper experiment runner (not only the planner)."""
from __future__ import annotations

import json

from w2v_factory.config import Cfg, DataCfg, EvalCfg, ModelCfg, RunCfg, TrainCfg
from w2v_factory.experiments.plan import Case
from w2v_factory.experiments.runner import run_case, summarize_results


def test_experiment_case_train_evaluate_reuse_and_report(tmp_path):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text(("alpha beta gamma delta epsilon\n" * 30), encoding="utf-8")
    analogies = tmp_path / "analogies.txt"
    analogies.write_text(
        ": capital-world\nalpha beta gamma delta\nalpha beta unknown delta\n",
        encoding="utf-8",
    )
    out = tmp_path / "case"
    root = tmp_path / "suite"
    cfg = Cfg(
        SEED=3,
        RUN=RunCfg(out_dir=str(out), tb=False),
        DATA=DataCfg(
            input_files=[str(corpus)], lowercase=True,
            min_count=1, max_vocab=100, subsample_t=None,
        ),
        TRAIN=TrainCfg(
            epochs=1, batch_size=32, lr=0.025, lr_schedule="linear",
            optimizer="sgd", device="cpu",
        ),
        MODEL=ModelCfg(
            arch="cbow", dim=8, window=2, window_strategy="fixed",
            ns_neg_k=2, loss="hs",
        ),
        EVAL=EvalCfg(analogy_file=str(analogies)),
    )
    case = Case(
        case_id="tiny-test", suite="tiny", arm="control", seed=3,
        source="original_1301", hypothesis="End-to-end execution is reproducible.",
        config_hash="test-hash", cfg=cfg,
    )
    first = run_case(case, root)
    assert first["status"] == "completed"
    assert first["eval"]["overall"]["total"] == 2
    assert first["eval"]["overall"]["eligible"] == 1
    assert first["eval"]["overall"]["coverage"] == 0.5
    assert first["optimizer_steps"] > 0

    second = run_case(case, root, reuse=True)
    assert second == first
    assert (out / "run_manifest.json").exists()
    assert (out / "analogy_metrics.json").exists()
    assert json.loads((out / "experiment_metadata.json").read_text())["config_hash"] == "test-hash"
    summaries = summarize_results([case], root)
    assert len(summaries) == 1
    assert (root / "summary.csv").exists()
    assert (root / "summary.md").exists()
    assert (root / "aggregate.json").exists()
    events = [json.loads(s) for s in (root / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["started", "completed", "reused"]
