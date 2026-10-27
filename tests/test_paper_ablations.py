from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from w2v_factory.experiments.evaluation import read_analogies, score_analogies
from w2v_factory.experiments.plan import build_plan, load_suite


def _suite(tmp_path: Path, *, design: str = "one_factor", cases: int = 15) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text("MODEL:\n  dim: 8\n  loss: hs\n  arch: skipgram\n", encoding="utf-8")
    spec = tmp_path / "suite.yaml"
    spec.write_text(
        f"""base_cfg: {base}
out_root: {tmp_path}/out
seeds: [1, 2]
max_cases: {cases}
suites:
  - id: foo
    source: original_1301
    hypothesis: A higher dimension changes the quality-to-cost relationship.
    design: {design}
    axes:
      MODEL.arch: [skipgram, cbow]
      MODEL.dim: [8, 16]
""",
        encoding="utf-8",
    )
    return spec


def test_ofat_plan_deterministic_and_paired(tmp_path):
    suite = load_suite(_suite(tmp_path))
    cases = build_plan(suite)
    assert len(cases) == 6
    assert len({c.case_id for c in cases}) == 6
    assert [(x.arm, x.seed) for x in cases] == [
        ("control", 1), ("control", 2),
        ("MODEL_arch-1", 1), ("MODEL_arch-1", 2),
        ("MODEL_dim-1", 1), ("MODEL_dim-1", 2),
    ]
    assert cases[0].cfg.MODEL.dim == 8
    assert cases[4].cfg.MODEL.dim == 16
    assert cases[4].cfg.MODEL.arch == "skipgram"
    assert [c.case_id for c in build_plan(suite)] == [c.case_id for c in cases]


def test_factorial_and_budget(tmp_path):
    suite = load_suite(_suite(tmp_path, design="factorial"))
    assert len(build_plan(suite)) == 8
    with pytest.raises(ValueError, match="max_cases"):
        build_plan(load_suite(_suite(tmp_path, design="factorial", cases=7)))


def test_section_coverage_and_target_exclusion(tmp_path):
    path = tmp_path / "analogies.txt"
    path.write_text(
        ": capital-world\nA B C D\nA B MISSING D\n"
        ": gram7-past-tense\nA B C D\n",
        encoding="utf-8",
    )
    words = ["a", "b", "c", "d", "noise"]
    # The intended target is closest to b-a+c; a,b,c must be excluded.
    vectors = np.asarray([[1, 0], [0, 1], [1, 1], [-0.2, 1], [-1, -1]], dtype=np.float32)
    questions = read_analogies(path, lowercase=True)
    scores = score_analogies(vectors, words, questions, batch_size=2)
    assert scores["overall"]["total"] == 3
    assert scores["overall"]["eligible"] == 2
    assert scores["overall"]["correct"] == 2
    assert scores["overall"]["coverage"] == pytest.approx(2 / 3)
    assert scores["overall"]["accuracy_all"] == pytest.approx(2 / 3)
    assert scores["families"]["semantic"]["eligible"] == 1
    assert scores["families"]["syntactic"]["eligible"] == 1


def test_no_covered_questions_not_reported_as_zero_accuracy():
    scores = score_analogies(np.eye(2, dtype=np.float32), ["a", "b"], [])
    assert scores["overall"]["accuracy_covered"] is None


def test_smoke_yaml_plan_from_repo_root():
    suite = load_suite("configs/experiments/smoke.yaml")
    cases = build_plan(suite)
    assert len(cases) == 5
    assert all(c.cfg.RUN.out_dir.startswith(str(suite.out_root)) for c in cases)
