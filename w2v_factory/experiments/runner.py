"""Sequential experiment execution, structured events and reproducible artifacts."""
from __future__ import annotations

import csv
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from ..log import setup_logging
from ..trainer.engine import Trainer
from .evaluation import read_analogies, read_embedding_words, score_analogies
from .plan import Case


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_event(root: Path, *, event: str, case: Case, details: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "case_id": case.case_id,
        "source": case.source,
        "config_hash": case.config_hash,
        **(details or {}),
    }
    with (root / "events.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(data, sort_keys=True) + "\n")


def case_metadata(case: Case) -> dict:
    return {
        "case_id": case.case_id,
        "suite": case.suite,
        "arm": case.arm,
        "seed": case.seed,
        "source": case.source,
        "hypothesis": case.hypothesis,
        "config_hash": case.config_hash,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device_requested": case.cfg.TRAIN.device,
            "git_sha_if_ci": os.environ.get("GITHUB_SHA"),
        },
    }


def run_case(case: Case, root: Path, *, reuse: bool = True, eval_batch_size: int = 16) -> dict:
    out = Path(case.cfg.RUN.out_dir)
    result_path = out / "case_result.json"
    if reuse and result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if (saved.get("config_hash") == case.config_hash and saved.get("status") == "completed"
                and (out / "run_manifest.json").exists() and (out / "embeddings.npy").exists()):
            append_event(root, event="reused", case=case)
            return saved
        raise ValueError(f"Output exists but does not match completed case: {out}; inspect before re-running")
    if result_path.exists():
        raise ValueError(f"Existing incomplete output requires manual inspection: {out}")

    out.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out))
    write_json(out / "experiment_metadata.json", case_metadata(case))
    append_event(root, event="started", case=case)
    start = time.perf_counter()
    try:
        trainer = Trainer(case.cfg)
        trainer.train()
        elapsed_train = time.perf_counter() - start
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        runtime = manifest["runtime"]
        evaluation = None
        eval_start = time.perf_counter()
        if case.cfg.EVAL.analogy_file:
            questions = read_analogies(
                case.cfg.EVAL.analogy_file, lowercase=case.cfg.DATA.lowercase,
            )
            words = read_embedding_words(out / "embeddings.txt")
            vectors = np.load(out / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            evaluation = score_analogies(vectors, words, questions, batch_size=eval_batch_size)
            write_json(out / "analogy_metrics.json", evaluation)
        elapsed_eval = time.perf_counter() - eval_start
        result = {
            **{k: v for k, v in case_metadata(case).items() if k != "environment"},
            "status": "completed",
            "out_dir": str(out),
            "train_seconds": elapsed_train,
            "eval_seconds": elapsed_eval,
            "vocab_size": runtime["vocab_size"],
            "trainable_tokens": runtime["epoch_stats"][0]["trainable_tokens"] if runtime["epoch_stats"] else 0,
            "actual_examples": runtime["actual_examples"],
            "optimizer_steps": runtime["actual_optimizer_steps"],
            "tokens_per_second": (
                runtime["epoch_stats"][0]["trainable_tokens"] * case.cfg.TRAIN.epochs / elapsed_train
                if runtime["epoch_stats"] and elapsed_train > 0 else 0.0
            ),
            "examples_per_second": runtime["actual_examples"] / elapsed_train if elapsed_train > 0 else 0.0,
            "architecture": case.cfg.MODEL.arch,
            "objective": case.cfg.MODEL.loss,
            "dimension": case.cfg.MODEL.dim,
            "window": case.cfg.MODEL.window,
            "epochs": case.cfg.TRAIN.epochs,
            "eval": evaluation,
        }
        write_json(result_path, result)
        append_event(root, event="completed", case=case, details={
            "train_seconds": elapsed_train,
            "accuracy_covered": evaluation["overall"]["accuracy_covered"] if evaluation else None,
        })
        return result
    except Exception as exc:
        append_event(root, event="failed", case=case, details={
            "exception_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()[-6000:],
        })
        logger.exception("Ablation {} failed", case.case_id)
        raise


def _number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


def collect_results(cases: list[Case]) -> list[dict]:
    collected = []
    for case in cases:
        path = Path(case.cfg.RUN.out_dir) / "case_result.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("config_hash") == case.config_hash and result.get("status") == "completed":
                collected.append(result)
    return collected


def summarize_results(cases: list[Case], root: Path) -> list[dict]:
    results = collect_results(cases)
    columns = [
        "case_id", "suite", "arm", "seed", "source", "architecture", "objective",
        "dimension", "window", "epochs", "vocab_size", "trainable_tokens",
        "optimizer_steps", "train_seconds", "eval_seconds", "tokens_per_second",
        "coverage", "accuracy_covered", "accuracy_all", "semantic_accuracy",
        "syntactic_accuracy", "out_dir",
    ]
    rows = []
    for result in results:
        ev = result.get("eval") or {}
        overall = ev.get("overall") or {}
        families = ev.get("families") or {}
        rows.append({
            **{key: result.get(key) for key in columns},
            "coverage": overall.get("coverage"),
            "accuracy_covered": overall.get("accuracy_covered"),
            "accuracy_all": overall.get("accuracy_all"),
            "semantic_accuracy": (families.get("semantic") or {}).get("accuracy_covered"),
            "syntactic_accuracy": (families.get("syntactic") or {}).get("accuracy_covered"),
        })
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    control = {(r["suite"], r["seed"]): r for r in rows if r["arm"] == "control"}
    lines = [
        "# Paper ablation results", "",
        f"Completed {len(rows)}/{len(cases)} planned runs. No statistical significance is implied by a single seed.", "",
        "| Suite | Arm | Seed | Dim | Accuracy (covered) | Coverage | Train s | Tokens/s | Δ vs paired control |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        key = (row["suite"], row["seed"])
        baseline = control.get(key)
        delta = None
        if baseline and row["accuracy_covered"] is not None and baseline["accuracy_covered"] is not None:
            delta = row["accuracy_covered"] - baseline["accuracy_covered"]
        lines.append(
            f"| {row['suite']} | {row['arm']} | {row['seed']} | {row['dimension']} | "
            f"{_number(row['accuracy_covered'])} | {_number(row['coverage'])} | "
            f"{_number(row['train_seconds'])} | {_number(row['tokens_per_second'])} | "
            f"{_number(delta)} |"
        )
    lines.extend([
        "", "## Interpretation rules", "",
        "Accuracy on covered questions is conditional on four-word vocabulary membership. "
        "Coverage and accuracy over all questions must be reported alongside it.",
        "", "Elapsed training includes vocabulary construction, training, export and manifest writing. "
        "Throughput uses in-vocabulary source tokens; it is not pair classifications per second.",
        "", "A control delta is paired on seed and suite only; factorial designs have no OFAT control. "
        "For uncertainty use at least three seeds and compare matched-seed deltas.",
        "", "The 2013 paper's Google News/LDC data and DistBelief hardware are not bundled. "
        "The presence of an experiment here does not establish numerical reproduction.",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "summary.json", rows)
    return rows
