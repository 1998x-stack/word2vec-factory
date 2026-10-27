"""Reproducible ablation factory CLI: plan -> train -> evaluate -> summarize."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from ..experiments.evaluate import evaluate_file
from ..experiments.plan import Experiment, build_plan
from ..experiments.summary import write_table
from ..log import setup_logging
from ..trainer.engine import Trainer


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _event(root: Path, name: str, **values) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": name, **values,
        }, ensure_ascii=False, default=str) + "\n")


def _run(root: Path, exp: Experiment, force: bool = False) -> str:
    out = Path(exp.cfg.RUN.out_dir)
    state_path = out / "experiment.json"
    manifest = out / "run_manifest.json"
    config = asdict(exp.cfg)
    if manifest.exists() and state_path.exists() and not force:
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("config") != config:
            raise RuntimeError(f"Existing run config mismatch (refusing overwrite): {out}")
        if existing.get("status") == "completed":
            _event(root, "skip", run_id=exp.run_id, reason="already_completed")
            return "skipped"
    setup_logging(str(out))
    _event(root, "start", run_id=exp.run_id, variant=exp.variant, seed=exp.seed)
    state = {
        "run_id": exp.run_id, "variant": exp.variant, "seed": exp.seed,
        "changes": exp.changes, "status": "running",
    }
    _write_json(state_path, state)
    start = time.perf_counter()
    try:
        trainer = Trainer(exp.cfg)
        trainer.train()
        state["wall_time_s"] = time.perf_counter() - start
        if exp.cfg.EVAL.analogy_file:
            state["evaluation"] = evaluate_file(
                out / "embeddings.txt",
                exp.cfg.EVAL.analogy_file,
                lowercase=exp.cfg.DATA.lowercase,
            )
            _write_json(out / "evaluation.json", state["evaluation"])
        state["status"] = "completed"
        _write_json(state_path, state)
        _event(root, "completed", run_id=exp.run_id, wall_time_s=state["wall_time_s"])
        return "completed"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["wall_time_s"] = time.perf_counter() - start
        _write_json(state_path, state)
        _event(root, "failed", run_id=exp.run_id, error=state["error"])
        logger.exception("Experiment {} failed", exp.run_id)
        return "failed"


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-traceable Word2Vec experiment factory")
    p.add_argument("--grid", required=True)
    p.add_argument("--dry-run", action="store_true", help="Show the deterministic run plan; train nothing")
    p.add_argument("--max-runs", type=int, default=None, help="Fail fast if plan exceeds this many runs")
    p.add_argument("--only", type=str, default=None, help="Execute exactly one planned run_id")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--force", action="store_true", help="Explicitly replace completed artifacts")
    args = p.parse_args()
    runs, root = build_plan(args.grid, max_runs=args.max_runs)
    if args.only is not None:
        runs = [r for r in runs if r.run_id == args.only]
        if not runs:
            p.error(f"Unknown run_id: {args.only}")
    print(f"strategy plan: {len(runs)} distinct runs; root={root}")
    for exp in runs:
        print(f"{exp.run_id}  {exp.changes}")
    if args.dry_run:
        return
    failed = 0
    for exp in runs:
        status = _run(root, exp, force=args.force)
        if status == "failed":
            failed += 1
            if args.fail_fast:
                break
    write_table(root)
    if failed:
        raise SystemExit(f"{failed} experiment(s) failed; inspect events.jsonl and experiment.json")


if __name__ == "__main__":
    main()
