"""Paper-aware experiment planner, executor and plotting entrypoint.

Default invocation plans only; add --run to train. Does not replace the
legacy cartesian grid CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from loguru import logger

from ..experiments.paper_plan import build_plan, load_suite
from ..experiments.runner import run_case, summarize_results, write_json
from ..experiments.visualization import render_charts


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Word2Vec paper ablations")
    parser.add_argument("--suite", required=True, help="configs/experiments/*.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true", help="Execute plan (default is dry-run)")
    modes.add_argument("--report-only", action="store_true", help="Rebuild report and charts only")
    parser.add_argument("--no-plots", action="store_true", help="Skip optional Matplotlib figures")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failures and continue")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    cases = build_plan(suite)
    suite.out_root.mkdir(parents=True, exist_ok=True)
    write_json(suite.out_root / "plan.json", [
        {
            "case_id": c.case_id, "suite": c.suite, "arm": c.arm, "seed": c.seed,
            "source": c.source, "hypothesis": c.hypothesis,
            "config_hash": c.config_hash, "cfg": asdict(c.cfg),
        } for c in cases
    ])
    logger.info("Planned {} cases under {}", len(cases), suite.out_root)
    if not args.run and not args.report_only:
        logger.info("Dry-run only. Inspect plan.json; pass --run to train.")
        return

    errors = 0
    if args.run:
        for pos, case in enumerate(cases, 1):
            logger.info("[{}/{}] {}", pos, len(cases), case.case_id)
            try:
                run_case(case, suite.out_root, eval_batch_size=args.eval_batch_size)
            except Exception:
                errors += 1
                if not args.continue_on_error:
                    break

    rows = summarize_results(cases, suite.out_root)
    if not args.no_plots and rows:
        try:
            charts = render_charts(rows, suite.out_root)
            logger.info("Wrote {} chart files", len(charts))
        except RuntimeError as exc:
            logger.warning("Plots omitted: {}", exc)
    logger.info("Completed {}/{} cases; {} errors this invocation", len(rows), len(cases), errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
