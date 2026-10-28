"""Validating, deterministic, resource-bounded paper ablation planning.

An OFAT suite measures one factor at a time against a fixed control.
A factorial suite measures interactions. Both use the normal training Cfg.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import Cfg, _from_dict, _to_dict, load_cfg
from ..registry import Registry

DESIGN_REG = Registry()
SOURCE_TYPES = {"original_1301", "followup_1310", "engineering"}
IDENTIFIER = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Case:
    case_id: str
    suite: str
    arm: str
    seed: int
    source: str
    hypothesis: str
    config_hash: str
    cfg: Cfg


@dataclass(frozen=True)
class Suite:
    path: Path
    base_cfg: Cfg
    out_root: Path
    seeds: tuple[int, ...]
    specifications: tuple[dict[str, Any], ...]
    eval_file: Path | None
    max_cases: int


def _config_identity(cfg: Cfg) -> str:
    data = _to_dict(cfg)
    data["RUN"]["out_dir"] = "<case-output>"
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _resolve_path(value: str, suite_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Existing repository grid conventions interpret paths relative to CWD.
    candidate = (Path.cwd() / path).resolve()
    return candidate if candidate.exists() else (suite_path.parent / path).resolve()


def _set_path(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Unknown override path: {key}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ValueError(f"Unknown override path: {key}")
    if key in {"RUN.out_dir", "RUN.resume_from", "SEED"}:
        raise ValueError(f"Managed field cannot be overridden by an axis: {key}")
    node[parts[-1]] = value


@DESIGN_REG.register("one_factor")
def _one_factor(axes: dict[str, list[Any]], baseline: dict[str, Any]):
    yield "control", {}
    for field, options in axes.items():
        baseline_value = _lookup(baseline, field)
        for index, value in enumerate(options):
            if value != baseline_value:
                yield f"{field.replace('.', '_')}-{index}", {field: value}


@DESIGN_REG.register("factorial")
def _factorial(axes: dict[str, list[Any]], baseline: dict[str, Any]):
    keys = tuple(axes)
    for index, combo in enumerate(itertools.product(*(axes[k] for k in keys))):
        yield f"cell-{index:03d}", dict(zip(keys, combo, strict=True))


def _lookup(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Unknown axis: {path}")
        node = node[part]
    return node


def load_suite(path: str | Path) -> Suite:
    suite_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Suite YAML must contain a mapping")
    unknown = set(raw) - {"base_cfg", "out_root", "seeds", "max_cases", "eval_file", "suites"}
    if unknown:
        raise ValueError(f"Unknown suite fields: {sorted(unknown)}")
    if not isinstance(raw.get("base_cfg"), str):
        raise ValueError("base_cfg must be a YAML file path")
    base_cfg = load_cfg(str(_resolve_path(raw["base_cfg"], suite_path)))
    if not isinstance(raw.get("suites"), list) or not raw["suites"]:
        raise ValueError("suites must be a non-empty list")
    seeds = raw.get("seeds", [42, 43, 44])
    if not isinstance(seeds, list) or not seeds or any(type(x) is not int for x in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated seed in suite")
    cap = raw.get("max_cases", 128)
    if type(cap) is not int or cap < 1:
        raise ValueError("max_cases must be a positive integer")
    out_root = Path(raw.get("out_root", "runs/paper-ablations")).expanduser()
    if not out_root.is_absolute():
        out_root = (Path.cwd() / out_root).resolve()
    eval_file = raw.get("eval_file")
    if eval_file is not None and not isinstance(eval_file, str):
        raise ValueError("eval_file must be a file path or null")
    eval_path = _resolve_path(eval_file, suite_path) if eval_file else None
    return Suite(suite_path, base_cfg, out_root, tuple(seeds), tuple(raw["suites"]), eval_path, cap)


def build_plan(suite: Suite) -> list[Case]:
    cases: list[Case] = []
    seen_ids: set[str] = set()
    base = _to_dict(suite.base_cfg)
    for spec in suite.specifications:
        if not isinstance(spec, dict):
            raise ValueError("Each suite must be a mapping")
        if set(spec) - {"id", "source", "hypothesis", "design", "axes", "fixed"}:
            raise ValueError(f"Unknown experiment properties: {set(spec)}")
        name = spec.get("id")
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or name in seen_ids:
            raise ValueError(f"Invalid/duplicate suite id: {name!r}")
        seen_ids.add(name)
        source = spec.get("source")
        if source not in SOURCE_TYPES:
            raise ValueError(f"Suite {name}: invalid paper provenance {source!r}")
        hypothesis = spec.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"Suite {name}: a falsifiable hypothesis is required")
        fixed = spec.get("fixed", {})
        if not isinstance(fixed, dict):
            raise ValueError("fixed must be a mapping of dotted paths")
        baseline = deepcopy(base)
        for path, value in fixed.items():
            _set_path(baseline, path, value)
        axes = spec.get("axes")
        if not isinstance(axes, dict) or not axes:
            raise ValueError(f"Suite {name}: axes cannot be empty")
        for field, options in axes.items():
            _lookup(baseline, field)
            if field in {"SEED", "RUN.out_dir", "RUN.resume_from"}:
                raise ValueError(f"Managed axis: {field}")
            if not isinstance(options, list) or not options:
                raise ValueError(f"Axis {field} must be a non-empty list")
            # Reject redundant cells rather than counting duplicate runs as replications.
            if len({json.dumps(v, sort_keys=True) for v in options}) != len(options):
                raise ValueError(f"Duplicate values for axis {field}")
        design = spec.get("design", "one_factor")
        factory = DESIGN_REG.get(design)
        for arm, overrides in factory(axes, baseline):
            config_data = deepcopy(baseline)
            for path, value in overrides.items():
                _set_path(config_data, path, value)
            for seed in suite.seeds:
                cfg_data = deepcopy(config_data)
                cfg_data["SEED"] = seed
                if suite.eval_file is not None:
                    cfg_data["EVAL"]["analogy_file"] = str(suite.eval_file)
                cfg = _from_dict(cfg_data)
                digest = _config_identity(cfg)
                case_id = f"{name}__{arm}__s{seed}__{digest[:12]}"
                cfg.RUN.out_dir = str(suite.out_root / name / f"{arm}__s{seed}__{digest[:12]}")
                cases.append(Case(case_id, name, arm, seed, source, hypothesis, digest, cfg))
                if len(cases) > suite.max_cases:
                    raise ValueError(
                        f"Suite expands to more than max_cases={suite.max_cases}; "
                        "select smaller axes or explicitly increase your run budget"
                    )
    return cases
