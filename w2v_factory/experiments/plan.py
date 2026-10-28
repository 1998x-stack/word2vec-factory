"""Deterministic, validated experiment factory.

Supports an OFAT design with paired seeds (causal interpretation conditional on
a common corpus and protocol), or a deliberate full-factorial grid.
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

EXPERIMENT_REG = Registry()


@dataclass
class Experiment:
    run_id: str
    variant: str
    seed: int
    changes: dict[str, Any]
    cfg: Cfg


def _assign(dct: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    if len(keys) != 2 or not all(keys):
        raise ValueError(f"Expected SECTION.field key, got {path!r}")
    section, field = keys
    if section not in dct or not isinstance(dct[section], dict) or field not in dct[section]:
        raise ValueError(f"Unknown experiment factor: {path}")
    dct[section][field] = value


def _resolve_base(grid_path: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    # Compatible with the old repository grid that starts with configs/base.yaml.
    if p.exists():
        return p
    candidate = grid_path.parent / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"base_cfg not found: {value}")


@EXPERIMENT_REG.register("one_factor")
def _one_factor(factors: dict[str, list[Any]], base: dict[str, Any]):
    yield "baseline", {}
    for key, values in factors.items():
        for value in values:
            current = deepcopy(base)
            _assign(current, key, value)
            if current != base:
                yield f"{key}={value}", {key: value}


@EXPERIMENT_REG.register("grid")
def _grid(factors: dict[str, list[Any]], base: dict[str, Any]):
    keys = list(factors)
    for vals in itertools.product(*(factors[k] for k in keys)):
        changes = dict(zip(keys, vals, strict=True))
        yield ",".join(f"{k}={v}" for k, v in changes.items()) or "baseline", changes


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip(".-")[:72] or "baseline"


def build_plan(path: str | Path, max_runs: int | None = None) -> tuple[list[Experiment], Path]:
    grid_path = Path(path).resolve()
    with grid_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a mapping")
    extra = set(raw) - {"base_cfg", "out_root", "strategy", "seeds", "factors", "grid", "profiles", "max_runs"}
    if extra:
        raise ValueError(f"Unknown experiment keys: {sorted(extra)}")
    base_path = _resolve_base(grid_path, raw["base_cfg"])
    base = _to_dict(load_cfg(str(base_path)))
    strategy = raw.get("strategy", "grid")
    factors = raw.get("factors", raw.get("grid", {}))
    if not isinstance(factors, dict):
        raise ValueError("factors/grid must be a mapping")
    for k, vals in factors.items():
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"Factor {k} must have a nonempty list of values")
        _assign(deepcopy(base), k, vals[0])  # fail before any expensive training
    seeds = raw.get("seeds", [base["SEED"]])
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(s, int) or isinstance(s, bool) for s in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be a mapping")
    out_root = Path(raw.get("out_root", "runs/ablations")).expanduser().resolve()
    budget = max_runs if max_runs is not None else raw.get("max_runs", 256)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("max_runs must be a positive integer")
    experiments: list[Experiment] = []
    designs = list(EXPERIMENT_REG.get(strategy)(factors, base))
    for label, changes in profiles.items():
        if not isinstance(label, str) or not isinstance(changes, dict):
            raise ValueError("profiles must map names to SECTION.field overrides")
        designs.append((f"profile:{label}", changes))
    if len(designs) * len(seeds) > budget:
        raise ValueError(f"Planned {len(designs)*len(seeds)} runs exceeds max_runs={budget}")
    seen: set[tuple[str, int]] = set()
    for variant, changes in designs:
        for seed in seeds:
            dct = deepcopy(base)
            for key, value in changes.items():
                _assign(dct, key, value)
            dct["SEED"] = seed
            # Config hash deliberately excludes destination directory.
            canonical = json.dumps(dct, sort_keys=True, ensure_ascii=False, default=str)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
            key = (digest, seed)
            if key in seen:
                continue
            seen.add(key)
            run_id = f"{_slug(variant)}__seed-{seed}__{digest}"
            cfg = _from_dict(dct)
            cfg.RUN.out_dir = str(out_root / run_id)
            cfg.RUN.resume_from = None
            experiments.append(Experiment(run_id, variant, seed, dict(changes), cfg))
    return experiments, out_root
