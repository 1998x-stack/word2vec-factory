from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_FORMAT = "word2vec-factory"
CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> str:
    """Atomically save a checkpoint and a SHA-256 sidecar."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        with open(tmp_name, "rb+") as f:
            os.fsync(f.fileno())
        digest = _sha256_file(Path(tmp_name))
        os.replace(tmp_name, output)
        _atomic_write_text(output.with_suffix(output.suffix + ".sha256"), digest + "\n")
        return digest
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    """Load a trusted checkpoint after integrity validation."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointError(f"Checkpoint not found: {checkpoint}")

    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").strip()
        actual = _sha256_file(checkpoint)
        if expected != actual:
            raise CheckpointError(f"Checkpoint SHA-256 mismatch: {checkpoint}")

    try:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"Could not load checkpoint: {checkpoint}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("Unrecognized checkpoint format")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint version: {payload.get('version')!r}"
        )
    return payload


def training_signature(cfg) -> str:
    """Hash training semantics while excluding output/recovery controls."""
    data = asdict(cfg)
    run = data.pop("RUN", {})
    # These controls do not change model semantics.
    run.pop("out_dir", None)
    run.pop("tb", None)
    run.pop("checkpoint_every_epochs", None)
    run.pop("resume_from", None)
    if run:
        data["RUN"] = run
    data.pop("EVAL", None)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def capture_global_rng_state() -> dict[str, Any]:
    """Capture RNG state needed for deterministic continuation."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_global_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by capture_global_rng_state()."""
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy_legacy"])
        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state:
            if not torch.cuda.is_available():
                raise CheckpointError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
            cuda_states = state["torch_cuda"]
            if len(cuda_states) != torch.cuda.device_count():
                raise CheckpointError(
                    "CUDA device count differs from checkpoint RNG state"
                )
            torch.cuda.set_rng_state_all(cuda_states)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("Invalid RNG state in checkpoint") from exc
