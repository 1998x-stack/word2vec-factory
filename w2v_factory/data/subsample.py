from __future__ import annotations

import numpy as np

Scalar = float | int | str | None


def _coerce_t(t: Scalar) -> float | None:
    """Coerce a subsampling threshold into a validated float or None.

    None, false-like strings, and numeric zero disable subsampling. Invalid,
    negative, non-finite, and boolean values fail fast so configuration mistakes
    cannot silently change the training distribution.
    """
    if t is None:
        return None
    if isinstance(t, bool):
        if not t:
            return None
        raise ValueError("subsample_t=true is invalid; use a positive number or null")

    value: float
    if isinstance(t, str):
        s = t.strip().lower()
        if s in {"", "null", "none", "false"}:
            return None
        try:
            value = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid subsample_t value: {t!r}") from exc
    elif isinstance(t, (float, int)):
        value = float(t)
    else:
        raise ValueError(f"Invalid subsample_t type: {type(t).__name__}")

    if value == 0:
        return None
    if not np.isfinite(value) or value < 0:
        raise ValueError("subsample_t must be a finite positive number or null")
    return value


def compute_discard_probs(counts: list[int], total_tokens: int, t: Scalar) -> np.ndarray:
    """计算 高频词 的丢弃概率（次采样）。

    Args:
        counts: 每个词的出现次数。
        total_tokens: 语料总 token 数。
        t: 次采样阈值（典型 1e-5）；None/0/false-like string 表示关闭。

    Returns:
        与词表同长的丢弃概率向量（float32）。
    """
    t_val = _coerce_t(t)
    n = len(counts)
    if t_val is None or n == 0 or total_tokens <= 0:
        return np.zeros(n, dtype=np.float32)

    freqs = np.asarray(counts, dtype=np.float64) / float(total_tokens)
    freqs = np.clip(freqs, 1e-12, None)  # 避免除零
    p_discard = 1.0 - np.sqrt(t_val / freqs)
    p_discard = np.clip(p_discard, 0.0, 1.0)
    return p_discard.astype(np.float32)
