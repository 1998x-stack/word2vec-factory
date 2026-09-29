from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamingEpochStats:
    """Observable counters for one streaming corpus pass."""

    epoch: int
    source_sentences: int = 0
    trainable_tokens: int = 0
    retained_tokens: int = 0
    examples: int = 0
    optimizer_steps: int = 0
    max_buffer_examples: int = 0
    end_lr: float = 0.0
