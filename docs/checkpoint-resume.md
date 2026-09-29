# Checkpoint and resume

The trainer supports deterministic recovery at **completed epoch boundaries**.
It intentionally does not claim arbitrary mid-batch or mid-sentence resume.

## Why epoch boundaries

The streaming trainer recreates subsampling and context-window RNG streams from
`(SEED, stream_id, epoch)`. A completed epoch is therefore a natural
transaction boundary: the corpus fingerprint has been verified, the token
scheduler has advanced to the exact epoch boundary, and there is no partially
filled example batch to serialize.

This keeps recovery semantics explicit and testable.

## Configuration

```yaml
RUN:
  out_dir: runs/exp1
  checkpoint_every_epochs: 1   # null disables periodic checkpoints
  resume_from: null            # path to a trusted .pt checkpoint
```

The CLI can override these controls:

```bash
python -m w2v_factory.cli.train \
  --cfg configs/skipgram_ns.yaml \
  --checkpoint-every-epochs 1

python -m w2v_factory.cli.train \
  --cfg configs/skipgram_ns.yaml \
  --resume runs/exp1/checkpoints/epoch-0003.pt
```

Checkpoint cadence and output paths are recovery controls, not model semantics,
so they may change between the original and resumed process.

## Saved state

A checkpoint contains:

- format/version metadata;
- a SHA-256 hash of training-semantic configuration;
- corpus token-stream SHA-256 and vocabulary metadata;
- token-progress training plan;
- next epoch and completed optimizer steps;
- prior per-epoch streaming statistics;
- model and optimizer state;
- token-progress scheduler state;
- negative-sampling generator state when NS is active;
- Python, legacy NumPy, PyTorch CPU, and when applicable CUDA RNG state.

Per-epoch subsampling and context-window generators do not need mutable state in
the checkpoint because they are deterministically recreated from the root seed
and epoch number.

## Integrity and atomicity

Checkpoint files are written to a temporary file in the destination directory,
flushed, and atomically replaced into place. A `.sha256` sidecar is written for
every checkpoint and is required during restore.

A hash mismatch or unreadable payload fails before training resumes. The trainer
also rejects checkpoints whose configuration, vocabulary, corpus fingerprint, or
training plan differs from the current run.

## Trust boundary

PyTorch checkpoint loading uses Python deserialization. Only resume checkpoints
you created or otherwise trust. A SHA-256 sidecar detects accidental corruption;
it does not make an untrusted checkpoint safe.

## Deterministic recovery guarantee

CPU regression tests compare:

1. an uninterrupted two-epoch run; and
2. epoch 1, checkpoint, reconstruction of a new Trainer, restore, and epoch 2.

Model tensors are required to match exactly (`rtol=0, atol=0`), and the
scheduler progress, optimizer LR, completed-step count, and future
negative-sampler draws must also match.

Cross-version, cross-device, and nondeterministic CUDA kernels remain outside
that bit-for-bit guarantee.
