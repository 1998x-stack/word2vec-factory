from __future__ import annotations

import numpy as np
import pytest
import torch

from w2v_factory.config import Cfg, DataCfg, ModelCfg, RunCfg, TrainCfg
from w2v_factory.trainer.checkpoint import CheckpointError, load_checkpoint
from w2v_factory.trainer.engine import Trainer


CORPUS = (
    "alpha beta gamma delta epsilon\n"
    "beta gamma delta epsilon alpha\n"
    "gamma delta alpha beta epsilon\n"
) * 8


def _cfg(corpus, out_dir, *, resume_from=None, checkpoint_every_epochs=None):
    return Cfg(
        SEED=19,
        RUN=RunCfg(
            out_dir=str(out_dir),
            tb=False,
            checkpoint_every_epochs=checkpoint_every_epochs,
            resume_from=None if resume_from is None else str(resume_from),
        ),
        DATA=DataCfg(
            input_files=[str(corpus)],
            min_count=1,
            max_vocab=100,
            subsample_t=0.02,
            max_sent_len=100,
        ),
        TRAIN=TrainCfg(
            epochs=2,
            batch_size=7,
            lr=0.01,
            lr_schedule="linear",
            optimizer="adam",
            device="cpu",
        ),
        MODEL=ModelCfg(
            arch="skipgram",
            loss="ns",
            dim=6,
            window=3,
            ns_neg_k=3,
        ),
    )


def _clone_state(trainer):
    return {name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()}


def test_epoch_checkpoint_resume_matches_uninterrupted_training_exactly(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")

    baseline = Trainer(_cfg(corpus, tmp_path / "baseline"))
    baseline.train()
    baseline_state = _clone_state(baseline)

    interrupted = Trainer(_cfg(corpus, tmp_path / "interrupted"))
    step, stats = interrupted._train_epoch(step0=0, epoch=0)
    interrupted.epoch_stats.append(stats)
    interrupted.completed_steps = step
    interrupted.next_epoch = 1
    checkpoint = interrupted._save_checkpoint(next_epoch=1)

    resumed = Trainer(
        _cfg(
            corpus,
            tmp_path / "resumed",
            resume_from=checkpoint,
            checkpoint_every_epochs=1,
        )
    )
    assert resumed.next_epoch == 1
    assert resumed.completed_steps == step
    resumed.train()

    for name, expected in baseline_state.items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )

    assert resumed.completed_steps == baseline.completed_steps
    assert resumed.sched is not None and baseline.sched is not None
    assert resumed.sched.progress == baseline.sched.progress
    assert resumed.optim.param_groups[0]["lr"] == baseline.optim.param_groups[0]["lr"]

    assert resumed.neg_sampler is not None and baseline.neg_sampler is not None
    np.testing.assert_array_equal(
        resumed.neg_sampler.sample(100),
        baseline.neg_sampler.sample(100),
    )


def test_checkpoint_cadence_writes_integrity_sidecars(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    cfg = _cfg(corpus, tmp_path / "run", checkpoint_every_epochs=1)
    trainer = Trainer(cfg)
    trainer.train()

    for epoch in (1, 2):
        path = tmp_path / "run" / "checkpoints" / f"epoch-{epoch:04d}.pt"
        assert path.exists()
        assert path.with_suffix(".pt.sha256").exists()
        payload = load_checkpoint(path, map_location="cpu")
        assert payload["next_epoch"] == epoch


def test_corrupt_checkpoint_is_rejected_before_deserialization(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    trainer = Trainer(_cfg(corpus, tmp_path / "run"))
    step, stats = trainer._train_epoch(step0=0, epoch=0)
    trainer.epoch_stats.append(stats)
    trainer.completed_steps = step
    trainer.next_epoch = 1
    checkpoint = trainer._save_checkpoint(next_epoch=1)

    with checkpoint.open("ab") as f:
        f.write(b"corruption")

    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        load_checkpoint(checkpoint, map_location="cpu")


def test_resume_rejects_training_config_mismatch(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    trainer = Trainer(_cfg(corpus, tmp_path / "run"))
    step, stats = trainer._train_epoch(step0=0, epoch=0)
    trainer.epoch_stats.append(stats)
    trainer.completed_steps = step
    trainer.next_epoch = 1
    checkpoint = trainer._save_checkpoint(next_epoch=1)

    cfg = _cfg(corpus, tmp_path / "resume", resume_from=checkpoint)
    cfg.MODEL.window = 2
    with pytest.raises(CheckpointError, match="configuration"):
        Trainer(cfg)


def test_resume_rejects_equal_count_corpus_reordering(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    trainer = Trainer(_cfg(corpus, tmp_path / "run"))
    step, stats = trainer._train_epoch(step0=0, epoch=0)
    trainer.epoch_stats.append(stats)
    trainer.completed_steps = step
    trainer.next_epoch = 1
    checkpoint = trainer._save_checkpoint(next_epoch=1)

    lines = CORPUS.splitlines()
    corpus.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="corpus fingerprint"):
        Trainer(_cfg(corpus, tmp_path / "resume", resume_from=checkpoint))
