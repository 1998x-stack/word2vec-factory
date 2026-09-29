import numpy as np
import pytest

from w2v_factory.config import Cfg, DataCfg, ModelCfg, RunCfg, TrainCfg
from w2v_factory.data.dataset import SentenceIndexer
from w2v_factory.trainer import engine
from w2v_factory.trainer.engine import SUBSAMPLE_RNG_STREAM, Trainer
from w2v_factory.utils import make_numpy_rng


def _tiny_cfg(corpus, out_dir, *, epochs=2):
    return Cfg(
        SEED=13,
        RUN=RunCfg(out_dir=str(out_dir), tb=False),
        DATA=DataCfg(
            input_files=[str(corpus)],
            min_count=1,
            subsample_t=None,
            max_sent_len=100,
        ),
        TRAIN=TrainCfg(
            epochs=epochs,
            batch_size=4,
            lr=0.02,
            lr_schedule="linear",
            optimizer="sgd",
            device="cpu",
        ),
        MODEL=ModelCfg(
            arch="skipgram",
            loss="ns",
            dim=4,
            window=2,
            ns_neg_k=2,
        ),
    )


def test_indexer_progress_counts_discarded_but_not_oov_tokens():
    stoi = {"keep": 0, "drop": 1}
    discard = np.array([0.0, 1.0], dtype=np.float32)
    indexer = SentenceIndexer(stoi, discard, rng=np.random.default_rng(0))

    ids, positions, eligible = indexer.encode_with_positions(
        ["drop", "unknown", "keep"]
    )
    assert ids == [0]
    assert positions == [2]
    assert eligible == 2


def test_epoch_subsampling_stream_is_replayable_and_epoch_local():
    stoi = {"x": 0}
    discard = np.array([0.5], dtype=np.float32)
    sent = ["x"] * 200

    def retained(epoch):
        indexer = SentenceIndexer(
            stoi,
            discard,
            rng=make_numpy_rng(7, SUBSAMPLE_RNG_STREAM, epoch),
        )
        return indexer.encode_with_positions(sent)[:2]

    assert retained(0) == retained(0)
    assert retained(0) != retained(1)


def test_trainer_rereads_corpus_for_every_epoch(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma delta\n" * 4, encoding="utf-8")
    cfg = _tiny_cfg(corpus, tmp_path / "run", epochs=3)

    original = engine.iter_tokens
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(engine, "iter_tokens", counted)
    trainer = Trainer(cfg)
    trainer.train()

    assert calls == 1 + cfg.TRAIN.epochs
    assert len(trainer.epoch_stats) == cfg.TRAIN.epochs
    assert all(
        stats.max_buffer_examples <= cfg.TRAIN.batch_size
        for stats in trainer.epoch_stats
    )


def test_corpus_drift_after_vocab_build_is_rejected(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma\n" * 2, encoding="utf-8")
    cfg = _tiny_cfg(corpus, tmp_path / "run", epochs=1)
    trainer = Trainer(cfg)

    corpus.write_text("alpha beta gamma alpha beta\n" * 2, encoding="utf-8")
    with pytest.raises(RuntimeError, match="drift"):
        trainer._train_epoch(step0=0, epoch=0)
