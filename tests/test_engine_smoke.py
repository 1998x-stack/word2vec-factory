import pytest

from w2v_factory.config import Cfg, DataCfg, EvalCfg, ModelCfg, RunCfg, TrainCfg
from w2v_factory.trainer.engine import Trainer

CORPUS = (
    "the quick brown fox jumps over the lazy dog\n"
    + "the quick brown fox jumps over the lazy dog\n"
    + "fox jumps over the dog\n"
    + "the dog sleeps on the mat\n"
    + "the dog is on the mat\n"
    + "fox and dog are quick animals\n"
) * 20


@pytest.mark.parametrize(
    "arch,loss",
    [("skipgram", "ns"), ("cbow", "ns"), ("skipgram", "hs"), ("cbow", "hs")],
)
def test_train_smoke(tmp_path, arch, loss):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS)

    out_dir = tmp_path / f"out_{arch}_{loss}"
    cfg = Cfg(
        SEED=0,
        RUN=RunCfg(out_dir=str(out_dir)),
        DATA=DataCfg(
            input_files=[str(corpus)],
            lowercase=True,
            min_count=2,
            max_vocab=10000,
            subsample_t=None,
            max_sent_len=1000,
        ),
        TRAIN=TrainCfg(epochs=1, batch_size=16, lr=0.01, lr_schedule="none", optimizer="sgd", device="cpu"),
        MODEL=ModelCfg(arch=arch, dim=8, window=2, ns_neg_k=2, loss=loss, share_input_output=False),
        EVAL=EvalCfg(),
    )
    Trainer(cfg).train()

    assert (out_dir / "embeddings.txt").exists()
    assert (out_dir / "embeddings.npy").exists()
