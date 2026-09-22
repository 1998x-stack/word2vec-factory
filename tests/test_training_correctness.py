"""Regression tests for mathematically correct Word2Vec training targets."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from w2v_factory.config import Cfg, DataCfg, ModelCfg, RunCfg, TrainCfg
from w2v_factory.losses.negative_sampling import draw_negatives
from w2v_factory.models.skipgram import SkipGram
from w2v_factory.trainer import engine


class RepeatedCollisionSampler:
    n = 3

    def __init__(self):
        self.calls = 0

    def sample(self, size):
        self.calls += 1
        # The second attempt is still a collision. One redraw is not enough.
        word = 1 if self.calls <= 2 else 2
        return np.full(size, word, dtype=np.int64)


def test_negative_sampling_retries_until_all_positives_are_excluded():
    sampler = RepeatedCollisionSampler()
    negatives = draw_negatives(sampler, B=2, K=4, forbid=torch.tensor([1, 1]))
    assert sampler.calls == 3
    assert negatives.shape == (2, 4)
    assert torch.all(negatives == 2)


def test_negative_sampling_rejects_impossible_vocab_and_invalid_shapes():
    sampler = RepeatedCollisionSampler()
    sampler.n = 1
    with pytest.raises(ValueError, match="one-word vocabulary"):
        draw_negatives(sampler, B=1, K=2, forbid=torch.tensor([0]))
    with pytest.raises(ValueError, match="shape"):
        draw_negatives(sampler, B=2, K=2, forbid=torch.tensor([0]))


def test_skipgram_hs_uses_context_word_paths(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma\n" * 3, encoding="utf-8")
    cfg = Cfg(
        RUN=RunCfg(out_dir=str(tmp_path / "run"), tb=False),
        DATA=DataCfg(input_files=[str(corpus)], min_count=1, subsample_t=None),
        TRAIN=TrainCfg(epochs=1, batch_size=16, device="cpu", lr_schedule="none"),
        MODEL=ModelCfg(arch="skipgram", loss="hs", dim=4, window=1),
    )
    trainer = engine.Trainer(cfg)
    packed_target_ids = []
    original = engine.pack_hs_batch

    def capture_target_ids(word_ids, paths, codes):
        packed_target_ids.extend(word_ids.tolist())
        return original(word_ids, paths, codes)

    monkeypatch.setattr(engine, "pack_hs_batch", capture_target_ids)
    trainer._train_epoch([[0, 1, 2]], step0=0)
    assert packed_target_ids == [1, 0, 2, 1]


def test_skipgram_hs_matches_masked_scalar_reference_and_gradients():
    torch.manual_seed(0)
    model = SkipGram(vocab_size=4, dim=3, out_vocab_size=7)
    centers = torch.tensor([0, 1, 2])
    paths = torch.tensor([[4, 5], [6, 0], [4, 6]])
    codes = torch.tensor([[0, 1], [1, 0], [1, 0]])
    lengths = torch.tensor([2, 1, 2])

    vector_loss = model.forward_hs(centers, paths, codes, lengths).loss
    reference_terms = []
    for i in range(len(centers)):
        center = model.in_embed(centers[i])
        nodes = model.out_embed(paths[i, : lengths[i]])
        signs = 2 * codes[i, : lengths[i]].float() - 1
        reference_terms.append(-F.logsigmoid(signs * (nodes * center).sum(dim=1)).sum())
    reference_loss = torch.stack(reference_terms).mean()
    torch.testing.assert_close(vector_loss, reference_loss)

    reference_grads = torch.autograd.grad(
        reference_loss, (model.in_embed.weight, model.out_embed.weight), retain_graph=True
    )
    vector_grads = torch.autograd.grad(vector_loss, (model.in_embed.weight, model.out_embed.weight))
    for actual, expected in zip(vector_grads, reference_grads):
        torch.testing.assert_close(actual, expected)
