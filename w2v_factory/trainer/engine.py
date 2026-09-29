from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ..data.dataset import (
    SentenceIndexer,
    generate_cbow_pairs_with_progress,
    generate_skipgram_pairs_with_progress,
)
from ..data.huffman import build_huffman_codes
from ..data.sampler import build_unigram_sampler
from ..data.subsample import compute_discard_probs
from ..data.text_reader import iter_tokens
from ..data.vocab import TokenStreamFingerprint, Vocab, build_vocab
from ..losses.hierarchical_softmax import pack_hs_batch
from ..losses.negative_sampling import draw_negatives
from ..models.cbow import CBOW
from ..models.skipgram import SkipGram
from ..trainer.exporter import save_json, save_numpy, save_word2vec_txt
from ..trainer.lr_schedulers import get_scheduler
from ..trainer.planning import PAIR_RNG_STREAM, TrainingPlan, build_training_plan, pair_rng
from ..trainer.streaming import StreamingEpochStats
from ..utils import make_numpy_rng, pick_device, set_seed

SUBSAMPLE_RNG_STREAM = 1
NEGATIVE_RNG_STREAM = 2


class Trainer:
    """Train CBOW or Skip-gram with bounded-memory streaming corpus passes."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        if cfg.TRAIN.batch_size <= 0 or cfg.TRAIN.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if cfg.TRAIN.lr <= 0:
            raise ValueError("learning rate must be positive")
        if cfg.TRAIN.lr_schedule not in {"linear", "none"}:
            raise ValueError(f"Unsupported lr_schedule: {cfg.TRAIN.lr_schedule}")
        if cfg.MODEL.window <= 0 or cfg.MODEL.dim <= 0:
            raise ValueError("window and embedding dimension must be positive")
        if cfg.MODEL.arch not in {"skipgram", "cbow"} or cfg.MODEL.loss not in {"ns", "hs"}:
            raise ValueError("Unsupported Word2Vec architecture or loss")
        if cfg.MODEL.loss == "ns" and cfg.MODEL.ns_neg_k <= 0:
            raise ValueError("ns_neg_k must be positive")

        self.device = pick_device(cfg.TRAIN.device)
        set_seed(cfg.SEED)
        self.writer: SummaryWriter | None = None
        if cfg.RUN.tb:
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.RUN.out_dir, "tb"))

        logger.info("Building vocabulary from streaming corpus pass...")
        token_stream = iter_tokens(
            cfg.DATA.input_files,
            cfg.DATA.lowercase,
            cfg.DATA.tokenizer,
            cfg.DATA.max_sent_len,
        )
        self.vocab: Vocab = build_vocab(token_stream, cfg.DATA.min_count, cfg.DATA.max_vocab)
        if self.vocab.size < 2:
            raise ValueError("Training requires at least two vocabulary words; check corpus and min_count")
        self.training_plan: TrainingPlan = build_training_plan(self.vocab.counts, cfg.TRAIN.epochs)
        logger.info(
            "Vocab size={}, raw_tokens={}, trainable_tokens_per_epoch={}",
            self.vocab.size,
            self.vocab.total_tokens,
            self.training_plan.trainable_tokens_per_epoch,
        )

        self.discard_probs = compute_discard_probs(
            self.vocab.counts,
            self.vocab.total_tokens,
            cfg.DATA.subsample_t,
        )

        self.hs_paths: list[list[int]] | None = None
        self.hs_codes: list[list[int]] | None = None
        self.neg_sampler = None
        if cfg.MODEL.loss == "hs":
            self.hs_paths, self.hs_codes = build_huffman_codes(self.vocab.counts)
            out_nodes = self.vocab.size - 1
        else:
            self.neg_sampler = build_unigram_sampler(
                self.vocab.counts,
                rng=make_numpy_rng(cfg.SEED, NEGATIVE_RNG_STREAM),
            )
            out_nodes = self.vocab.size

        model_type = SkipGram if cfg.MODEL.arch == "skipgram" else CBOW
        self.model = model_type(
            vocab_size=self.vocab.size,
            dim=cfg.MODEL.dim,
            share_io=cfg.MODEL.share_input_output,
            out_vocab_size=out_nodes,
        ).to(self.device)

        if cfg.TRAIN.optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=cfg.TRAIN.lr)
        elif cfg.TRAIN.optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.TRAIN.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.TRAIN.optimizer}")

        self.sched = get_scheduler(
            self.optim,
            cfg.TRAIN.lr_schedule,
            self.training_plan.total_progress_tokens,
        )
        self.epoch_stats: list[StreamingEpochStats] = []
        self.completed_steps = 0

    def _train_batch(
        self,
        payloads: list[Any],
        step: int,
        progress_before_batch: int,
    ) -> None:
        arch = self.cfg.MODEL.arch
        loss = self.cfg.MODEL.loss

        if self.sched is not None:
            self.sched.set_progress(progress_before_batch)
        lr_used = float(self.optim.param_groups[0]["lr"])

        if arch == "skipgram":
            centers, targets = zip(*payloads)
            inputs = torch.tensor(centers, dtype=torch.long, device=self.device)
            target_ids = torch.tensor(targets, dtype=torch.long, device=self.device)
            lengths = None
        else:
            contexts, targets = zip(*payloads)
            lengths = torch.tensor(
                [len(ctx) for ctx in contexts],
                dtype=torch.long,
                device=self.device,
            )
            context_ids = np.zeros((len(payloads), max(map(len, contexts))), dtype=np.int64)
            for row, ctx in enumerate(contexts):
                context_ids[row, : len(ctx)] = ctx
            inputs = torch.from_numpy(context_ids).to(self.device)
            target_ids = torch.tensor(targets, dtype=torch.long, device=self.device)

        if loss == "hs":
            assert self.hs_paths is not None and self.hs_codes is not None
            paths, codes, path_lens = pack_hs_batch(
                target_ids.cpu(),
                self.hs_paths,
                self.hs_codes,
            )
            paths, codes, path_lens = (
                paths.to(self.device),
                codes.to(self.device),
                path_lens.to(self.device),
            )
            if arch == "skipgram":
                result = self.model.forward_hs(inputs, paths, codes, path_lens)
            else:
                assert lengths is not None
                result = self.model.forward_hs(inputs, lengths, paths, codes, path_lens)
        else:
            assert self.neg_sampler is not None
            negatives = draw_negatives(
                self.neg_sampler,
                len(payloads),
                self.cfg.MODEL.ns_neg_k,
                forbid=target_ids,
            ).to(self.device)
            if arch == "skipgram":
                result = self.model.forward_ns(inputs, target_ids, negatives)
            else:
                assert lengths is not None
                result = self.model.forward_ns(inputs, lengths, target_ids, negatives)

        self.optim.zero_grad(set_to_none=True)
        result.loss.backward()
        self.optim.step()

        if step % 200 == 0:
            value = result.loss.item()
            logger.info(
                "[{}-{}][step={}] loss={:.4f} lr={:.6g} token_progress={}/{}",
                arch,
                loss,
                step,
                value,
                lr_used,
                progress_before_batch,
                self.training_plan.total_progress_tokens,
            )
            if self.writer is not None:
                self.writer.add_scalar("train/loss", value, step)
                self.writer.add_scalar("train/lr", lr_used, step)
                self.writer.add_scalar("train/token_progress", progress_before_batch, step)

    def _train_epoch(self, step0: int, epoch: int) -> tuple[int, StreamingEpochStats]:
        """Stream one corpus pass without retaining encoded sentences or pairs."""
        stats = StreamingEpochStats(epoch=epoch)
        indexer = SentenceIndexer(
            self.vocab.stoi,
            self.discard_probs,
            rng=make_numpy_rng(self.cfg.SEED, SUBSAMPLE_RNG_STREAM, epoch),
        )
        window_rng = pair_rng(self.cfg.SEED, epoch)
        if self.cfg.MODEL.arch == "skipgram":
            pair_fn = generate_skipgram_pairs_with_progress
        else:
            pair_fn = generate_cbow_pairs_with_progress

        epoch_offset = epoch * self.training_plan.trainable_tokens_per_epoch
        local_trainable_seen = 0
        fingerprint = TokenStreamFingerprint()
        step = step0
        batch: list[tuple[Any, int]] = []

        def flush_batch() -> None:
            nonlocal step
            if not batch:
                return
            progress_before = max(epoch_offset, batch[0][1] - 1)
            self._train_batch(
                [payload for payload, _ in batch],
                step,
                progress_before,
            )
            step += 1
            stats.optimizer_steps += 1
            batch.clear()

        for tokens in iter_tokens(
            self.cfg.DATA.input_files,
            self.cfg.DATA.lowercase,
            self.cfg.DATA.tokenizer,
            self.cfg.DATA.max_sent_len,
        ):
            fingerprint.update(tokens)
            stats.source_sentences += 1
            ids, positions, eligible = indexer.encode_with_positions(tokens)
            sentence_base = local_trainable_seen
            local_trainable_seen += eligible
            stats.trainable_tokens += eligible
            stats.retained_tokens += len(ids)
            if local_trainable_seen > self.training_plan.trainable_tokens_per_epoch:
                raise RuntimeError(
                    "Corpus/tokenizer drift detected after vocabulary build: "
                    "observed more in-vocabulary tokens than the training plan"
                )

            if len(ids) < 2:
                continue

            for payload, sentence_progress in pair_fn(
                ids,
                positions,
                self.cfg.MODEL.window,
                rng=window_rng,
            ):
                global_progress = epoch_offset + sentence_base + sentence_progress
                batch.append((payload, global_progress))
                stats.examples += 1
                stats.max_buffer_examples = max(stats.max_buffer_examples, len(batch))
                if len(batch) >= self.cfg.TRAIN.batch_size:
                    flush_batch()

        flush_batch()

        expected = self.training_plan.trainable_tokens_per_epoch
        if local_trainable_seen != expected:
            raise RuntimeError(
                "Corpus/tokenizer drift detected after vocabulary build: "
                f"expected {expected} in-vocabulary tokens, observed {local_trainable_seen}"
            )
        if fingerprint.hexdigest() != self.vocab.corpus_sha256:
            raise RuntimeError(
                "Corpus/tokenizer drift detected after vocabulary build: "
                "tokenized stream fingerprint changed"
            )

        if self.sched is not None:
            self.sched.set_progress((epoch + 1) * expected)
        stats.end_lr = float(self.optim.param_groups[0]["lr"])
        return step, stats

    def _write_manifest(self) -> None:
        actual_examples = sum(stats.examples for stats in self.epoch_stats)
        peak_buffer = max((stats.max_buffer_examples for stats in self.epoch_stats), default=0)
        manifest = {
            "config": asdict(self.cfg),
            "rng_streams": {
                "subsampling": {
                    "stream": SUBSAMPLE_RNG_STREAM,
                    "per_epoch": True,
                },
                "negative_sampling": {
                    "stream": NEGATIVE_RNG_STREAM,
                    "per_epoch": False,
                },
                "context_windows": {
                    "stream": PAIR_RNG_STREAM,
                    "per_epoch": True,
                },
            },
            "training_plan": asdict(self.training_plan),
            "runtime": {
                "memory_mode": "streaming",
                "corpus_passes": 1 + self.cfg.TRAIN.epochs,
                "device": str(self.device),
                "vocab_size": self.vocab.size,
                "raw_tokens": self.vocab.total_tokens,
                "source_sentences": self.vocab.sentences,
                "corpus_sha256": self.vocab.corpus_sha256,
                "actual_examples": actual_examples,
                "actual_optimizer_steps": self.completed_steps,
                "peak_buffer_examples": peak_buffer,
                "final_lr": float(self.optim.param_groups[0]["lr"]),
                "epoch_stats": [asdict(stats) for stats in self.epoch_stats],
            },
        }
        path = os.path.join(self.cfg.RUN.out_dir, "run_manifest.json")
        save_json(path, manifest)
        logger.info("Saved run manifest: {}", path)

    def train(self) -> None:
        """Train by rereading and subsampling the corpus independently each epoch."""
        try:
            step = 0
            for epoch in range(self.cfg.TRAIN.epochs):
                logger.info(
                    "Epoch {}/{}: streaming corpus pass",
                    epoch + 1,
                    self.cfg.TRAIN.epochs,
                )
                step, stats = self._train_epoch(step, epoch)
                self.epoch_stats.append(stats)
                logger.info(
                    "Epoch {}/{} complete: trainable_tokens={}, retained_tokens={}, "
                    "examples={}, optimizer_steps={}, peak_buffer={}",
                    epoch + 1,
                    self.cfg.TRAIN.epochs,
                    stats.trainable_tokens,
                    stats.retained_tokens,
                    stats.examples,
                    stats.optimizer_steps,
                    stats.max_buffer_examples,
                )

            if step <= 0:
                raise ValueError("No training examples remain after vocabulary filtering and subsampling")
            self.completed_steps = step

            embeddings = self.model.in_embed.weight.detach().cpu().numpy()
            txt_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.txt")
            npy_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.npy")
            save_word2vec_txt(txt_path, self.vocab.itos, embeddings)
            save_numpy(npy_path, embeddings)
            self._write_manifest()
            logger.info("Saved vectors: {}, {}", txt_path, npy_path)
        finally:
            if self.writer is not None:
                self.writer.close()
