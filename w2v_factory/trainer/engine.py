from __future__ import annotations

import os
from dataclasses import asdict
from itertools import islice

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ..data.dataset import SentenceIndexer, generate_cbow_pairs, generate_skipgram_pairs
from ..data.huffman import build_huffman_codes
from ..data.sampler import build_unigram_sampler
from ..data.subsample import compute_discard_probs
from ..data.text_reader import iter_tokens
from ..data.vocab import Vocab, build_vocab
from ..losses.hierarchical_softmax import pack_hs_batch
from ..losses.negative_sampling import draw_negatives
from ..models.cbow import CBOW
from ..models.skipgram import SkipGram
from ..trainer.exporter import save_json, save_numpy, save_word2vec_txt
from ..trainer.lr_schedulers import get_scheduler
from ..trainer.planning import PAIR_RNG_STREAM, EpochPlan, build_epoch_plans, pair_rng
from ..utils import make_numpy_rng, pick_device, set_seed

SUBSAMPLE_RNG_STREAM = 1
NEGATIVE_RNG_STREAM = 2


class Trainer:
    """Train CBOW or Skip-gram with negative sampling or hierarchical softmax."""

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

        logger.info("Building vocabulary...")
        token_stream = iter_tokens(
            cfg.DATA.input_files, cfg.DATA.lowercase, cfg.DATA.tokenizer, cfg.DATA.max_sent_len
        )
        self.vocab: Vocab = build_vocab(token_stream, cfg.DATA.min_count, cfg.DATA.max_vocab)
        if self.vocab.size < 2:
            raise ValueError("Training requires at least two vocabulary words; check corpus and min_count")
        logger.info("Vocab size={}, total_tokens={}", self.vocab.size, self.vocab.total_tokens)

        discard_probs = compute_discard_probs(
            self.vocab.counts, self.vocab.total_tokens, cfg.DATA.subsample_t
        )
        self.indexer = SentenceIndexer(
            self.vocab.stoi,
            discard_probs,
            rng=make_numpy_rng(cfg.SEED, SUBSAMPLE_RNG_STREAM),
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

        self.sched = None
        self.epoch_plans: list[EpochPlan] = []
        self.completed_steps = 0

    def _train_epoch(self, sents: list[list[int]], step0: int, epoch: int = 0) -> int:
        """Train one epoch from a deterministic, bounded pair stream."""
        arch = self.cfg.MODEL.arch
        loss = self.cfg.MODEL.loss
        window = self.cfg.MODEL.window
        batch_size = self.cfg.TRAIN.batch_size
        pair_fn = generate_skipgram_pairs if arch == "skipgram" else generate_cbow_pairs
        window_rng = pair_rng(self.cfg.SEED, epoch)
        examples = (pair for sent in sents for pair in pair_fn(sent, window, rng=window_rng))
        step = step0

        while batch := list(islice(examples, batch_size)):
            if arch == "skipgram":
                centers, targets = zip(*batch)
                inputs = torch.tensor(centers, dtype=torch.long, device=self.device)
                target_ids = torch.tensor(targets, dtype=torch.long, device=self.device)
            else:
                contexts, targets = zip(*batch)
                lengths = torch.tensor([len(ctx) for ctx in contexts], dtype=torch.long, device=self.device)
                context_ids = np.zeros((len(batch), max(map(len, contexts))), dtype=np.int64)
                for row, ctx in enumerate(contexts):
                    context_ids[row, : len(ctx)] = ctx
                inputs = torch.from_numpy(context_ids).to(self.device)
                target_ids = torch.tensor(targets, dtype=torch.long, device=self.device)

            if loss == "hs":
                assert self.hs_paths is not None and self.hs_codes is not None
                paths, codes, path_lens = pack_hs_batch(
                    target_ids.cpu(), self.hs_paths, self.hs_codes
                )
                paths, codes, path_lens = (
                    paths.to(self.device), codes.to(self.device), path_lens.to(self.device)
                )
                if arch == "skipgram":
                    result = self.model.forward_hs(inputs, paths, codes, path_lens)
                else:
                    result = self.model.forward_hs(inputs, lengths, paths, codes, path_lens)
            else:
                assert self.neg_sampler is not None
                negatives = draw_negatives(
                    self.neg_sampler, len(batch), self.cfg.MODEL.ns_neg_k, forbid=target_ids
                ).to(self.device)
                if arch == "skipgram":
                    result = self.model.forward_ns(inputs, target_ids, negatives)
                else:
                    result = self.model.forward_ns(inputs, lengths, target_ids, negatives)

            lr_used = float(self.optim.param_groups[0]["lr"])
            self.optim.zero_grad(set_to_none=True)
            result.loss.backward()
            self.optim.step()
            if self.sched is not None:
                self.sched.step()

            if step % 200 == 0:
                value = result.loss.item()
                logger.info(
                    "[{}-{}][step={}] loss={:.4f} lr={:.6g}",
                    arch,
                    loss,
                    step,
                    value,
                    lr_used,
                )
                if self.writer is not None:
                    self.writer.add_scalar("train/loss", value, step)
                    self.writer.add_scalar("train/lr", lr_used, step)
            step += 1

        return step

    def _write_manifest(self, planned_steps: int) -> None:
        manifest = {
            "config": asdict(self.cfg),
            "rng_streams": {
                "subsampling": SUBSAMPLE_RNG_STREAM,
                "negative_sampling": NEGATIVE_RNG_STREAM,
                "context_windows": {
                    "stream": PAIR_RNG_STREAM,
                    "per_epoch": True,
                },
            },
            "runtime": {
                "device": str(self.device),
                "vocab_size": self.vocab.size,
                "total_tokens": self.vocab.total_tokens,
                "planned_optimizer_steps": planned_steps,
                "actual_optimizer_steps": self.completed_steps,
                "final_lr": float(self.optim.param_groups[0]["lr"]),
                "epoch_plans": [asdict(plan) for plan in self.epoch_plans],
            },
        }
        path = os.path.join(self.cfg.RUN.out_dir, "run_manifest.json")
        save_json(path, manifest)
        logger.info("Saved run manifest: {}", path)

    def train(self) -> None:
        """Encode once, then train against an exact deterministic epoch plan."""
        try:
            logger.info("Encoding corpus with subsampling...")
            sents = []
            for tokens in iter_tokens(
                self.cfg.DATA.input_files,
                self.cfg.DATA.lowercase,
                self.cfg.DATA.tokenizer,
                self.cfg.DATA.max_sent_len,
            ):
                ids = self.indexer.encode(tokens)
                if len(ids) >= 2:
                    sents.append(ids)
            if not sents:
                raise ValueError("No training pairs remain after vocabulary filtering and subsampling")

            self.epoch_plans = build_epoch_plans(
                sents=sents,
                arch=self.cfg.MODEL.arch,
                window=self.cfg.MODEL.window,
                batch_size=self.cfg.TRAIN.batch_size,
                epochs=self.cfg.TRAIN.epochs,
                seed=self.cfg.SEED,
            )
            planned_steps = sum(plan.optimizer_steps for plan in self.epoch_plans)
            planned_examples = sum(plan.examples for plan in self.epoch_plans)
            logger.info(
                "Training plan: epochs={}, examples={}, optimizer_steps={}",
                len(self.epoch_plans),
                planned_examples,
                planned_steps,
            )
            self.sched = get_scheduler(self.optim, self.cfg.TRAIN.lr_schedule, planned_steps)

            step = 0
            for plan in self.epoch_plans:
                logger.info(
                    "Epoch {}/{}: planned_examples={}, planned_steps={}",
                    plan.epoch + 1,
                    self.cfg.TRAIN.epochs,
                    plan.examples,
                    plan.optimizer_steps,
                )
                step_before = step
                step = self._train_epoch(sents, step, epoch=plan.epoch)
                actual_epoch_steps = step - step_before
                if actual_epoch_steps != plan.optimizer_steps:
                    raise RuntimeError(
                        f"Training plan drift in epoch {plan.epoch + 1}: "
                        f"planned {plan.optimizer_steps} steps, got {actual_epoch_steps}"
                    )

            if step != planned_steps:
                raise RuntimeError(f"Training plan drift: planned {planned_steps} steps, got {step}")
            self.completed_steps = step

            embeddings = self.model.in_embed.weight.detach().cpu().numpy()
            txt_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.txt")
            npy_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.npy")
            save_word2vec_txt(txt_path, self.vocab.itos, embeddings)
            save_numpy(npy_path, embeddings)
            self._write_manifest(planned_steps)
            logger.info("Saved vectors: {}, {}", txt_path, npy_path)
        finally:
            if self.writer is not None:
                self.writer.close()
