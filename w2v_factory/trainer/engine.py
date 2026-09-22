from __future__ import annotations

import os
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
from ..trainer.exporter import save_numpy, save_word2vec_txt
from ..trainer.lr_schedulers import get_scheduler
from ..utils import pick_device, set_seed


class Trainer:
    """Train CBOW or Skip-gram with negative sampling or hierarchical softmax."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        if cfg.TRAIN.batch_size <= 0 or cfg.TRAIN.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
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
        self.indexer = SentenceIndexer(self.vocab.stoi, discard_probs)
        self.hs_paths: list[list[int]] | None = None
        self.hs_codes: list[list[int]] | None = None
        self.neg_sampler = None
        if cfg.MODEL.loss == "hs":
            self.hs_paths, self.hs_codes = build_huffman_codes(self.vocab.counts)
            out_nodes = 2 * self.vocab.size - 1
        else:
            self.neg_sampler = build_unigram_sampler(self.vocab.counts)
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

    def _train_epoch(self, sents: list[list[int]], step0: int) -> int:
        """Build bounded batches of fresh pairs rather than materializing every pair."""
        arch = self.cfg.MODEL.arch
        loss = self.cfg.MODEL.loss
        window = self.cfg.MODEL.window
        batch_size = self.cfg.TRAIN.batch_size
        pair_fn = generate_skipgram_pairs if arch == "skipgram" else generate_cbow_pairs
        examples = (pair for sent in sents for pair in pair_fn(sent, window))
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
                # Always use the *predicted target* as the Huffman label. For
                # Skip-gram this is the context, never the input center word.
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

            self.optim.zero_grad(set_to_none=True)
            result.loss.backward()
            self.optim.step()
            if self.sched is not None:
                self.sched.step()
            if step % 200 == 0:
                value = result.loss.item()
                logger.info("[{}-{}][step={}] loss={:.4f}", arch, loss, step, value)
                if self.writer is not None:
                    self.writer.add_scalar("train/loss", value, step)
            step += 1
        return step

    def train(self) -> None:
        """Encode once; regenerate local context windows for each epoch."""
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

            approx_pairs = sum(max(0, len(s) - 1) * self.cfg.MODEL.window for s in sents)
            est_steps = max(1, int(
                approx_pairs * self.cfg.TRAIN.epochs / self.cfg.TRAIN.batch_size
            ))
            self.sched = get_scheduler(self.optim, self.cfg.TRAIN.lr_schedule, est_steps)

            step = 0
            for epoch in range(self.cfg.TRAIN.epochs):
                logger.info("Epoch {}/{}", epoch + 1, self.cfg.TRAIN.epochs)
                step = self._train_epoch(sents, step)

            embeddings = self.model.in_embed.weight.detach().cpu().numpy()
            txt_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.txt")
            npy_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.npy")
            save_word2vec_txt(txt_path, self.vocab.itos, embeddings)
            save_numpy(npy_path, embeddings)
            logger.info("Saved vectors: {}, {}", txt_path, npy_path)
        finally:
            if self.writer is not None:
                self.writer.close()
