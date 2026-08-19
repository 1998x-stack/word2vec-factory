from __future__ import annotations

import os

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
    """训练引擎：包含数据预处理、采样、训练循环、评测与导出。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = pick_device(cfg.TRAIN.device)
        set_seed(cfg.SEED)
        self.writer: SummaryWriter | None = None
        if cfg.RUN.tb:
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.RUN.out_dir, "tb"))

        # 读取与词表
        logger.info("Building vocabulary...")
        # 第一次迭代统计词频
        token_stream1 = iter_tokens(cfg.DATA.input_files, cfg.DATA.lowercase, cfg.DATA.tokenizer, cfg.DATA.max_sent_len)
        vocab = build_vocab(token_stream1, cfg.DATA.min_count, cfg.DATA.max_vocab)
        self.vocab: Vocab = vocab
        logger.info(f"Vocab size={vocab.size}, total_tokens={vocab.total_tokens}")

        # 次采样概率
        discard_probs = compute_discard_probs(vocab.counts, vocab.total_tokens, cfg.DATA.subsample_t)
        self.indexer = SentenceIndexer(vocab.stoi, discard_probs)

        # Huffman/负采样构件
        self.hs_paths: list[list[int]] | None = None
        self.hs_codes: list[list[int]] | None = None
        self.neg_sampler = None
        if cfg.MODEL.loss == "hs":
            logger.info("Building Huffman tree...")
            paths, codes = build_huffman_codes(vocab.counts)
            self.hs_paths, self.hs_codes = paths, codes
            out_nodes = 2 * vocab.size - 1  # HS 需要为内部节点留位置
        else:
            logger.info("Building unigram^0.75 sampler...")
            self.neg_sampler = build_unigram_sampler(vocab.counts)
            out_nodes = vocab.size

        # 模型
        if cfg.MODEL.arch == "skipgram":
            self.model = SkipGram(
                vocab_size=vocab.size,
                dim=cfg.MODEL.dim,
                share_io=cfg.MODEL.share_input_output,
                out_vocab_size=out_nodes,
            )
        else:
            self.model = CBOW(
                vocab_size=vocab.size,
                dim=cfg.MODEL.dim,
                share_io=cfg.MODEL.share_input_output,
                out_vocab_size=out_nodes,
            )
        self.model.to(self.device)

        # 优化器与调度
        if cfg.TRAIN.optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=cfg.TRAIN.lr)
        else:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.TRAIN.lr)
        self.total_steps = None  # 线性退火按 token 对数估计也可，这里按粗略 batch 估计
        self.sched = None

    def _train_epoch_skipgram_ns(self, sents: list[list[int]], step0: int) -> int:
        B = self.cfg.TRAIN.batch_size
        K = self.cfg.MODEL.ns_neg_k
        step = step0
        # 生成所有正样本（注意：大语料下建议在线生成 + 分块；此处教学实现）
        pos_pairs: list[tuple[int, int]] = []
        for toks in sents:
            pos_pairs.extend(generate_skipgram_pairs(toks, self.cfg.MODEL.window))
        logger.info(f"Pos pairs (epoch): {len(pos_pairs)}")
        # 打乱
        rng = np.random.default_rng(self.cfg.SEED + step0)
        rng.shuffle(pos_pairs)

        for i in range(0, len(pos_pairs), B):
            batch = pos_pairs[i : i + B]
            centers = torch.tensor([c for c, _ in batch], dtype=torch.long, device=self.device)
            pos_ctx = torch.tensor([o for _, o in batch], dtype=torch.long, device=self.device)
            neg_ctx = draw_negatives(self.neg_sampler, centers.shape[0], K, forbid=pos_ctx).to(self.device)
            out = self.model.forward_ns(centers, pos_ctx, neg_ctx)

            self.optim.zero_grad(set_to_none=True)
            out.loss.backward()
            self.optim.step()
            if self.sched:
                self.sched.step()

            if step % 200 == 0:
                logger.info(f"[NS][step={step}] loss={out.loss.item():.4f}")
                if self.writer:
                    self.writer.add_scalar("train/loss", out.loss.item(), step)
            step += 1
        return step

    def _train_epoch_skipgram_hs(self, sents: list[list[int]], step0: int) -> int:
        B = self.cfg.TRAIN.batch_size
        step = step0
        paths, codes = self.hs_paths, self.hs_codes
        assert paths is not None and codes is not None
        # 准备 (center → 路径)
        pos_words: list[int] = []
        for toks in sents:
            for c, _ in generate_skipgram_pairs(toks, self.cfg.MODEL.window):
                pos_words.append(c)
        logger.info(f"Centers (epoch): {len(pos_words)}")

        for i in range(0, len(pos_words), B):
            w = torch.tensor(pos_words[i : i + B], dtype=torch.long, device=self.device)
            p, cd, ln = pack_hs_batch(w.cpu(), paths, codes)
            p, cd, ln = p.to(self.device), cd.to(self.device), ln.to(self.device)
            out = self.model.forward_hs(w, p, cd, ln)

            self.optim.zero_grad(set_to_none=True)
            out.loss.backward()
            self.optim.step()
            if self.sched:
                self.sched.step()

            if step % 200 == 0:
                logger.info(f"[HS][step={step}] loss={out.loss.item():.4f}")
                if self.writer:
                    self.writer.add_scalar("train/loss", out.loss.item(), step)
            step += 1
        return step

    def _train_epoch_cbow_ns(self, sents: list[list[int]], step0: int) -> int:
        B = self.cfg.TRAIN.batch_size
        K = self.cfg.MODEL.ns_neg_k
        step = step0
        pairs: list[tuple[list[int], int]] = []
        for toks in sents:
            pairs.extend(generate_cbow_pairs(toks, self.cfg.MODEL.window))
        logger.info(f"CBOW pairs (epoch): {len(pairs)}")

        for i in range(0, len(pairs), B):
            batch = pairs[i : i + B]
            ctx_lens = [len(x) for x, _ in batch]
            Lmax = max(ctx_lens) if batch else 0
            ctx = np.zeros((len(batch), Lmax), dtype=np.int64)
            tgt = np.zeros((len(batch),), dtype=np.int64)
            for bi, (xs, y) in enumerate(batch):
                ctx[bi, : len(xs)] = xs
                tgt[bi] = y
            ctx_t = torch.from_numpy(ctx).to(self.device)
            lens_t = torch.tensor(ctx_lens, dtype=torch.long, device=self.device)
            tgt_t = torch.from_numpy(tgt).to(self.device)
            neg = draw_negatives(self.neg_sampler, ctx_t.shape[0], K, forbid=tgt_t).to(self.device)
            out = self.model.forward_ns(ctx_t, lens_t, tgt_t, neg)

            self.optim.zero_grad(set_to_none=True)
            out.loss.backward()
            self.optim.step()
            if self.sched:
                self.sched.step()

            if step % 200 == 0:
                logger.info(f"[CBOW-NS][step={step}] loss={out.loss.item():.4f}")
                if self.writer:
                    self.writer.add_scalar("train/loss", out.loss.item(), step)
            step += 1
        return step

    def _train_epoch_cbow_hs(self, sents: list[list[int]], step0: int) -> int:
        B = self.cfg.TRAIN.batch_size
        step = step0
        paths, codes = self.hs_paths, self.hs_codes
        assert paths is not None and codes is not None
        pairs: list[tuple[list[int], int]] = []
        for toks in sents:
            pairs.extend(generate_cbow_pairs(toks, self.cfg.MODEL.window))
        logger.info(f"CBOW pairs (epoch): {len(pairs)}")

        for i in range(0, len(pairs), B):
            batch = pairs[i : i + B]
            ctx_lens = [len(x) for x, _ in batch]
            Lmax_ctx = max(ctx_lens) if batch else 0
            ctx = np.zeros((len(batch), Lmax_ctx), dtype=np.int64)
            tgt = np.zeros((len(batch),), dtype=np.int64)
            for bi, (xs, y) in enumerate(batch):
                ctx[bi, : len(xs)] = xs
                tgt[bi] = y
            ctx_t = torch.from_numpy(ctx).to(self.device)
            lens_t = torch.tensor(ctx_lens, dtype=torch.long, device=self.device)
            tgt_t = torch.from_numpy(tgt).to(self.device)
            p, cd, ln = pack_hs_batch(tgt_t.cpu(), paths, codes)
            p, cd, ln = p.to(self.device), cd.to(self.device), ln.to(self.device)
            out = self.model.forward_hs(ctx_t, lens_t, p, cd, ln)

            self.optim.zero_grad(set_to_none=True)
            out.loss.backward()
            self.optim.step()
            if self.sched:
                self.sched.step()

            if step % 200 == 0:
                logger.info(f"[CBOW-HS][step={step}] loss={out.loss.item():.4f}")
                if self.writer:
                    self.writer.add_scalar("train/loss", out.loss.item(), step)
            step += 1
        return step

    def train(self) -> None:
        # 再次读取语料，这次进行编码+次采样
        logger.info("Encoding corpus with subsampling...")
        sents: list[list[int]] = []
        for toks in iter_tokens(
            self.cfg.DATA.input_files, self.cfg.DATA.lowercase, self.cfg.DATA.tokenizer, self.cfg.DATA.max_sent_len
        ):
            ids = self.indexer.encode(toks)
            if len(ids) >= 2:
                sents.append(ids)

        # 估算 total_steps 以便线性退火（粗略）：
        # 用样本数 / batch_size 近似
        approx_pairs = sum(max(0, len(s) - 1) * self.cfg.MODEL.window for s in sents)
        est_steps = int((approx_pairs / max(self.cfg.TRAIN.batch_size, 1)) * self.cfg.TRAIN.epochs)
        if self.cfg.TRAIN.lr_schedule == "linear":
            self.sched = get_scheduler(self.optim, "linear", est_steps)
        else:
            self.sched = None

        step = 0
        for ep in range(self.cfg.TRAIN.epochs):
            logger.info(f"Epoch {ep+1}/{self.cfg.TRAIN.epochs} ...")
            if self.cfg.MODEL.arch == "skipgram" and self.cfg.MODEL.loss == "ns":
                step = self._train_epoch_skipgram_ns(sents, step)
            elif self.cfg.MODEL.arch == "skipgram" and self.cfg.MODEL.loss == "hs":
                step = self._train_epoch_skipgram_hs(sents, step)
            elif self.cfg.MODEL.arch == "cbow" and self.cfg.MODEL.loss == "ns":
                step = self._train_epoch_cbow_ns(sents, step)
            else:
                step = self._train_epoch_cbow_hs(sents, step)

        # 导出向量
        emb = self.model.in_embed.weight.detach().cpu().numpy()
        txt_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.txt")
        npy_path = os.path.join(self.cfg.RUN.out_dir, "embeddings.npy")
        save_word2vec_txt(txt_path, self.vocab.itos, emb)
        save_numpy(npy_path, emb)
        logger.info(f"Saved vectors: {txt_path}, {npy_path}")
