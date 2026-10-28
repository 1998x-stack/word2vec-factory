# Architecture — Word2Vec Factory

本工程通过**单一训练器 + Registry Factory + 可审计实验层**避免分叉训练逻辑。核心算法由 PyTorch/NumPy 从零实现；不调用 gensim 训练器，也不声称复现原文 DistBelief 分布式系统。

## 组件关系

```text
configs/*.yaml ──┐
                 ├─> config.load_cfg() ─> strict Cfg
suite YAML ──────┘     experiments.plan.build_plan()
                           │
                 EXPERIMENT_REG(one_factor / grid)
                           │
                    1 run / paired seed
                           │
corpus files ─> TOKENIZER_REG ─> vocab + corpus fingerprint
                           │
             MODEL_REG + LOSS_REG + SCHED_REG
                           │
                       Trainer
               ┌───────────┴───────────┐
        objective + optimizer     metrics/checkpoint
               │                       │
          embeddings.*         manifest / logs / tb
               └───────────┬───────────┘
                   experiments.evaluate
                           │
                  per-category analogy
                           │
                  summary.csv / figures
```

## Registry 与真实调用点

| Registry | 注册/选择 | 调用者 |
| --- | --- | --- |
| `MODEL_REG` | CBOW、Skip-gram | `models/__init__.py` → `make_model` → `trainer/engine.py` |
| `LOSS_REG` | HS、NS 的 batch objective 分发 | `trainer/engine.py` → `_train_batch` |
| `TOKENIZER_REG` | simple、whitespace | `data/text_reader.py` → `iter_tokens` |
| `SCHED_REG` | none、linear token progress | `trainer/lr_schedulers.py` → `get_scheduler` |
| `EXPERIMENT_REG` | one_factor、grid | `experiments/plan.py` → `build_plan` |

模型在自身 `forward_hs` / `forward_ns` 中完成张量数学；objective Registry 负责选择并组装批次。不要把“工厂注册”误解为每个模块都有独立的可替换整个 Trainer 的接口。

## 数据流与内存

训练开始先流式构造词表、count 和 token-stream SHA-256。每轮重新流式遍历语料，通过 `SentenceIndexer` 按 epoch 独立次采样，再用 `generate_*_pairs_with_progress` 产生 CBOW/SG 样本。Trainer 仅缓存当前句子与最多 `TRAIN.batch_size` 个训练样本。独立 RNG streams 用于次采样、负采样及上下文窗口。source-token-progress LR 不依赖实际样本对数量。

HS 在 V 个词上使用 V−1 个 Huffman 内部 output 节点；Skip-gram 的 HS target 是上下文词，CBOW 的 target 是中心词。NS 按 unigram 幂次采样并排除与正例冲突的词。数学及历史修复见 [算法推导](from-scratch.md) 和 [正确性审查](training-correctness-review.md)。

## 可恢复性与证据

每个 epoch 可原子化保存模型、optimizer、scheduler、采样 RNG、全局 RNG、词表/语料签名和统计；恢复仅支持受控的 epoch 边界。Trainer 在每次语料 pass 核对 fingerprint，以检测源文件或 tokenization 意外漂移。每次完成的训练导出输入 embedding 和 `run_manifest.json`。实验层增补 `experiment.json`、`evaluation.json`、`events.jsonl`、`summary.csv`，可选择绘图。

准确率为分类 3CosAdd，必须同时报告 OOV 覆盖率。数据/硬件/模型预算不同会混淆公平比较。有关配置与流程的完整约定，参见 [配置参考](config-reference.md)、[流式训练](training-plan.md)、[恢复协议](checkpoint-resume.md)、[产物契约](experiment-artifacts.md)。
