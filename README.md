# word2vec-factory

**论文级 Word2Vec Factory / Ablation Lab：PyTorch、可复现、可审计。**

基于 [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) 研究 CBOW、Skip-gram、Huffman Hierarchical Softmax、上下文窗口、维度与语义/语法类比。Negative Sampling 与高频词 Subsampling 属于 [1310.4546](https://arxiv.org/abs/1310.4546) 的后续扩展，单独实验，不混写贡献来源。

## Factory 流程

    experiment YAML -> EXPERIMENT_REG (OFAT / grid / paired seeds)
                    -> TOKENIZER_REG -> streaming vocabulary / corpus hash
                    -> MODEL_REG + LOSS_REG + SCHED_REG -> Trainer (PyTorch)
                    -> category-aware 3CosAdd / OOV coverage
                    -> Loguru + TensorBoard + JSONL -> CSV / figures

核心能力保留：流式有界内存、精确 Huffman paths、负采样 alias table、独立 RNG streams、token-progress LR、epoch checkpoints、训练向量导出、CI 回归测试。

## 快速开始

```bash
python -m pip install -e '.[viz]'
python -m pytest -q

# 单次训练
python -m w2v_factory.cli.train --cfg configs/cbow_hs.yaml

# 先打印而不是直接启动大规模矩阵
python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --dry-run
python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --fail-fast

# 区分后续论文的扩展
python -m w2v_factory.cli.ablate --grid configs/post2013_extensions.yaml --dry-run

# 汇总所有实际完成的运行
python -m w2v_factory.cli.report --root runs/paper1301/ablations
tensorboard --logdir runs/paper1301/ablations
```

**默认 sample_corpus.txt 是烟测数据，不是论文的 Google News 或 LDC 原始训练集。** 正式对照需要替换 configs/paper1301_base.yaml 的 DATA.input_files 并记录语料版权、来源与切分。源码没有运行的实验不得对外声称已取得论文结果。

## 研究文档

- [原论文逐节对应代码与缺口](docs/paper-1301-traceability.md)
- [消融矩阵与评价协议](docs/ablation-protocol.md)
- [日志、结果、可视化](docs/experiment-artifacts.md)
- [阶段任务与验收](docs/reproduction-roadmap.md)
- [工程架构](docs/architecture.md)
- [配置参考](docs/config-reference.md)
- [数值正确性审计](docs/training-correctness-review.md)
- [训练规划](docs/training-plan.md)
- [Checkpoint/Resume](docs/checkpoint-resume.md)
