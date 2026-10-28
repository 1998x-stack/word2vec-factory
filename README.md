# word2vec-factory

**PyTorch Word2Vec：从零实现、可追溯论文研究与消融实验工厂**

Word2Vec Factory 是一个面向算法研究和教学的实验工程，覆盖 CBOW / Skip-gram、Huffman Hierarchical Softmax、Negative Sampling、流式训练、可恢复执行、自动化消融、类比评测及可视化。核心目标是让研究结论可以追溯到**原论文、实现、配置、数据、测试和实验产物**，而不是构建通用 NLP 框架。

> **项目状态：** 主要算法与实验基础设施已实现并配备自动化测试；原论文的大规模语料实验、DistBelief 及外部基线尚未完整复现。`sample_corpus.txt` 是功能测试语料，不能据此声称复现论文准确率。

## 1. 论文与研究边界

| 一手资料 | 研究机制 | 本仓库实现状态 |
| --- | --- | --- |
| [Mikolov et al., 2013 — arXiv:1301.3781](https://arxiv.org/abs/1301.3781) | CBOW、Skip-gram、Huffman HS、上下文窗口、embedding 维度、语义与语法类比 | 核心模型及消融配置已实现；大规模实证尚未完成 |
| [Mikolov et al., 2013 — arXiv:1310.4546](https://arxiv.org/abs/1310.4546) | Negative Sampling、unigram 指数、频繁词 Subsampling | 作为**后续扩展**单独实验；不归因于 1301.3781 |
| 原文外部系统与任务 | DistBelief、NNLM/RNNLM、Sentence Completion、原始训练大语料 | 尚未复现，不提供模拟指标 |

论文每节对应实现、验证和缺口，参见 [论文可追溯矩阵](docs/paper-1301-traceability.md)。

## 2. 核心能力

- **Factory-first：** Registry 实际分发模型、目标函数、分词器、调度器与实验规划；配置采用严格 YAML schema 和 `INCLUDE` 继承。
- **正确性：** CBOW padding mask；Skip-gram 邻词目标；紧凑 Huffman 内部节点；NS 正样本冲突排除；3CosAdd 原始向量偏移。
- **流式训练：** 按 epoch 重读语料、有界 batch、独立 RNG streams、source-token progress 学习率、epoch checkpoint/resume。
- **研究设计：** OFAT 单因素、full grid、paired seeds、配置哈希 run ID、预算限制与 `--dry-run`、失败留痕及已完成跳过。
- **观测与分析：** Loguru、TensorBoard、JSONL、分类类比准确率与 OOV 覆盖率、CSV、可选 Matplotlib 与 PCA 诊断。

## 3. 安装与快速开始

要求：**Python 3.10+**。CPU 即可运行功能测试；使用 CUDA 需有兼容的 PyTorch 运行环境。以下命令均在仓库**根目录**执行。

```bash
git clone https://github.com/1998x-stack/word2vec-factory.git
cd word2vec-factory

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[viz]'
python -m pip install pytest

python -m pytest -q

# 示例训练：使用仓库的 Grimm 童话文本，非论文复现
python -m w2v_factory.cli.train --cfg configs/cbow_hs.yaml
```

仅需训练、不生成可选图表时，可以执行 `python -m pip install -e .`。上例向量、日志等产物默认写到 `runs/exp1/`。

## 4. 消融工厂：先规划，再训练

```bash
# 原论文核心消融：只打印运行计划，不训练
python -m w2v_factory.cli.ablate \
  --grid configs/paper1301_ablations.yaml --dry-run

# 在确认真实语料、成本、评测协议后才执行完整计划
python -m w2v_factory.cli.ablate \
  --grid configs/paper1301_ablations.yaml --fail-fast

# Negative Sampling / Subsampling 属于后续论文扩展
python -m w2v_factory.cli.ablate \
  --grid configs/post2013_extensions.yaml --dry-run
```

单因素研究固定基线并在各变体间配对随机种子；历史 profiles 可能同时改变多个参数，因此**不构成单因素归因**。全因素矩阵成本为每个因子取值数与 seed 数的乘积。配置选择、前置条件及完整目录见 [configs/README.md](configs/README.md)。

## 5. 论文 Table 2：训练词数 × 向量维度

仓库提供精确 token 预算的嵌套语料切片工具及对应网格。**原论文训练语料不在仓库内。** 需自行准备许可清晰、顺序及预处理策略可复核的文本，执行：

```bash
python -m w2v_factory.cli.corpus_budget \
  --input data/source.txt \
  --out-dir data/paper1301_slices \
  --budgets 24000000 49000000 98000000 196000000 391000000 783000000

python -m w2v_factory.cli.ablate \
  --grid configs/paper1301_table2_grid.yaml --dry-run
```

生成的 `budgets.json` 记录 token 数量、SHA-256 与数据处理策略。切片使用**原文档顺序前缀**，不是随机采样；如数据带有时间或主题顺序，应在切片前进行可追溯的文档级顺序控制。实现了实验设计，不代表已取得论文的结果。

## 6. 评测、运行日志与可视化

```bash
# 单个向量文件的类比评测
python -m w2v_factory.cli.eval \
  --vectors runs/exp1/embeddings.txt \
  --analogy questions-words.txt

# 实验工厂结果汇总（需要已完成的运行）
python -m w2v_factory.cli.report --root runs/paper1301/ablations

# 可选：训练标量曲线
tensorboard --logdir runs/paper1301/ablations
```

| 产物 | 说明 |
| --- | --- |
| `runs/<run_id>/run_manifest.json` | 完整配置、词表、语料指纹、训练和恢复状态 |
| `runs/<run_id>/experiment.json` / `evaluation.json` | 实验因子、seed、耗时、评测数据及状态 |
| `runs/<run_id>/metrics.jsonl` / `train.log` / `tb/` | 训练标量、Loguru、TensorBoard |
| `runs/<run_id>/embeddings.txt` / `embeddings.npy` | Word2Vec 格式及 NumPy 输入词向量 |
| `runs/<suite>/events.jsonl` / `summary.csv` | 任务事件和已完成实验结果 |
| `runs/<suite>/*.png` | 质量/速度、语料规模/维度和投影诊断（可选） |

**评测解释：** 每个指标同时报告 `total`、`eligible`、`correct`、`coverage`；未覆盖的题目不进入 eligible 分母，eligible 为零时不能声称存在有效准确率。示例语料与类比测试题集词表重合度有限。不能以 PCA 图像观感、不同硬件的吞吐率或 OOV 覆盖率不同的原始准确率单独宣称模型改善。

详见 [评测协议](docs/evaluation.md) 和 [实验产物文档](docs/experiment-artifacts.md)。

## 7. 工程目录

```text
word2vec-factory/
├── README.md                # 项目入口
├── AGENTS.md                # AI coding agent 开发契约
├── CONTRIBUTING.md          # 开发与贡献规范
├── configs/README.md        # YAML 类型与用途索引
├── docs/README.md           # 文档导航与研究入口
├── w2v_factory/
│   ├── cli/                 # train / ablate / eval / report / corpus_budget
│   ├── data/                # 文本、词表、采样、Huffman 与窗口
│   ├── models/              # CBOW / Skip-gram
│   ├── losses/              # HS / NS 辅助
│   ├── experiments/         # 实验规划、评价及统计图
│   ├── trainer/             # 训练、checkpoint、metrics、调度
│   ├── config.py            # 配置 schema
│   └── registry.py          # Factory registries
├── tests/                   # 数学正确性及功能回归
├── scripts/                 # 便捷操作
├── sample_corpus.txt        # 示例文本
└── questions-words.txt      # 类比问题文件
```

## 8. 文档与贡献

[文档中心](docs/README.md) · [配置目录](configs/README.md) · [研究协议](docs/ablation-protocol.md) · [阶段路线图](docs/reproduction-roadmap.md) · [工程架构](docs/architecture.md) · [参与开发](CONTRIBUTING.md) · [Agent 指令](AGENTS.md)

任何实验结果应保留代码版本、语料来源、配置、随机种子、覆盖率及评测协议。`runs/`、本地 `data/`、checkpoint、模型权重和密钥不提交。软件采用 [MIT License](LICENSE)，第三方语料与基准的许可另行核实。
