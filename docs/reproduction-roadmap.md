# 论文复现阶段路线图与验收标准

本路线图区分**已实现代码**、**有自动化测试证据**与**尚未开展的真实数据实验**。GitHub CI 只能证明当前测试覆盖的性质，不能证明模型达到论文指标。

## Stage 00 — 原始论文与研究边界

**代码/文档已建立：** 1301.3781（CBOW、Skip-gram、HS、类比、规模）与 1310.4546（NS、subsampling）分开归因；外部 DistBelief、LDC/Google News、Sentence Completion 已列为缺口。

**验收：** [论文可追溯矩阵](paper-1301-traceability.md) 能把研究结论定位到论文段落、源码和可检验条件。

## Stage 01 — 数学正确性与 Factory

**代码已实现，现有测试覆盖：** MODEL/LOSS/SCHED/TOKENIZER Registry，CBOW mean/sum、window fixed/uniform、NS exponent、Huffman HS、随机流、streaming、checkpoint/resume。

**验收：** `pytest` 中包含 CBOW/Skip-gram × HS/NS smoke、scalar-vs-vectorized 梯度和目标检查、RNG/恢复回归。每次改动需重新执行，不能把旧 CI 状态自动套用到新 commit。

## Stage 02 — 消融编排、可观测性和报告

**代码已实现，真实大规模结果未实测：** one_factor/grid、paired seeds、dry-run/max_runs、Run IDs、JSONL、manifest、category-aware 3CosAdd、OOV coverage、CSV 和可选 Matplotlib。

**验收：** 同样输入得到稳定计划；仅完成状态进入汇总；零 eligible 不报告有效准确率；所有图表有相应实际运行文件支撑。

## Stage 03 — 原论文的规模与公平比较

**工具已实现、数据实验待完成：** `cli/corpus_budget.py` 生成带 SHA-256 的嵌套 24M–783M token 前缀；`configs/paper1301_table2_grid.yaml` 规划维度 × 数据预算。仍需获得许可明确的训练语料，固定文本顺序、tokenization、原始词数、训练曝光及评测 split。

**验收：** 各语料预算与维度按种子提供真实训练结果、覆盖率、效率与不确定性估计；相同硬件上复测。原论文 Table 5“更多唯一语料 vs 重复 epoch”需独立设置数据/曝光预算，而不能仅修改 epochs。

## Stage 04 — 外部系统与任务基线

**尚未实现：** 原论文 DistBelief 分布式训练、合规外部 NNLM/RNNLM 参考、Sentence Completion 对应评分；独立性能剖析可在有研究需要时另建任务。

**验收：** 取得明确数据、硬件和评分协议并构建独立基准，明确不可比条件。不允许伪造缺少的数据或将 toy-corpus 曲线当原论文结果。

执行入口见 [README](../README.md)、[配置目录](../configs/README.md) 与 [研究协议](ablation-protocol.md)。
