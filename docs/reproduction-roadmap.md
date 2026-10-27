# 分阶段交付与客观验收

## Stage 00 — 一手论文校验（已建立）
- 以 arXiv:1301.3781 为主，1310.4546 为扩展，逐节建立原文、算法、配置、测试之间的可追溯矩阵。
- 标注外部依赖：LDC、Google News 6B、DistBelief、MSR sentence completion。
**验收：** 任一能力的 source paper attribution 可核查，不冒充已完成大规模实证。

## Stage 01 — 工厂模式与数值正确性（已提交代码，待 CI）
- MODEL_REG 创建 CBOW/Skip-gram，LOSS_REG 选择 HS/NS 目标，SCHED_REG 选择学习率策略，TOKENIZER_REG 控制分词。
- 加入 fixed/uniform window、mean/sum pooling、ns noise power 消融钩子。
- 保留既有 streaming、negative collision exclusion、HS target path、RNG isolation、checkpoint test。
**验收：** 原有测试 + tests/test_paper1301_factory.py 通过；预测目标和解析路径数学一致。

## Stage 02 — 消融编排、日志、分析（已提交代码，待实测）
- 支持历史对照、单因素/全因素、paired seeds、预算限制、dry-run、故障记录、已完成跳过。
- 分类别 analogy accuracy/OOV coverage，CSV 汇总，Matplotlib 质量和成本图。
- TensorBoard、Loguru、JSONL structured metrics、实验清单。
**验收：** 相同输入重复规划得到稳定 ID；零 eligible 不虚构准确率；报告只处理完成的运行。

## Stage 03 — 真实原论文表格复现（待外部语料）
- 取得许可合规的训练语料与固定词表/预处理，准备 24M-783M nested token budgets。
- 实施维度×数据量交互项；控制训练 tokens、unique tokens 和 epochs。
- 固定评测集哈希，输出 semantic/syntactic 分项、覆盖率、误差估计和吞吐。
**验收：** 不同设备实验分组，结果可复核，论文表格仅作外部参照而非替代测量。

## Stage 04 — 外部基线与推理系统（待计划）
- Sentence completion 原协议及合规数据、NNLM/RNNLM 独立基线、端到端内存与吞吐剖析。
- 若资源允许，研究 Async/Adagrad/分布式；必须记录硬件和算法是否与 DistBelief 可比。
**验收：** 新机制配套单元/集成/基准，能解释与原论文的差异。

不包括：凭空复制原文数据、制造漂亮但无实验依据的图表、自动合并未经 CI 验证的分支。
