# 配置目录与实验方案

单次训练 YAML 经 `w2v_factory.config.load_cfg()` 加载；实验规划 YAML 经 `w2v_factory.experiments.plan.build_plan()` 加载。两种 schema 不可互换；命令均从仓库根目录运行。

| 文件 | 类型 | 用途 |
| --- | --- | --- |
| [base.yaml](base.yaml) | 训练基线 | 默认设置；含示例语料与后续论文 NS/subsampling |
| [cbow_hs.yaml](cbow_hs.yaml) | 单次训练 | CBOW + HS |
| [skipgram_ns.yaml](skipgram_ns.yaml) | 单次训练 | Skip-gram + NS |
| [paper1301_base.yaml](paper1301_base.yaml) | 研究基线 | 原论文 CBOW/HS 基线；仍使用小数据 |
| [paper1301_ablations.yaml](paper1301_ablations.yaml) | 研究方案 | OFAT、paired seeds、历史联合 profiles |
| [paper1301_table2_grid.yaml](paper1301_table2_grid.yaml) | 研究方案 | dim × token 预算网格，依赖先生成切片 |
| [post2013_extensions_base.yaml](post2013_extensions_base.yaml) | 扩展基线 | 后续论文 Negative Sampling 实验 |
| [post2013_extensions.yaml](post2013_extensions.yaml) | 扩展方案 | NS、采样指数、subsampling |
| [ablation_grid.yaml](ablation_grid.yaml) | 兼容旧网格 | 通用全因素矩阵，规模可能很大 |
| [paper_repro.yaml](paper_repro.yaml) | 旧演示 | 历史命名；**不能当作严格原文复现** |

## 操作顺序

1. 选择实验 YAML，先执行 `python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --dry-run`。
2. 核对运行数、`max_runs`、seed、设备与全因素规模，防止意外产生大量训练。
3. 真实研究应替换 `DATA.input_files` 的演示语料，记录来源、许可、预处理、词表规则与分割方式。
4. Table 2 网格要求先用 `python -m w2v_factory.cli.corpus_budget` 创建 `data/paper1301_slices/`，以实际文件和 `budgets.json` 为证据。
5. `INCLUDE` 的路径相对当前 YAML 解析；实验 `base_cfg` 可使用当前工作目录路径或相对实验 YAML 目录路径。
6. 在 HS 下，`ns_neg_k` 与 `ns_power` 不生效；请避免将无效维度误报成独立消融。原论文与后续论文实验应使用各自的配置。
7. 确认计划之后才执行训练，并保留产物 manifest、日志、评测覆盖率。

深入阅读：[配置字段](../docs/config-reference.md)、[论文追溯](../docs/paper-1301-traceability.md)、[消融协议](../docs/ablation-protocol.md)。
