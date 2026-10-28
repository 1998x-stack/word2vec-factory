# 文档中心 · Documentation Hub

当前规范与历史计划在此明确区分；先阅读已合并源码、当前测试与主文档，再参考旧规划稿。

## 首次使用

| 目标 | 文档 |
| --- | --- |
| 产品定位、安装、训练和报告 | [根目录 README](../README.md) |
| AI coding agent 执行标准 | [AGENTS.md](../AGENTS.md) |
| YAML 清单及其用途 | [配置索引](../configs/README.md) |
| 整体架构、算法推导 | [架构](architecture.md)、[从零实现](from-scratch.md) |

## 论文与消融研究

| 文档 | 内容 |
| --- | --- |
| [论文追溯矩阵](paper-1301-traceability.md) | 1301.3781 各节和实现、缺口 |
| [研究实验协议](ablation-protocol.md) | OFAT、paired seeds、控制变量 |
| [评价方法](evaluation.md) | 3CosAdd、语义/语法分组、OOV |
| [实验产物](experiment-artifacts.md) | Loguru、JSONL、TensorBoard、CSV 与图表 |
| [复现阶段路线图](reproduction-roadmap.md) | 已实现、可测、待获取真实语料的工作 |
| [消融索引（兼容旧链接）](ablation.md) | 快速跳转至当前实验协议 |

## 工程与可靠性

| 文档 | 内容 |
| --- | --- |
| [配置字段参考](config-reference.md) | YAML schema、默认值与参数语义 |
| [流式训练](training-plan.md) | LR progress、RNG、内存预算 |
| [Checkpoint/Resume](checkpoint-resume.md) | 恢复、数据一致性 |
| [正确性审查](training-correctness-review.md) | 曾发现的问题、修复及限制 |

## 历史设计档案

[superpowers/](superpowers/README.md) 为 2026-08 的历史设计/实施计划，其部分前提（例如无 CI、英文文档限定）已被后续实现取代。历史材料不作为当前 Agent 工作指令。

维护规则：新文档添加至本索引；新实验配置添加至 [configs/README.md](../configs/README.md)。明确区分“代码已实现”“CI 已验证”“真实语料已测量”“尚待复现”。
