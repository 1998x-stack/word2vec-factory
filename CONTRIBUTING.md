# Contributing to word2vec-factory

本项目欢迎数学正确性修复、可复现消融、测试、文档与性能观测方面的贡献。目标是**可审查的研究实现**，不是不断扩张的框架功能。

## 快速开发环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[viz]'
python -m pip install pytest ruff black

python -m compileall -q w2v_factory tests
python -m pytest -q
python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --dry-run
```

`make lint` 检查 Ruff/Black；部分历史文件可能还不满足格式门禁。请在 PR 中明确区分**本次新增问题**与既有的格式问题，不要为了文档变更大规模重排无关源码。GitHub Actions 默认检查编译、pytest 及消融计划。

## 变更流程

1. 在新分支完成最小范围改动；说明问题来源、实现方式和客观验收条件。
2. 算法修改必须核对 [论文追溯矩阵](docs/paper-1301-traceability.md)，区分 1301.3781 与 1310.4546 的方法归属。
3. 复用现有 Factory 与 Trainer：配置字段变化要同步 schema、实验 YAML、checkpoint 签名兼容性、测试及文档。
4. 保留训练数值回归：Skip-gram 邻词目标、HS/NS、CBOW mask、RNG、流式进度、resume 和 OOV 评价。
5. 新文档链接加入 [docs/README.md](docs/README.md)，新实验配置登记在 [configs/README.md](configs/README.md)。中英文均可，但技术术语和配置命名应一致。
6. 对任何较重训练先使用 `--dry-run`，核对运行数、数据来源/许可、硬件、seed 与评测协议。不提交训练产物或第三方受限语料。
7. 提交专注的 commit / PR，列出运行的测试、CI 结果及未完成项。未经确认不直接合并到 `main`。

## PR 最低验收

- [ ] 变更影响、论文依据与实现边界清晰。
- [ ] 新行为有测试，相关旧测试通过。
- [ ] 实验规划可复核，计算预算已评估。
- [ ] README、AGENTS、配置文档、链接及状态表述一致。
- [ ] 任何准确率/吞吐率都来自真实运行，并携带 OOV 覆盖率、配置和数据溯源信息。

具体的 agent 执行规范见 [AGENTS.md](AGENTS.md)；整体文档入口见 [docs/README.md](docs/README.md)。
