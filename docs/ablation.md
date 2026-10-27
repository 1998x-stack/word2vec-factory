# Ablations / 论文级消融

- [论文逐节对应](paper-1301-traceability.md)
- [研究设计与消融矩阵](ablation-protocol.md)
- [日志及可视化](experiment-artifacts.md)
- [分阶段复现任务](reproduction-roadmap.md)

原有 configs/ablation_grid.yaml 的 grid 模式保持兼容；新增的 paper1301_ablations.yaml 采用 paired-seed one_factor，post2013_extensions.yaml 研究另一篇论文的 NS/subsampling。

先执行：

    python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --dry-run

再在有真实语料及计算预算时执行：

    python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --fail-fast
    python -m w2v_factory.cli.report --root runs/paper1301/ablations

默认小语料仅用于验证程序，不能替代原论文结果。
