# Ablations — 兼容导航页

此文件保留给已有 `docs/ablation.md` 链接。**现行研究协议及运行方式**集中维护在：

- [消融设计、变量和解释限制](ablation-protocol.md)
- [全部实验 YAML 配置目录](../configs/README.md)
- [原论文—代码追溯矩阵](paper-1301-traceability.md)
- [日志及报告说明](experiment-artifacts.md)
- [完整文档索引](README.md)

```bash
python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --dry-run
```

仅在具备许可明确的语料、正确评测协议和预算时再启动实际训练。默认小语料是开发用例，不是论文准确率的复现。
