# Evaluation — 3CosAdd 类比测试

本仓库的 `questions-words.txt` 包含按 `: category` 标记的 Google analogy 风格题目。训练后的输入向量用于 `a : b :: c : d` 类比，查询向量使用：

```text
q = v_b - v_a + v_c
score(candidate) = cosine(q, v_candidate)
```

**先用原始向量计算偏移，再对 q 和候选向量计算余弦**；从候选集合排除 `a`、`b`、`c`。不得在构造偏移之前将三个词逐个归一化。

## 两个评测入口

**单个模型的传统 CLI：**

```bash
python -m w2v_factory.cli.eval \
  --vectors runs/exp1/embeddings.txt \
  --analogy questions-words.txt --topk 1
```

`w2v_factory.trainer.metrics.evaluate_analogy` 返回一个整体命中率；它会跳过 OOV 题目，但**不会同时显示 eligible / total 及分类覆盖率**。因此它适合粗略 smoke test，不应单独用于论文级结论。

**实验工厂的研究评测：**

`w2v_factory.experiments.evaluate.evaluate_file` 在已完成的 ablation run 上计算并保存 `evaluation.json`。它包含文件 SHA-256、overall、semantic、syntactic 和各 `category` 的：

- `total`：该类别全部题目数；
- `eligible`：四个词都在词表中的题目数；
- `correct`：eligible 中 top-1 预测正确数；
- `coverage = eligible / total`；
- `accuracy = correct / eligible`，若 eligible 为零则是 `null`，不是 0%。

可运行一个基线方案，再使用：

```bash
python -m w2v_factory.cli.ablate \
  --grid configs/paper1301_ablations.yaml --dry-run

python -m w2v_factory.cli.report --root runs/paper1301/ablations
```

报告命令只读取**已完成**的实验，`--dry-run` 不生成实际分数。

## 公平比较与边界

- 当前实验工厂默认根据 `DATA.lowercase` 对题目转换大小写；单文件旧 CLI 无此自动对齐，训练文本转小写时必须注意评测协议差异。
- 词表不同 → eligible 子集不同；不能只比较两组 eligible 上的准确率而忽略覆盖率。需要严谨对比时应采用共同 eligible 子集。
- 稀疏语料可能只有很少 eligible 题目，质量曲线没有实质统计解释力。
- 仓库包含评测问题文件，**它不是训练语料**；不要将测试题混入训练。
- `embeddings.txt` 采用 Word2Vec 文本格式：首行 `V dim`，后续每行是 `word value1 ... valueD`。格式加载函数见 `w2v_factory/cli/eval.py`。
- Sentence Completion 与语义/语法 analogy 是不同任务，当前并未实现原论文 Sentence Completion 的匹配评分协议。

关于完整的统计约束与可视化，请阅读 [消融实验协议](ablation-protocol.md) 和 [实验产物说明](experiment-artifacts.md)。
