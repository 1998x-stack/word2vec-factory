# arXiv:1301.3781：论文—代码—实验可追溯矩阵

**第一手资料**：Mikolov, Chen, Corrado, Dean. Efficient Estimation of Word Representations in Vector Space (2013, v3), https://arxiv.org/abs/1301.3781 。下表描述实现位置，**不是**声称已经在原文数据集上完成复现。

| 原文 | 核心方法与事实 | 代码和配置 | 验收 / 限制 |
| --- | --- | --- | --- |
| §2 Eq.(1) | 复杂度 O=E×T×Q | trainer/engine.py、training_plan | 记录 epoch、tokens、examples、真实耗时；尚未测量精确 FLOPs |
| §2.1 | 频率 Huffman 树；高频词编码更短 | data/huffman.py、losses/hierarchical_softmax.py | test_huffman.py、test_training_correctness.py |
| §2.3 | DistBelief 多副本 + Adagrad | **未实现** | 不能声称原文分布式实验复现 |
| §3.1 Eq.(4) | CBOW：上下文词向量平均，无词序，历史4+未来4 | models/cbow.py；window=4、window_sampling=fixed | mean/sum 反事实消融，测试验证 padding 不改变均值 |
| §3.2 Eq.(5) | Skip-gram：中心词预测邻词；R∼Uniform{1,...,C}，实验 C=10 | data/dataset.py；window=10、window_sampling=uniform | fixed/full-window 是对照，不属于原论文默认 SG |
| §4.1 | 3CosAdd 排除输入词，5 种 semantic + 9 种 syntactic | experiments/evaluate.py | 分类 exact top-1 与 OOV eligibility |
| §4.2 Table 2 | 维度 50/100/300/600 与词量 24M 至 783M 联合变化 | configs/paper1301_ablations.yaml | **维度已设计；词量受控切片尚未实现** |
| §4.2 | SGD，初始 LR=0.025，线性降低，3 epochs | configs/paper1301_base.yaml；trainer/lr_schedulers.py | 当前实现使用 token-progress decay；不等于逐步复刻原始实现 |
| §4.3 Tables 3-5 | 模型/语义语法差异；1 vs 3 epochs；更多独立语料 vs 重复迭代 | experiments/plan.py、summary.py | 已设计受控消融；无真实结果则不下定论 |
| §4.4 Table 6 | Google News 6B、多副本分布式 | **未实现** | 原文大型专有语料和硬件不可替代 |
| §4.5 Table 7 | Microsoft Sentence Completion Challenge | **未实现** | 需要单独评分协议与许可数据 |
| §5 Table 8 | 类比偏移及多示例关系 | experiments/evaluate.py | PCA 只作诊断，不能代替定量类比 |
| 后续 1310.4546 | 高频词 subsampling、negative sampling、noise power=0.75 | data/subsample.py、data/sampler.py、configs/post2013_extensions.yaml | **独立扩展线，不混入 1301 原论文复现结论** |

## 核心数学契约

1. CBOW 使用 h = (1 / |ctx|) Σ(v_context) 预测中间词，pad 位必须由长度 mask 清除。
2. Skip-gram 使用中心词 v_t 预测被选中的周围词；最大半径 C 下，每个中心词的 R 均匀采样于 1..C。距离 d 的出现概率在非句边界条件下为 (C-d+1)/C。
3. Hierarchical Softmax 在 V 个单词上仅有 V-1 个内部节点，每个目标词沿 Huffman 路径优化所有二元判别项。Skip-gram 的预测标签必须是**邻词**。
4. 类比采用原始输入向量 q=v_b-v_a+v_c，再计算余弦 cos(q,v_candidate)，排除 a、b、c。不允许在求偏移之前悄悄对输入向量逐个归一化。
5. 参数变化必须对应实际训练行为：CBOW pooling 只作用于 CBOW；negative-sampling 的 K/power 只在 loss=ns 时有意义。

## 哪些实验不能自称“复现成功”

默认 sample_corpus.txt 不包含原论文语料，不足以比较 semantic / syntactic accuracy。真实复现还需要受控的 nested corpus size、同一评测词表的覆盖率控制，以及跨随机种子统计。论文 Table 5 中“1 epoch 更多新词”和“3 epochs 重复较少词”不能简单只变 TRAIN.epochs 当作等价实验。现代 PyTorch 单机实现不能直接对比 DistBelief 的 days × CPU-cores。
