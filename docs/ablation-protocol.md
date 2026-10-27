# 消融研究协议：固定假设、独立变量、可信评价

## 实验设计

- **one_factor（推荐）**：以 CBOW/HS/window=4/fixed/pooling=mean 为基线，逐个改变唯一因素。3 个预设 seed (11,23,37) 成对比较。不能把它当成大规模统计显著性检验。
- **profiles**：额外记录原文条件 CBOW(4,fixed) 与 Skip-gram(10,uniform)，属于**多个配置同时变化的历史画像**，不可把结果差异归因于 architecture 一项。
- **grid**：供后续互作用研究采用笛卡尔积。max_runs 是必须启用的计算预算护栏；先 dry-run 再执行。

| ID | 对比 | 原文与推理 |
| --- | --- | --- |
| A01 | arch: CBOW / SG | §3：预测方向不同；分别报告语义和语法 |
| A02 | dim: 50/100/300/600 | §4.2：容量与计算开销的均衡 |
| A03 | window: 2/4/8/10 | §3.2：范围与语境噪声、训练样本数 |
| A04 | window_sampling: fixed / uniform | §3.2：远距离 token 更少被选择 |
| A05 | CBOW pooling: mean / sum | §3.1：确认平均操作与 context 数量归一化的贡献 |
| A06 | epochs: 1/3/5 | §4.2-4.3：多遍曝光；此对比**未**固定总 token 数 |
| A07 | LR: .0125/.025/.05 | §4.2：优化器初始步长与衰减 |
| X01 | HS / NS | 仅 **1310.4546** 后续，不应写作 1301 的技术贡献 |
| X02 | negatives: 2/5/10 | 仅 loss=ns 的相关参数 |
| X03 | noise exponent: .5/.75/1.0 | 仅 loss=ns，固定 vocab 与 subsampling |
| X04 | subsample_t: null/1e-5/1e-4 | 后续论文频繁词处理 |

## 评测协议

质量指标：overall、semantic、syntactic 及每个 Google analogy section 的 strict 3CosAdd top-1。每项存 total/eligible/correct/coverage，eligible=0 时 accuracy=null。用户必须同时查看 coverage，避免只在容易且 OOV 很少的一部分问题上比较准确率。严格对齐 tokenization、语料与评测文件 sha256。

效率指标：experiment.json 记录 wall time；summary.csv 记录全流程 tokens/s（**含构建词表**）、实际训练 examples、optimizer steps、vocab size；若硬件不同不得直接比较吞吐。训练日志：train.log (Loguru)、metrics.jsonl (step/epoch)、events.jsonl（跨实验状态）、TensorBoard 的 step 与 epoch scalar。图表：accuracy±seed standard error、quality/speed 散点，以及可选 PCA；降维图不等于质量验证。

## 原论文仍需完成的受控实验

1. Table 2：在一个版本锁定且许可清晰的大语料上预生成可追溯的嵌套 24M/49M/98M/196M/391M/783M token 子集，和四个 dim 交叉；固定训练词表规则，解释 vocabulary shift 的影响。
2. Table 5：增加“unique training tokens”受控对比；较大语料单遍与较小语料三遍不止训练轮数一个因素。
3. Table 3：与 NNLM/RNNLM 对比前，必须先取得其原文数据条件与成本统计；否则应明确为非等价历史参考。
4. Sentence completion 及 DistBelief 并行训练均暂不实现，没有真实数据和测试不得声称复现。

## 正式验收

执行全部测试、单次训练 sanity、dry-run 验证 run_id 稳定。之后运行真实语料，逐 seed 报告完整结果、置信区间与统计检验（样本量足够时才做推断）。禁止编造论文数值或通过 toy corpus 断言模型性能。对比不同 embedding 维度时应同时报告模型大小和实际训练成本。
