# Configuration reference — YAML / dataclass

配置由 `w2v_factory.config.load_cfg()` 严格解析。`INCLUDE` 路径相对当前 YAML 文件；未知字段、循环继承和非法 subsampling threshold 会明确报错。单次训练 YAML 与实验规划 YAML 的格式不同，参见 [configs/README.md](../configs/README.md)。

## 通用与数据

| YAML key | 默认值 | 说明 |
| --- | --- | --- |
| `SEED` | `42` | 随机数根种子；训练通过独立 streams 派生 |
| `RUN.out_dir` | `runs/exp1` | 运行输出目录 |
| `RUN.tb` | `true` | 是否写 TensorBoard |
| `RUN.checkpoint_every_epochs` | `null` | 每 N 完整 epoch checkpoint |
| `RUN.resume_from` | `null` | 可信 epoch-boundary checkpoint 路径 |
| `DATA.input_files` | `[]` | 逐行文本文件列表 |
| `DATA.lowercase` | `true` | 是否统一小写 |
| `DATA.tokenizer` | `simple` | simple / whitespace（当前均以空格分词） |
| `DATA.min_count` | `5` | 词表最低频数 |
| `DATA.max_vocab` | `1000000` | 词表最大规模 |
| `DATA.max_sent_len` | `10000` | 每条输入记录最大 token 数 |
| `DATA.subsample_t` | `1e-5` | 高频词下采样阈值；`null`/`0`/`false` 禁用；属于 1310.4546 扩展 |

## 训练配置

| YAML key | 默认值 | 说明 |
| --- | --- | --- |
| `TRAIN.epochs` | `10` | 语料遍历轮数 |
| `TRAIN.batch_size` | `1024` | batch 缓冲最大样本数 |
| `TRAIN.lr` | `0.025` | 初始学习率 |
| `TRAIN.lr_schedule` | `linear` | source-token 进度线性衰减；`none` 不衰减 |
| `TRAIN.optimizer` | `sgd` | `sgd` / `adam` |
| `TRAIN.device` | `auto` | `auto`、`cpu`、`cuda:N` 等 |

## 模型配置

| YAML key | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL.arch` | `skipgram` | `skipgram` / `cbow` |
| `MODEL.dim` | `300` | 向量维度 |
| `MODEL.window` | `5` | 最大上下文半径 |
| `MODEL.window_sampling` | `uniform` | 每中心词在 1..window 中抽半径；`fixed` 固定最大半径 |
| `MODEL.cbow_pooling` | `mean` | CBOW mean（论文）/ sum（对照干预） |
| `MODEL.loss` | `ns` | `hs` / `ns` |
| `MODEL.ns_neg_k` | `5` | NS 每个正例的负样本个数（仅 `loss=ns` 时生效） |
| `MODEL.ns_power` | `0.75` | unigram noise exponent（仅 `loss=ns` 时生效） |
| `MODEL.share_input_output` | `false` | 输入与输出 embedding 是否共享；HS 不支持共享 |

`EVAL.analogy_file` 默认为 `null`。实验工厂配置的研究基线会指定仓库内 `questions-words.txt`。旧的单文件评价 CLI 仍使用单独的命令行参数。

## 实验 YAML（不是 Cfg）

`base_cfg`、`out_root`、`strategy`（`one_factor` 或 `grid`）、`seeds`、`max_runs`、`factors`/`grid` 及 `profiles` 由 `w2v_factory.experiments.plan.build_plan()` 处理。`MODEL.dim` 等带点路径对应 Cfg 中的字段；不支持任意未声明键。变更会影响实验 run ID 和 checkpoint 训练签名，不要随意复用旧 checkpoint。

完整示例在 [配置清单](../configs/README.md)，实验解释在 [消融协议](ablation-protocol.md)。
