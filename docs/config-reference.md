# Configuration reference

All run-level settings live in YAML and are loaded by `w2v_factory.config.load_cfg`.
A config can `INCLUDE` another file to inherit and override it. Relative include paths are resolved from the child config's directory, include cycles are rejected, and unknown keys fail fast instead of being silently ignored.

## Top-level

| Key | Default | Meaning |
| --- | --- | --- |
| `SEED` | `42` | non-negative root seed used to derive independent training RNG streams |

## `RUN`

| Key | Default | Meaning |
| --- | --- | --- |
| `out_dir` | `runs/exp1` | output directory for vectors + logs |
| `tb` | `true` | write TensorBoard logs |
| `checkpoint_every_epochs` | `null` | save an atomic checkpoint every N completed epochs; positive integer |
| `resume_from` | `null` | trusted epoch-boundary checkpoint path to restore |

## `DATA`

| Key | Default | Meaning |
| --- | --- | --- |
| `input_files` | `[]` | list of corpus paths (one sentence per line) |
| `lowercase` | `true` | lowercase tokens |
| `min_count` | `5` | drop words in the corpus below this count |
| `max_vocab` | `1000000` | cap vocabulary size |
| `subsample_t` | `1e-5` | frequent-word subsample threshold; `null`, `false`, or `0` disables; invalid/negative values fail fast |
| `tokenizer` | `simple` | tokenizer name (`simple`) |
| `max_sent_len` | `10000` | truncate over-long lines |

## `TRAIN`

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | `10` | number of passes over the corpus |
| `batch_size` | `1024` | maximum number of generated training examples buffered per optimizer step |
| `lr` | `0.025` | learning rate |
| `lr_schedule` | `linear` | `linear` decays by in-vocabulary source-token progress; `none` keeps LR fixed |
| `optimizer` | `sgd` | `sgd` or `adam` |
| `device` | `auto` | `auto`, `cpu`, or `cuda:N` |

## `MODEL`

| Key | Default | Meaning |
| --- | --- | --- |
| `arch` | `skipgram` | `skipgram` or `cbow` |
| `dim` | `300` | embedding size |
| `window` | `5` | max context window |
| `ns_neg_k` | `5` | number of negatives for NS |
| `loss` | `ns` | `ns` or `hs` |
| `share_input_output` | `false` | share the input/output embedding matrices |

> Note: the config was cleaned so every key above actually affects training
> (dead keys like `num_workers`, `save_every_steps`, and `hs_use_huffman` were
> removed from the schema).

## 新增研究消融控制（2026-10）

| YAML key | 默认值 | 含义 |
| --- | --- | --- |
| MODEL.window_sampling | uniform | uniform 每个中心词独立抽 R∈[1,C]；fixed 始终 C |
| MODEL.cbow_pooling | mean | mean 原论文 CBOW 平均；sum 非论文主设置的干预 |
| MODEL.ns_power | 0.75 | loss=ns 时 unigram 分布指数，来自 1310.4546 |

新增参数包含在 checkpoint training signature 中，升级后旧 checkpoint 的兼容性需要单独验证。研究 baseline 参阅 configs/paper1301_base.yaml。
