# Configuration reference

All run-level settings live in YAML and are loaded by `w2v_factory.config.load_cfg`.
A config can `INCLUDE` another file to inherit and override it.

## Top-level

| Key | Default | Meaning |
| --- | --- | --- |
| `SEED` | `42` | global random seed |

## `RUN`

| Key | Default | Meaning |
| --- | --- | --- |
| `out_dir` | `runs/exp1` | output directory for vectors + logs |
| `tb` | `true` | write TensorBoard logs |

## `DATA`

| Key | Default | Meaning |
| --- | --- | --- |
| `input_files` | `[]` | list of corpus paths (one sentence per line) |
| `lowercase` | `true` | lowercase tokens |
| `min_count` | `5` | drop words in the corpus below this count |
| `max_vocab` | `1000000` | cap vocabulary size |
| `subsample_t` | `1e-5` | frequent-word subsample threshold; `null`/`'null'` disables |
| `tokenizer` | `simple` | tokenizer name (`simple`) |
| `max_sent_len` | `10000` | truncate over-long lines |

## `TRAIN`

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | `10` | number of passes over the corpus |
| `batch_size` | `1024` | number of target/completed examples per step |
| `lr` | `0.025` | learning rate |
| `lr_schedule` | `linear` | `linear` or `none` |
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