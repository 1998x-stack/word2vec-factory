# 实验产物与日志契约

## 数据管线

YAML -> EXPERIMENT_REG(one_factor/grid) -> validated Cfg -> TOKENIZER_REG -> MODEL_REG、LOSS_REG、SCHED_REG -> streaming Trainer -> 3CosAdd evaluator -> summary collector -> matplotlib。

每个 run 目录包含完整 resolved config、corpus_sha256、训练词表信息、RNG stream、实际样本数以及训练 epoch 信息。运行目录含 run_id、seed、配置哈希，保证可回溯；实验默认串行，避免由于 GPU 竞争造成失真。

| 文件 | 用途 |
| --- | --- |
| events.jsonl | UTC 时间戳、start/completed/failed/skip、run_id、异常 |
| RUN/train.log | Loguru 训练日志与轮转 |
| RUN/metrics.jsonl | step loss、学习率、token 进度、epoch 汇总 |
| RUN/tb/ | TensorBoard step / epoch 曲线 |
| RUN/run_manifest.json | 配置、corpus fingerprint、训练恢复状态、随机流、各 epoch 计数 |
| RUN/experiment.json | 研究因素、seed、状态、耗时、eval |
| RUN/evaluation.json | analogy 文件哈希、类别分解、OOV coverage |
| summary.csv | 每个完整 run 一行，无合成或缺失填补 |
| analogy_accuracy.png | 按 variant 的 seed 均值 ± 标准误；需检查 coverage |
| quality_speed.png | 吞吐与合法 analogy 准确率，固定设备时才有可比性 |
| embedding_pca.png | 最常见词向量的可视诊断，不是论文的定量准确率 |

已完成且 config 相同的实验默认跳过；不同配置不允许隐式复用。--force 显式重新运行。失败写入 failed 及异常原因，不得被 summary 作为已完成结果。集成的日志和向量数据不自动上传至 GitHub 仓库。

## 安装与执行

    python -m pip install -e '.[viz]'
    python -m pytest -q
    python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --dry-run
    python -m w2v_factory.cli.ablate --grid configs/paper1301_ablations.yaml --fail-fast
    python -m w2v_factory.cli.ablate --grid configs/post2013_extensions.yaml --fail-fast
    python -m w2v_factory.cli.report --root runs/paper1301/ablations

注意：默认语料 sample_corpus.txt 只用于烟测。没有真实实验输出时报告“尚未实测”，不要生成假曲线。PNG 生成需 matplotlib 可选依赖。


环境证据（environment）包括：Python、PyTorch、NumPy 版本、主机平台、可读取时的 Git SHA 及实际 device；run_manifest 还记录 trainable_parameters。论文复杂度 O=E×T×Q 属于理论模型，不能把吞吐率误写成完全同等硬件上的 FLOP/s。
