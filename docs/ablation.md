# Ablations

Run ablations with a grid file. See `configs/ablation_grid.yaml`.

```yaml
grid:
  MODEL.arch: ["skipgram", "cbow"]
  MODEL.loss: ["hs", "ns"]
  MODEL.dim: [100, 300, 600]
  MODEL.window: [5, 10]
  TRAIN.epochs: [5, 10]
  DATA.subsample_t: [null, 1e-5]
base_cfg: configs/base.yaml
out_root: runs/ablations
```

Run the grid (the cartesian product of every choice):

```bash
python -m w2v_factory.cli.ablate --grid configs/ablation_grid.yaml
```

Each combination runs the full training pipeline and writes vectors to:

```
runs/ablations/MODEL_arch=skipgram__MODEL_loss=hs__.../embeddings.txt
```

The folder name encodes every varied key, so it is easy to compare runs.

## Tips

- Keep the base cfg fixed; point `base_cfg` at the config that sets `DATA.input_files`.
- Set `DATA.subsample_t` to `null` in the grid when you want to disable it.
- Evaluate each produced `embeddings.txt` with `scripts/eval_analogy.sh`.