# Deterministic training plans

Word2Vec training uses randomized context windows, subsampling, and negative
sampling. These operations affect different parts of an experiment and must not
share one mutable random-number stream.

## RNG topology

The trainer derives independent NumPy generators from the configured `SEED`:

| Stream | Purpose | Lifetime |
| --- | --- | --- |
| 1 | frequent-word subsampling | corpus encoding |
| 2 | negative sampling | full training run |
| 3 + epoch | context-window sampling | recreated per epoch |

The context-window generator is replayable. Consuming more negative samples
therefore cannot change which positive context pairs are generated.

The project still seeds Python, legacy NumPy, and PyTorch global RNGs for model
initialization and compatibility. A fixed seed does not promise bit-for-bit
identity across PyTorch versions, devices, or kernels.

## Exact epoch planning

Before optimization begins, `build_epoch_plans()` calculates:

- positive training examples for every epoch;
- optimizer steps for every epoch as `ceil(examples / batch_size)`;
- the exact total optimizer-step count supplied to the linear LR scheduler.

CBOW emits one example per retained target token. Skip-gram example counts depend
on randomized window widths, so the planner uses the same per-epoch seed as the
training iterator and replays that window stream.

The trainer asserts after every epoch that actual optimizer steps equal the
plan. A mismatch is treated as a correctness failure instead of silently
changing the LR schedule.

## Run manifest

Every successful training run writes `run_manifest.json` beside the vectors.
It records:

- the fully resolved dataclass configuration;
- vocabulary and corpus token counts;
- selected device;
- RNG stream assignments;
- per-epoch example/step plans;
- planned and actual optimizer steps;
- final learning rate.

This artifact is intended to make ablation results auditable without parsing
logs.
