# Stage gates and research backlog

| Stage | Scope | Acceptance | State |
| --- | --- | --- | --- |
| 00 | Paper and code evidence map, 1301 vs 1310 separation | Primary section/table ↔ code ↔ test crosswalk | Implemented |
| 01 | Model and optimizer factory entrypoints | Engine instantiates models and optimizers using registries | Implemented |
| 02 | Controlled OFAT/factorial planning | Stable configs, validation, seed pairs, resource cap, dry-run | Implemented |
| 03 | Source-tagged metrics, OOV and runtime | Valid denominators, CSV/JSON/Markdown, per-case events | Implemented |
| 04 | Visual reporting | Actual-run quality/cost, dim/quality, OOV coverage plots | Implemented, optional Matplotlib |
| 05 | Independent smoke and CI | Tests for planner, categories, window policy, sampler exponent | Implemented; CI result required |
| 06 | Large-corpus replication | Original data provenance and permission, varied corpus sizes | Requires external corpus |
| 07 | Variance/uncertainty | 3–5+ matched seeds, paired confidence intervals, interaction effect models | Future |
| 08 | Paper §§4.5, 5 | Microsoft sentence completion / multi-example relation / outliers | Future independent feature |
| 09 | True large-scale engineering | Benchmarked efficient multi-worker training, PyTorch profiler, memory/energy | Future |

Avoid interpreting Stage 06–09 as delivered. No original Google News accuracy
or DistBelief speed benchmark is claimed until matching datasets, test partitions,
compute and experimental settings can be verified.
