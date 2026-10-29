# Heatmap Archive Instructions

- One folder per collection version (`v4`, `v5`, ...), written by `src/research/exploration/heatmap_v5.py --tag <version>`.
- Every new collection version gets its own folder; never overwrite or hand-edit an existing one. A deliberate rebuild uses `--force` and must be mentioned in the report.
- `meta.json` records the source data folder, replay checksums, policy string and git head; keep it with the data.
