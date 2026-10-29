# Exploration Agent Instructions

- Agents must stay sensor-only: the agent side may read only its camera images, its own commands, its own map and the
  operator-given flight-area box. Mesh geometry, ground-truth pose, collision flags and renders of hypothetical views are
  for the simulator side (execution, evaluation) or for training labels only, and must be marked as such.
- `mapping5.py` stays free of simulator imports so it can be unit-tested (`tests/test_exploration_mapping5.py`).
- Result folders (`results*/`, `logs*/`, `../exploration_v5/`) and collected data under `data/` are local outputs; do not commit them.
- Use paths relative to the script, never absolute local paths.
