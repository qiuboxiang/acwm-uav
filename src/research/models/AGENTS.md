# Model Instructions

- Models expose architecture entirely through constructor arguments.
- Do not load datasets, parse CLI arguments, or write checkpoints in a model.
- Keep forward outputs stable and document any checkpoint-incompatible change.
