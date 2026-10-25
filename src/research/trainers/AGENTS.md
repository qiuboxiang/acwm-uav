# Trainer Instructions

- Trainers receive models and data; registry writes belong to `researchctl`.
- Save `best` and `last` checkpoints with explicit selection metric semantics.
- Distributed metrics must be globally reduced and only rank zero writes files.
- Resume behavior must restore model and optimizer state or document limitations.
