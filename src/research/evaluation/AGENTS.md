# Evaluation Runtime Instructions

- Evaluation is read-only with respect to model weights and datasets.
- Load inputs through registered IDs and report deterministic metric reductions.
- Keep compact scalar summaries separate from large predictions or embeddings.
