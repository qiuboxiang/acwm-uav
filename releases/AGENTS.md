# Release Instructions

- `current.yaml` is generated and represents a private moving candidate.
- Files under `history/` are immutable once published.
- Freeze only from clean Git state after strict validation and paper build.
- Publishing is always explicit; never make external repositories public by
  inference or as a side effect of validation.
