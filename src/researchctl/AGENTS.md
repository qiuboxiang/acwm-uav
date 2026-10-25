# Research Control Plane Instructions

- Preserve registry immutability, relative paths, checksums, and referential integrity.
- External writes require explicit confirmation and must support dry-run behavior.
- Network parsers store provenance and never substitute an unverified prior-year source.
- Filesystem mutations must be atomic where practical and stop before overwriting conflicts.
