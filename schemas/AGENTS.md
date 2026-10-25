# Schema Instructions

- JSON Schema files are generated from Pydantic contracts; never hand-edit them.
- Run `researchctl schema export` after any contract change.
- Backward-incompatible contract changes require a schema version increment and migration.
