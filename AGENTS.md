# Repository Instructions

## Mission

Maintain a reproducible chain from dataset versions through runs, checkpoints,
evaluations, paper evidence, and releases. Registry IDs and content hashes are
the public interface between subsystems.

## Non-negotiable rules

- Never commit raw datasets, model weights, credentials, local absolute paths,
  MLflow stores, or Overleaf tokens.
- Use `researchctl` to create or finalize registry records. Finalized records
  are immutable; corrections create a new record that references the old one.
- Resolve datasets, checkpoints, and evaluations by registry ID, never by an
  undocumented filesystem path.
- Update `paper/evidence.yaml` whenever a paper figure, table, or quantitative
  claim changes.
- Do not hand-edit `docs/generated/` or `releases/current.yaml`; regenerate them.
- Overleaf is the authority for prose. Pull before building, freezing, or
  pushing generated assets. Stop on a two-sided conflict.
- A public release requires clean Git state, complete licenses, checksums,
  successful paper build, and explicit confirmation.

## Standard verification

Run the narrowest relevant checks during development, then before handoff run:

```bash
python -m pytest
researchctl validate
researchctl release check
```

Use `uv run` before commands when working in the managed environment.

## Change discipline

- Preserve user-owned experiments and unrelated worktree changes.
- Keep generated files deterministic and include a header identifying their source.
- Add or update tests for schema, synchronization, release, or training behavior.
- Update the Chinese workflow documentation when a user-facing command changes.
- Keep code and data fully reproducible from the documentation. In the same change that adds or alters a script, dependency,
  scene, dataset, weight file, seed, default flag or output location, update `docs/workflow/` (and `README.md` when the
  entry point changes) so a fresh clone can rebuild the result by following it; a change is not done until the docs match.
- Verify documented commands and path checks by running them; state in the docs what could not be verified.

## Code review rules

- Flag any unregistered path used by evaluation or paper generation.
- Flag releases that omit source config, dataset version, checkpoint checksum,
  evaluation record, license, or paper evidence reference.
- Flag secrets, personal identifiers, and machine-specific paths in tracked data.
- Flag changes to code, data layout, dependencies or external assets whose workflow documentation was not updated.
