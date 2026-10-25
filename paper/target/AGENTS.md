# Venue Target Instructions

- Treat `venue-research.prompt.md` as the required workflow whenever a user asks
  to initialize, refresh, inspect, or install a submission target.
- The agent, not a filename regex, owns discovery: browse current official
  venue, submission-system, and template sources on every refresh.
- Rules and adapters are guardrails for schema validation, archive safety,
  hashing, and regression tests; do not treat them as authoritative evidence.
- Preserve the exact rule text, timezone, source URL, retrieval time, redirect
  target, and content hash needed for an independent audit.
- Never silently fall back to a previous year's template or an unofficial
  mirror. Stop with `awaiting_official_template` when official evidence is
  insufficient.
- Download candidates only to a temporary directory. Validate checksum and ZIP
  paths before writing under `paper/`.
- Always generate and inspect `template-diff.md` before explicit installation.
- Do not install large local environments merely to validate a template. Use
  the fixed paper container/CI when available and otherwise record the missing
  compiler dependency.
