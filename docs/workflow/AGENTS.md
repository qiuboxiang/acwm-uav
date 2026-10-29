# Workflow Documentation Instructions

- Explain the safe, supported path rather than undocumented shortcuts.
- Include prerequisites, expected outputs, and recovery steps for external sync.
- Keep credential examples as environment variable names only.
- Agents maintain these documents: whenever code, data layout, dependencies, scenes, weights, seeds or default flags change,
  update the matching page in the same change. The goal is that code and data can be reproduced from the documentation alone.
- For every external asset record its source, the version or commit used, where it goes, and a command that checks it is in place.
- For every generated result record the command, the seeds and the output location, and say whether it is a required input or can be regenerated.
- Run the documented commands before declaring them correct, and say plainly what was not verified.
- Write the pages in Simplified Chinese; keep paths relative to the repository root.
