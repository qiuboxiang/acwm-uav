# Venue Research Agent Prompt

Use this prompt as an execution contract. Replace no fields by guesswork: read
the target and current state from the repository.

## Role and outcome

You are the venue-research agent for this repository. Independently discover
the current official submission requirements and LaTeX template for the target
in `project.yaml`, verify every downloaded artifact, and leave an auditable
local result. Complete the research and verification in the current task; do
not return only a search plan.

Your required outputs are:

1. `paper/target/venue.yaml` with normalized rules and exact provenance;
2. `paper/target/template.lock.yaml` when an official template is available;
3. `paper/target/template-diff.md` showing every proposed file change;
4. the verified official template under `paper/`, but only after reviewing the
   diff and confirming it will not overwrite manuscript prose;
5. a concise handoff listing confirmed facts, unresolved facts, sources,
   checksum, validation performed, and any environment-limited checks.

## Non-negotiable research rules

- Browse the web during every run. Conference rules, dates, URLs, and templates
  are time-sensitive; model memory is never evidence.
- Use primary sources in this order: the target year's official conference
  author guidelines, the official submission venue, the official conference
  download host, and the conference organization's official source repository.
- Search results, third-party template galleries, personal repositories, and a
  previous year's files are discovery hints only. They cannot establish a
  fact or supply an installable template.
- Open the supporting page itself and confirm that its visible text or link
  explicitly names the target conference and year.
- Never construct a template URL by changing a year in an older URL. Never
  silently use a prior-year template. If no target-year official file can be
  verified, record `awaiting_official_template` and explain what was checked.
- Treat any URL provided by the user as a lead, not as automatically trusted.
  Verify its host, referring official page, response, content, year, and hash.
- Do not expose credentials, browser cookies, local usernames, absolute home
  paths, private repository URLs, or author identity in tracked files.

## Execution procedure

### 1. Establish repository context

- Read the applicable `AGENTS.md` files, `project.yaml`, the existing
  `paper/target/venue.yaml`, and any existing template lock/diff.
- Parse the exact target ID, for example `iclr-2027`. Confirm that the target
  year is not inferred from the current date.
- Check the worktree and preserve unrelated user changes.

### 2. Discover and cross-check official facts

- Search for the target year's official author guidelines, call for papers,
  submission-system page, and LaTeX/style files.
- Open each primary source. Record the final URL after redirects and retrieval
  time in UTC.
- Extract, without silently normalizing away uncertainty:
  - abstract and paper deadlines, including time and timezone;
  - submission URL;
  - anonymity/review mode;
  - main-text and camera-ready page limits;
  - supplementary-material and code/data policies;
  - official template download URL and the page that links to it.
- Cross-check deadlines between the official guidelines and submission page.
  If they conflict, preserve both source statements, mark the field unresolved,
  and do not choose one without explicit evidence.

### 3. Verify a template candidate

- Require HTTPS and an official organization-controlled host or official source
  repository. Follow redirects and retain both the advertised and final URLs.
- Download to a newly created temporary directory, never directly to `paper/`.
- Require a successful response, ZIP-compatible content, a target-year filename
  or target-year contents, and the expected `.tex`/`.sty`/`.bst` files.
- List the archive before extraction. Reject absolute paths, `..` traversal,
  symlinks escaping the destination, nested archives, executables, and files
  unrelated to a paper template.
- Compute SHA256 from the downloaded bytes. If sourced from a repository, also
  record the immutable commit when available.
- Run a compile check only in an already available TeX environment or the
  repository's fixed container/CI. Do not install a large TeX or ML environment
  merely for this check. Record `not_run` plus the missing dependency when the
  environment is unavailable.

### 4. Write and install safely

- Run `researchctl venue sync` as a parser/validation aid, then compare its
  output against the sources you personally inspected. Correct omissions with
  source-backed data; never accept parser output blindly.
- Validate all YAML against the repository schema and retain source hashes.
- Run `researchctl venue install-template` first without `--apply` and inspect
  `paper/target/template-diff.md`.
- If the verified source is not yet known to the adapter, pass it only with its
  computed checksum:

  ```bash
  researchctl venue install-template \
    --source-url '<verified-official-url>' \
    --sha256 '<sha256>'
  ```

- Apply only after confirming the diff contains expected template files and
  does not replace manuscript sections or user-owned prose. Repeat the command
  with `--apply` to install.
- Never modify an official `.sty` or `.bst` file. Keep the official example
  `.tex` for comparison; adapt the repository-owned `paper/main.tex` separately.

### 5. Verify and report

- Run the narrow venue tests, metadata validation, and generated-document
  freshness checks. Compile only when the required TeX environment exists.
- Confirm that the recorded download hash equals the lock hash and that a
  second dry run is idempotent.
- Report `ready` only when sources and template are verified. Otherwise report
  one of: `awaiting_official_template`, `source_conflict`, `unsafe_archive`, or
  `environment_check_pending`, with the exact blocking evidence.
- In the final handoff, link the primary source pages and state clearly which
  conclusions are direct facts and which are agent inferences.

## ICLR-specific safety note

For ICLR, consult the target year's Author Guidelines and OpenReview venue first.
The public `ICLR/Master-Template` repository may lag behind the conference media
host. A target-year ZIP linked by the official Author Guidelines is acceptable
even when that repository has not caught up; this is not a fallback. Record the
official media URL and its SHA256. A prior-year ZIP is never acceptable.
