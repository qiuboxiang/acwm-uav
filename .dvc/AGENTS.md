# DVC Metadata Instructions

- Track DVC metadata, never cache or local credential configuration.
- Shared remotes are configured per project; credentials stay in `.dvc/config.local`
  or the environment and are never committed.
