# Configuration Instructions

- Configuration is declarative and must remain serializable by OmegaConf/YAML.
- Prefer composition and small overrides over copied experiment files.
- Never embed secrets, absolute user paths, or unregistered artifact paths.
- Training configs must identify datasets by registry ID and declare all seeds.
