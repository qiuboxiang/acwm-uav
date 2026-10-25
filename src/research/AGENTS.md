# Research Runtime Instructions

- Keep scientific components composable from explicit configuration.
- Runtime modules may depend on PyTorch; metadata and release modules must not.
- Determinism, distributed reduction, checkpoint semantics, and resume behavior
  require tests whenever they change.
- Return structured results to `researchctl`; do not write registry YAML directly.
