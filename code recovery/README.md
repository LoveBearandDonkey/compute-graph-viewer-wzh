# PTO Tiling

Workspace for PTO tiling visualization experiments and references.

## References

- https://github.com/Deep-Learning-Profiling-Tools/triton-viz
- https://github.com/gpu-mode/triton-puzzles

## Structure

- `docs/` - notes, PRDs, integration plans, and research summaries.
- `src/` - prototype code for tiling analysis or visualization.
- `data/` - sample traces, tiling metadata, and generated fixtures.
- `assets/` - screenshots, diagrams, and static visual assets.

## Design-System Rule

Any UI built in this module should consume the PTO shared tokens and components first. Do not add a private visual style system here without a preview and explicit approval.
