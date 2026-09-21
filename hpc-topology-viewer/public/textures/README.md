# Chip textures

Drop two image files here and commit them — they are served at
`/<base>/textures/...` and mapped onto the NPU chip in every view (no local /
runtime side-loading needed). If a file is missing, the app falls back to
procedural chip geometry.

- `npu-chip.png` — the NPU package photo (mapped onto the chip top face)
- `logo.png` — the logo (PNG with transparent background; shown on the chip when
  no package photo is present)

Recommended: square-ish PNG, ≤ ~1 MB each.
