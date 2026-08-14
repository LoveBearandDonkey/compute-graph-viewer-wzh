# AscendPort example_mla_decode.py MLA Validation

## Scope

- Source: the exact `const CUDA` payload bundled in `ascendport_migration_MLA_A3_updated.zip` / `ascendport_migration/ascendport_migration_V3_MLA_pto_legacy.js`.
- Extracted mirror: `/Users/yin/pto/ascendport_migration-pangu/mla-model-architecture/outputs/example_mla_decode.py`.
- SHA-256: `badfb74e161a19be4b8022b8c13ca059b368598c7d7f8a5b77ea623347341962`.
- Coverage: `flashattn`, `main_split`, `main_no_split`, and the second-stage split combine kernel.
- Default path: `num_split=1`; the conditional split branch is retained but not presented as active by default.
- This is source architecture plus migration association data, not a profiling trace.

## Extraction result

- Canonical nodes: 29.
- Tensor/state edges: 42.
- Operator mappings: 18.
- Mapping implementation states: `emitted_no_split_only`=1, `emitted_s6`=13, `planned_not_emitted`=3, `prototype_divergence`=1.

## Pattern layout contract

- The execution spine is strictly top-to-bottom: dispatch → staging → QK/PE score → online softmax → P·V → normalize → store → output.
- Only the Q/KV input staging and conditional split-KV path occupy side lanes.
- The graph has three hierarchy levels: MLA Decode → major stages → operator sublayers.
- Depth-two sublayers are folded by default and are reprojected as parent nodes; expanding them restores source operators and derives parent bounds from visible children.
- Mapping selection expands any collapsed ancestor before focusing the associated operator.

## Operator association rules

- Every mapped graph node carries `attrs.mapping_ids`.
- Full source/target relationships live in `operator_mapping.json`.
- Mapping rows distinguish direct semantic mapping, memory/pipeline rewrites, many-to-one fusion, removed GPU scheduling concepts, and deferred split-KV work.
- Source and target evidence retain line references to the extracted project payloads.

## Important findings

1. `T.gemm(Q, KV)` and `T.gemm(Q_pe, K_pe)` become one fused target `Mmad` over `DIM + PE_DIM` in the S6 prototype.
2. `T.exp2` plus `log2(e)` must become natural `Exp`/`expf`; this is a numeric rewrite, not a simple API rename.
3. `T.use_swizzle` and `GemmWarpPolicy.FullCol` are removed because their GPU warp scheduling model has no direct Ascend equivalent.
4. S2 maps P·V to Cube `Mmad`, but the bundled S6 prototype emits Vector `Axpy`; the mapping is marked `prototype_divergence`.
5. Split-KV GM workspaces and the second combine kernel are present in the source architecture but are `planned_not_emitted` in S6.

## Source consistency warning

The top-level working-tree `ascendport_migration_V3_MLA_pto_legacy.js` starts with an unrelated FlashAttention V2 Triton payload. The project archive contains the TileLang `example_mla_decode.py` described by the migration analysis, so the extractor reads the archive member directly and verifies required source primitives before emitting artifacts.
