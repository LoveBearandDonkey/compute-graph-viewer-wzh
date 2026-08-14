#!/usr/bin/env python3
"""Extract AscendPort's bundled TileLang MLA decode kernel and operator mapping."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ARCHIVE = PROJECT_ROOT / "ascendport_migration_MLA_A3_updated.zip"
ARCHIVE_MEMBER = "ascendport_migration/ascendport_migration_V3_MLA_pto_legacy.js"
ANALYSIS = PROJECT_ROOT / "ascendport_migration_V3_MLA_pto.analysis.md"
PATTERN = Path("/Users/yin/pto-design-system/patterns/model-graphviz/pattern.json")
OUTPUT_DIR = HERE / "outputs"
SOURCE_MIRROR = OUTPUT_DIR / "example_mla_decode.py"
TARGET_S4_MIRROR = OUTPUT_DIR / "flash_mla_decode_s4.cpp"
TARGET_S6_MIRROR = OUTPUT_DIR / "flash_mla_decode_s6.cpp"
SCHEMA_OUTPUT = OUTPUT_DIR / "model_architecture.json"
GRAPH_OUTPUT = OUTPUT_DIR / "model_architecture_graph.json"
MAPPING_OUTPUT = OUTPUT_DIR / "operator_mapping.json"
VALIDATION_OUTPUT = OUTPUT_DIR / "model_architecture_validation.md"
MODEL_VIZ = HERE / "assets" / "modelviz.html"
GRAPH_PROJECTOR = Path("/Users/yin/.codex/skills/model-architecture-extractor/scripts/project_model_architecture_graph.py")


def extract_template(bundle: str, name: str) -> str:
    match = re.search(
        rf"const\s+{re.escape(name)}\s*=\s*String\.raw`(.*?)`;\r?\n",
        bundle,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"String.raw template {name} was not found in {ARCHIVE_MEMBER}")
    return match.group(1).replace("\r\n", "\n")


def load_bundle_sources() -> tuple[str, str, str, str]:
    if not ARCHIVE.exists():
        raise RuntimeError(f"project archive is missing: {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bundle = archive.read(ARCHIVE_MEMBER).decode("utf-8")
    source = extract_template(bundle, "CUDA")
    target_s4 = extract_template(bundle, "S4")
    target_s6 = extract_template(bundle, "S6")
    return bundle, source, target_s4, target_s6


def verify_source(source: str, target_s6: str) -> None:
    required_source = {
        "def flashattn(": "TileLang MLA entry",
        "def main_split(": "split-KV kernel",
        "def main_no_split(": "no-split kernel",
        "T.gemm(Q_shared, KV_shared": "Q·KV transpose GEMM",
        "T.gemm(Q_pe_shared, K_pe_shared": "position GEMM",
        "T.reduce_max(acc_s": "online softmax maximum",
        "T.exp2(acc_s": "source exponential primitive",
        "T.reduce_sum(acc_s": "online softmax sum",
        "T.gemm(S_shared, KV_shared": "probability-value GEMM",
        "if num_split > 1:": "split/no-split branch",
        "def ref_program(": "reference implementation",
    }
    required_target = {
        "GetBlockIdx()": "AI Core dispatch",
        "Mmad(logits": "Cube score implementation",
        "ReduceMax(qkScores": "Vector max implementation",
        "Exp(qkScores": "natural exponential rewrite",
        "ReduceSum(qkScores": "Vector sum implementation",
        "Axpy(outAcc": "emitted S6 probability-value accumulation",
    }
    missing = [fact for needle, fact in required_source.items() if needle not in source]
    missing += [fact for needle, fact in required_target.items() if needle not in target_s6]
    if missing:
        raise RuntimeError("source verification failed: " + ", ".join(missing))
    ast.parse(source, filename=str(SOURCE_MIRROR))


def line_of(text: str, needle: str, occurrence: int = 1) -> int:
    seen = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            seen += 1
            if seen == occurrence:
                return line_no
    raise RuntimeError(f"could not locate source fact: {needle}")


def cli_defaults(source: str) -> dict[str, int | float]:
    tree = ast.parse(source, filename=str(SOURCE_MIRROR))
    defaults: dict[str, int | float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default" or not isinstance(keyword.value, ast.Constant):
                continue
            if isinstance(keyword.value.value, (int, float)):
                defaults[first.value.removeprefix("--").replace("-", "_")] = keyword.value.value
    return defaults


def source_provenance(line: int, fact: str, confidence: str = "confirmed") -> list[dict[str, Any]]:
    return [{
        "source": "example_mla_decode",
        "path": str(SOURCE_MIRROR),
        "line": line,
        "fact": fact,
        "confidence": confidence,
    }]


def target_provenance(source_id: str, path: Path, line: int, fact: str) -> dict[str, Any]:
    return {
        "source": source_id,
        "path": str(path),
        "line": line,
        "fact": fact,
        "confidence": "prototype_evidence",
    }


def mapping_record(
    mapping_id: str,
    source: str,
    source_lines: list[int],
    graph_nodes: list[str],
    semantic_role: str,
    target_api: str,
    execution_unit: str,
    relation: str,
    implementation_status: str,
    note: str,
    *,
    target_source: str | None = None,
    target_line: int | None = None,
    target_fact: str | None = None,
    emitted_api: str | None = None,
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "api": target_api,
        "execution_unit": execution_unit,
        "implementation_status": implementation_status,
    }
    if emitted_api:
        target["emitted_api"] = emitted_api
    evidence: list[dict[str, Any]] = []
    if target_source and target_line and target_fact:
        if target_source == "ascend_s4":
            evidence.append(target_provenance(target_source, TARGET_S4_MIRROR, target_line, target_fact))
        elif target_source == "ascend_s6":
            evidence.append(target_provenance(target_source, TARGET_S6_MIRROR, target_line, target_fact))
    return {
        "id": mapping_id,
        "source": {
            "primitive": source,
            "lines": source_lines,
            "provenance": source_provenance(source_lines[0], semantic_role),
        },
        "graph_node_ids": graph_nodes,
        "semantic_role": semantic_role,
        "target": target,
        "relation": relation,
        "note": note,
        "target_evidence": evidence,
    }


def build_mappings(source: str, target_s4: str, target_s6: str) -> list[dict[str, Any]]:
    return [
        mapping_record(
            "map_kernel_dispatch", "T.Kernel(..., threads=256)",
            [line_of(source, "with T.Kernel(heads //")], ["kernel_dispatch"],
            "kernel grid and worker assignment", "GetBlockIdx + AI Core SPMD", "Scalar / scheduler",
            "structural_rewrite", "emitted_s6",
            "GPU thread blocks become AI Core work ownership for each batch/head group.",
            target_source="ascend_s6", target_line=line_of(target_s6, "this->batchIdx = GetBlockIdx()"),
            target_fact="S6 assigns work with GetBlockIdx",
        ),
        mapping_record(
            "map_shared_buffers", "T.alloc_shared", [line_of(source, "Q_shared = T.alloc_shared")],
            ["q_stage", "kv_stage"], "shared-memory staging", "TQue / T.alloc_L1 / T.alloc_ub", "MTE + L1/UB",
            "memory_space_rewrite", "emitted_s6",
            "Source shared buffers are split by consumer into L1 queues and Vector UB buffers.",
            target_source="ascend_s6", target_line=line_of(target_s6, "pipe.InitBuffer(qL1"),
            target_fact="S6 creates explicit L1 and UB buffers",
        ),
        mapping_record(
            "map_fragment_buffers", "T.alloc_fragment", [line_of(source, "acc_s = T.alloc_fragment")],
            ["qk_gemm", "pe_gemm", "score_block_max", "pv_gemm"], "register-fragment accumulators",
            "L0C LocalTensor + VECCALC buffer", "Cube L0C + Vector UB", "memory_space_rewrite", "emitted_s6",
            "Accumulator placement becomes explicit and follows the Cube/Vector split.",
            target_source="ascend_s6", target_line=line_of(target_s6, "LocalTensor<float> logits = cO.AllocTensor"),
            target_fact="S6 allocates logits in the Cube output queue",
        ),
        mapping_record(
            "map_q_copy", "T.copy(Q / Q_pe -> shared)",
            [line_of(source, "T.copy(Q[bid", occurrence=1), line_of(source, "T.copy(Q_pe[bid", occurrence=1), line_of(source, "T.copy(Q[bid", occurrence=2), line_of(source, "T.copy(Q_pe[bid", occurrence=2)],
            ["q_stage"], "query block load", "DataCopy GM -> L1 -> L0A", "MTE",
            "staged_direct_mapping", "emitted_s6", "Q and Q_pe are staged together before score computation.",
            target_source="ascend_s6", target_line=line_of(target_s6, "DataCopy(qLoc,"),
            target_fact="S6 copies Q and Q_pe into qL1",
        ),
        mapping_record(
            "map_kv_pipeline_copy", "T.Pipelined + T.copy(KV / K_pe)",
            [line_of(source, "for k in T.Pipelined(loop_range", occurrence=1), line_of(source, "T.copy(KV[bid", occurrence=1), line_of(source, "T.copy(K_pe[bid", occurrence=1), line_of(source, "for k in T.Pipelined(loop_range", occurrence=2), line_of(source, "T.copy(KV[bid", occurrence=2), line_of(source, "T.copy(K_pe[bid", occurrence=2)],
            ["kv_stage"], "KV tile load and overlap", "DataCopy GM -> L1 -> L0B + double buffer", "MTE",
            "pipeline_rewrite", "emitted_s6", "The source two-stage pipeline becomes TQue depth-2 prefetch of the next KV tile.",
            target_source="ascend_s6", target_line=line_of(target_s6, "CopyInKV(tile + 1)"),
            target_fact="S6 overlaps next-tile prefetch with compute",
        ),
        mapping_record(
            "map_qk_gemm", "T.gemm(Q_shared, KV_shared, transpose_B=True, clear_accum=True)",
            [line_of(source, "T.gemm(Q_shared, KV_shared", occurrence=1), line_of(source, "T.gemm(Q_shared, KV_shared", occurrence=2)], ["qk_gemm"],
            "non-positional score matrix multiply", "Mmad / T.gemm_v0(init=True)", "Cube",
            "many_to_one_fusion", "emitted_s6", "Q·KV transpose GEMM is fused with the PE GEMM by concatenating DIM and PE_DIM.",
            target_source="ascend_s6", target_line=line_of(target_s6, "Mmad(logits"),
            target_fact="S6 emits one fused Mmad over DIM + PE_DIM",
        ),
        mapping_record(
            "map_pe_gemm", "T.gemm(Q_pe_shared, K_pe_shared, transpose_B=True)",
            [line_of(source, "T.gemm(Q_pe_shared, K_pe_shared", occurrence=1), line_of(source, "T.gemm(Q_pe_shared, K_pe_shared", occurrence=2)], ["pe_gemm"],
            "positional score accumulation", "Mmad / T.gemm_v0(init=False)", "Cube",
            "many_to_one_fusion", "emitted_s6", "The second source GEMM is absorbed into the same concatenated Mmad as Q·KV.",
            target_source="ascend_s6", target_line=line_of(target_s6, "Mmad(logits"),
            target_fact="S6 fuses both score terms into one Mmad",
        ),
        mapping_record(
            "map_warp_policy", "T.GemmWarpPolicy.FullCol",
            [line_of(source, "policy=T.GemmWarpPolicy.FullCol", occurrence=1), line_of(source, "policy=T.GemmWarpPolicy.FullCol", occurrence=2), line_of(source, "policy=T.GemmWarpPolicy.FullCol", occurrence=3), line_of(source, "policy=T.GemmWarpPolicy.FullCol", occurrence=4), line_of(source, "policy=T.GemmWarpPolicy.FullCol", occurrence=5), line_of(source, "policy=T.GemmWarpPolicy.FullCol", occurrence=6)],
            ["qk_gemm", "pe_gemm", "pv_gemm"], "GPU warp-level GEMM policy",
            "removed; Cube manages L0 tiling", "Cube scheduler", "removed_with_replacement", "emitted_s6",
            "The GPU warp partition has no Ascend equivalent and must not survive code generation.",
            target_source="ascend_s6", target_line=line_of(target_s6, "GemmWarpPolicy 在昇腾无对应物"),
            target_fact="S6 explicitly removes GemmWarpPolicy",
        ),
        mapping_record(
            "map_swizzle", "T.use_swizzle(10)", [line_of(source, "T.use_swizzle(10)")],
            ["kernel_dispatch"], "GPU L2 swizzle schedule", "T.Persistent / AI Core work balancing", "Scheduler",
            "removed_with_replacement", "planned_not_emitted",
            "The analysis prescribes T.Persistent, but the current S6 prototype does not emit that directive.",
        ),
        mapping_record(
            "map_reduce_max", "T.reduce_max + T.max", [line_of(source, "T.reduce_max(acc_s", occurrence=1), line_of(source, "T.reduce_max(acc_s", occurrence=2)],
            ["score_block_max", "running_max_merge"], "online softmax maximum", "ReduceMax + fmaxf", "Vector",
            "semantic_direct_mapping", "emitted_s6", "Block max and historical max remain separate semantic steps.",
            target_source="ascend_s6", target_line=line_of(target_s6, "float mCurr = ReduceMax"),
            target_fact="S6 performs Vector ReduceMax and running maximum merge",
        ),
        mapping_record(
            "map_exp2", "T.exp2(x * log2(e))", [line_of(source, "scores_scale[i] = T.exp2", occurrence=1), line_of(source, "acc_s[i, j] = T.exp2", occurrence=1), line_of(source, "lse_logsum_local += T.exp2"), line_of(source, "scale_local = T.exp2"), line_of(source, "scores_scale[i] = T.exp2", occurrence=2), line_of(source, "acc_s[i, j] = T.exp2", occurrence=2)],
            ["score_exponential", "output_rescale"], "online softmax exponentials", "Exp / expf with natural softmax scale", "Vector",
            "numeric_rewrite", "emitted_s6", "Remove the 1.44269504 multiplier and keep LSE math in natural exp/ln space.",
            target_source="ascend_s6", target_line=line_of(target_s6, "Exp(qkScores"),
            target_fact="S6 emits natural Exp rather than exp2",
        ),
        mapping_record(
            "map_reduce_sum", "T.reduce_sum", [line_of(source, "T.reduce_sum(acc_s", occurrence=1), line_of(source, "T.reduce_sum(acc_s", occurrence=2)],
            ["score_sum"], "online softmax denominator reduction", "ReduceSum", "Vector",
            "semantic_direct_mapping", "emitted_s6", "The reduction axis changes from source dim=1 to the local tile's final dimension.",
            target_source="ascend_s6", target_line=line_of(target_s6, "float localSum = ReduceSum"),
            target_fact="S6 emits Vector ReduceSum",
        ),
        mapping_record(
            "map_online_state", "T.Parallel logsum / acc_o rescale", [line_of(source, "logsum[i] = logsum[i] * scores_scale", occurrence=1), line_of(source, "acc_o[i, j] *= scores_scale", occurrence=1), line_of(source, "logsum[i] = logsum[i] * scores_scale", occurrence=2), line_of(source, "acc_o[i, j] *= scores_scale", occurrence=2)],
            ["online_state_update", "output_rescale"], "online softmax recurrence", "Muls + scalar FP32 recurrence", "Vector + Scalar",
            "loop_body_rewrite", "emitted_s6", "FP32 running max, denominator, and output rescale preserve the online-softmax invariant.",
            target_source="ascend_s6", target_line=line_of(target_s6, "float lNew = lPrev * alpha + localSum"),
            target_fact="S6 updates the online-softmax state",
        ),
        mapping_record(
            "map_pv_gemm", "T.gemm(P, KV, acc_o)", [line_of(source, "T.gemm(acc_s_cast, KV_shared"), line_of(source, "T.gemm(S_shared, KV_shared")],
            ["pv_gemm"], "probability-value accumulation", "Mmad / Cube GEMM (S2 plan)", "Cube planned; Vector emitted",
            "semantic_mapping_with_codegen_gap", "prototype_divergence",
            "S2 maps P·V to Cube Mmad, while the bundled S6 prototype currently emits per-token Vector Axpy.",
            target_source="ascend_s6", target_line=line_of(target_s6, "Axpy(outAcc"),
            target_fact="S6 prototype implements P·V with Axpy", emitted_api="Axpy",
        ),
        mapping_record(
            "map_normalize", "acc_o /= logsum", [line_of(source, "acc_o[i, j] /= logsum", occurrence=1), line_of(source, "acc_o[i, j] /= logsum", occurrence=2)],
            ["output_normalize"], "final no-split normalization", "Div", "Vector",
            "semantic_direct_mapping", "emitted_s6", "The accumulated output is divided by the final online-softmax denominator.",
            target_source="ascend_s6", target_line=line_of(target_s6, "Div(outAcc"),
            target_fact="S6 normalizes output with Vector Div",
        ),
        mapping_record(
            "map_split_workspace", "T.alloc_global(glse / Output_partial)",
            [line_of(source, "glse = T.alloc_global"), line_of(source, "Output_partial = T.alloc_global")],
            ["partial_result_store", "partial_output_workspace", "split_lse_workspace"],
            "split-KV partial result exchange", "GM workspace between AI Cores", "MTE / GM",
            "cross_kernel_workspace", "planned_not_emitted",
            "The bundled S6 prototype intentionally implements num_split=1 and has not emitted the GM split workspace.",
        ),
        mapping_record(
            "map_split_combine", "second T.Kernel + LSE weighted combine", [line_of(source, "with T.Kernel(heads, batch") , 118],
            ["split_lse_max", "split_lse_sum", "partial_weight", "split_output_accumulate"],
            "flash-decoding split merge", "second-stage Vector reduction in natural exp/ln space", "Vector",
            "two_stage_rewrite", "planned_not_emitted",
            "The target needs a second kernel over GM partial output and LSE; this branch is deferred after no-split correctness.",
        ),
        mapping_record(
            "map_output_store", "T.copy(O_shared, Output) / Output[...] = ...",
            [line_of(source, "T.copy(O_shared, Output", occurrence=1), line_of(source, "Output[bz, hid, i] =")],
            ["no_split_store", "split_output_store"], "final output write", "DataCopy UB -> GM", "MTE",
            "staged_direct_mapping", "emitted_no_split_only", "The no-split path is emitted; split output store depends on the deferred combine kernel.",
            target_source="ascend_s6", target_line=line_of(target_s6, "DataCopy(outGm"),
            target_fact="S6 writes normalized output to GM",
        ),
    ]


def make_node(
    node_id: str,
    kind: str,
    label: str,
    line: int,
    fact: str,
    x: int,
    y: int,
    color_key: str,
    *,
    op_type: str | None = None,
    state_type: str | None = None,
    attrs: dict[str, Any] | None = None,
    mapping_ids: list[str] | None = None,
    width: int = 480,
    height: int = 62,
) -> dict[str, Any]:
    merged_attrs = dict(attrs or {})
    if mapping_ids:
        merged_attrs["mapping_ids"] = mapping_ids
    node: dict[str, Any] = {
        "id": node_id,
        "kind": kind,
        "label": label,
        "colorKey": color_key,
        "provenance": source_provenance(line, fact),
        "visual": {"x": x, "y": y, "width": width, "height": height},
    }
    if op_type:
        node["op_type"] = op_type
    if state_type:
        node["state_type"] = state_type
    if merged_attrs:
        node["attrs"] = merged_attrs
    return node


def make_edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    name: str,
    shape: str,
    dtype: str,
    line: int,
    fact: str,
    *,
    constraints: list[str] | None = None,
    edge_type: str = "activation",
    tag: str | None = None,
    tag_position: float | None = None,
    source_anchor: str | dict[str, Any] | None = None,
    target_anchor: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    tensor: dict[str, Any] = {"name": name, "shape": shape, "dtype": dtype}
    if constraints:
        tensor["constraints"] = constraints
    edge = {
        "id": edge_id,
        "source": source_id,
        "target": target_id,
        "attrs": {"flow_type": edge_type},
        "tensor": tensor,
        "provenance": source_provenance(line, fact),
    }
    if tag:
        edge["tag"] = tag
    if tag_position is not None:
        edge["tagPosition"] = tag_position
    if source_anchor is not None:
        edge["sourceAnchor"] = source_anchor
    if target_anchor is not None:
        edge["targetAnchor"] = target_anchor
    return edge


def build_schema(source: str, mappings: list[dict[str, Any]], defaults: dict[str, int | float], source_sha: str) -> dict[str, Any]:
    line = lambda needle, occurrence=1: line_of(source, needle, occurrence)  # noqa: E731
    # The canonical visual contract is a top-to-bottom execution spine. Only
    # the input staging and the conditional split-KV path use side lanes.
    nodes = [
        make_node("query", "input", "Query", 122, "main_no_split Q input", 220, 96, "io:input", attrs={"source_name": "Q"}, width=240, height=52),
        make_node("position_query", "input", "Position Query", 123, "main_no_split Q_pe input", 480, 96, "io:input", attrs={"source_name": "Q_pe"}, width=250, height=52),
        make_node("latent_kv", "input", "Latent KV", 124, "main_no_split KV input", 1040, 96, "io:input", attrs={"source_name": "KV", "dual_role": ["key", "value"]}, width=240, height=52),
        make_node("position_key", "input", "Position Key", 125, "main_no_split K_pe input", 1300, 96, "io:input", attrs={"source_name": "K_pe"}, width=250, height=52),
        make_node("runtime_config", "input", "Runtime Config", 14, "flashattn configuration arguments", 760, 190, "io:input", attrs={"default_path": "num_split=1"}, width=240, height=52),
        make_node("kernel_dispatch", "op", "Kernel Dispatch", 128, "T.Kernel assigns head groups and batches", 760, 300, "sem:comm", op_type="KernelLaunch", attrs={"source_primitives": ["T.Kernel", "T.use_swizzle"]}, mapping_ids=["map_kernel_dispatch", "map_swizzle"], height=64),
        make_node("q_stage", "op", "Query Block Stage", 145, "copy Q and Q_pe to shared memory", 540, 484, "sem:comm", op_type="Copy", attrs={"storage": "shared", "execution_paths": ["main_split", "main_no_split"]}, mapping_ids=["map_shared_buffers", "map_q_copy"], width=360),
        make_node("kv_stage", "op", "KV Tile Stage", 151, "pipelined KV and K_pe tile copies", 980, 484, "sem:comm", op_type="PipelinedCopy", attrs={"storage": "shared", "repeat": "ceildiv(seqlen_kv, block_N)"}, mapping_ids=["map_shared_buffers", "map_kv_pipeline_copy"], width=360),
        make_node("qk_gemm", "op", "Q·KVᵀ GEMM", 155, "non-positional attention score GEMM", 760, 720, "sem:attention", op_type="GEMM", attrs={"transpose_b": True, "clear_accum": True}, mapping_ids=["map_fragment_buffers", "map_qk_gemm", "map_warp_policy"]),
        make_node("pe_gemm", "op", "PE GEMM Accumulate", 156, "position score GEMM accumulates into acc_s", 760, 836, "sem:attention", op_type="GEMM", attrs={"transpose_b": True, "accumulate": True}, mapping_ids=["map_fragment_buffers", "map_pe_gemm", "map_warp_policy"]),
        make_node("score_block_max", "op", "Score Block Max", 159, "reduce maximum over current score tile", 760, 952, "sem:act", op_type="ReduceMax", mapping_ids=["map_fragment_buffers", "map_reduce_max"]),
        make_node("running_max_merge", "op", "Running Max Merge", 161, "merge tile maximum with previous maximum", 760, 1128, "sem:act", op_type="Maximum", mapping_ids=["map_reduce_max"]),
        make_node("score_exponential", "op", "Score Exponential", 165, "exp2-scaled probability numerator", 760, 1244, "sem:act", op_type="Exp2", attrs={"numeric_basis": "base2", "scale_multiplier": 1.44269504}, mapping_ids=["map_exp2"]),
        make_node("score_sum", "op", "Score Sum", 166, "reduce probability numerator over the tile", 760, 1360, "sem:act", op_type="ReduceSum", mapping_ids=["map_reduce_sum"]),
        make_node("online_state_update", "op", "Online Statistics Update", 169, "update logsum with tile sum and rescale", 760, 1476, "sem:act", op_type="OnlineSoftmaxUpdate", attrs={"accum_dtype": "float32"}, mapping_ids=["map_online_state"]),
        make_node("output_rescale", "op", "Output Rescale", 170, "rescale prior output accumulator", 760, 1592, "sem:act", op_type="Multiply", mapping_ids=["map_exp2", "map_online_state"]),
        make_node("pv_gemm", "op", "Probability Value GEMM", 172, "probability matrix times KV values", 760, 1768, "sem:attention", op_type="GEMM", attrs={"source_api": "T.gemm", "value_source": "KV_shared"}, mapping_ids=["map_fragment_buffers", "map_pv_gemm", "map_warp_policy"]),
        make_node("output_normalize", "op", "Output Normalize", 174, "divide accumulated output by logsum", 760, 1960, "sem:act", op_type="Divide", mapping_ids=["map_normalize"]),
        make_node("no_split_store", "op", "No-Split Store", 176, "copy normalized no-split output", 760, 2138, "sem:comm", op_type="Store", attrs={"condition": "num_split == 1"}, mapping_ids=["map_output_store"], width=360, height=60),
        make_node("lse_encode", "op", "LSE Encode", 87, "encode split logsum in base-2 space", 1260, 2138, "sem:act", op_type="Log2", attrs={"condition": "num_split > 1"}, mapping_ids=["map_exp2", "map_split_workspace"], width=360, height=60),
        make_node("partial_result_store", "op", "Partial Result Store", 88, "store split LSE and partial output", 1260, 2254, "sem:comm", op_type="Store", attrs={"condition": "num_split > 1"}, mapping_ids=["map_split_workspace"], width=360, height=60),
        make_node("partial_output_workspace", "state", "Partial Output Workspace", 31, "global partial output exchanged across kernels", 1640, 2254, "io:state", state_type="workspace", attrs={"persistent_across_kernels": True}, mapping_ids=["map_split_workspace"], width=260, height=54),
        make_node("split_lse_workspace", "state", "Split LSE Workspace", 30, "global split LSE exchanged across kernels", 1640, 2370, "io:state", state_type="workspace", attrs={"persistent_across_kernels": True}, mapping_ids=["map_split_workspace"], width=250, height=54),
        make_node("split_lse_max", "op", "Split LSE Max", 105, "maximum across split LSE values", 1260, 2370, "sem:act", op_type="ReduceMax", mapping_ids=["map_split_combine"], width=360, height=60),
        make_node("split_lse_sum", "op", "Split LSE Sum", 108, "sum normalized split LSE exponentials", 1260, 2486, "sem:act", op_type="ReduceSum", mapping_ids=["map_split_combine", "map_exp2"], width=360, height=60),
        make_node("partial_weight", "op", "Partial Weight", 114, "compute per-split output weight", 1260, 2602, "sem:act", op_type="Exp2", mapping_ids=["map_split_combine", "map_exp2"], width=360, height=60),
        make_node("split_output_accumulate", "op", "Split Output Accumulate", 116, "weighted sum of split partial outputs", 1260, 2718, "sem:act", op_type="WeightedSum", mapping_ids=["map_split_combine"], width=360, height=60),
        make_node("split_output_store", "op", "Split Output Store", 118, "store combined split output", 1260, 2834, "sem:comm", op_type="Store", attrs={"condition": "num_split > 1"}, mapping_ids=["map_output_store"], width=360, height=60),
        make_node("mla_output", "output", "MLA Output", 176, "final attention output", 760, 3030, "io:output", width=240, height=54),
    ]
    visual_nodes = {node["id"]: node.pop("visual") for node in nodes}
    edges = [
        make_edge("e_cfg_dispatch", "runtime_config", "kernel_dispatch", "launch_config", "{B,H,H_KV,N,D,D_PE,BLOCK_N,BLOCK_H,SPLIT}", "control", 14, "flashattn compile-time and runtime arguments", source_anchor="bottom", target_anchor="top"),
        make_edge("e_q_stage", "query", "q_stage", "Q", "[B,H,D]", "float16", 145, "query block input", source_anchor="bottom", target_anchor="top"),
        make_edge("e_qpe_stage", "position_query", "q_stage", "Q_pe", "[B,H,D_PE]", "float16", 146, "position-query block input", source_anchor="bottom", target_anchor="top"),
        make_edge("e_kv_stage", "latent_kv", "kv_stage", "KV_tile", "[B,BLOCK_N,H_KV,D]", "float16", 153, "latent KV tile input", constraints=["H_KV=1"], source_anchor="bottom", target_anchor="top"),
        make_edge("e_kpe_stage", "position_key", "kv_stage", "K_pe_tile", "[B,BLOCK_N,H_KV,D_PE]", "float16", 154, "position-key tile input", constraints=["H_KV=1"], source_anchor="bottom", target_anchor="top"),
        make_edge("e_dispatch_q", "kernel_dispatch", "q_stage", "program_ids", "[head_group,batch]", "int32", 128, "kernel ownership for query load", source_anchor="bottom", target_anchor="top"),
        make_edge("e_dispatch_kv", "kernel_dispatch", "kv_stage", "tile_schedule", "[ceildiv(N,BLOCK_N)]", "int32", 151, "pipelined KV tile schedule", source_anchor="bottom", target_anchor="top"),
        make_edge("e_q_qk", "q_stage", "qk_gemm", "Q_shared", "[BLOCK_H,D]", "float16", 155, "staged non-position query"),
        make_edge("e_kv_qk", "kv_stage", "qk_gemm", "KV_shared", "[BLOCK_N,D]", "float16", 155, "staged latent key/value tile"),
        make_edge("e_qk_pe_acc", "qk_gemm", "pe_gemm", "acc_s", "[BLOCK_H,BLOCK_N]", "float32", 156, "non-position score accumulator"),
        make_edge("e_qpe_pe", "q_stage", "pe_gemm", "Q_pe_shared", "[BLOCK_H,D_PE]", "float16", 156, "staged position query"),
        make_edge("e_kpe_pe", "kv_stage", "pe_gemm", "K_pe_shared", "[BLOCK_N,D_PE]", "float16", 156, "staged position key tile"),
        make_edge("e_pe_max", "pe_gemm", "score_block_max", "acc_s", "[BLOCK_H,BLOCK_N]", "float32", 159, "complete MLA score tile"),
        make_edge("e_max_merge", "score_block_max", "running_max_merge", "scores_max_tile", "[BLOCK_H]", "float32", 161, "current tile maximum"),
        make_edge("e_scores_exp", "pe_gemm", "score_exponential", "acc_s", "[BLOCK_H,BLOCK_N]", "float32", 165, "score tile before exponentiation"),
        make_edge("e_max_exp", "running_max_merge", "score_exponential", "scores_max", "[BLOCK_H]", "float32", 165, "running score maximum"),
        make_edge("e_exp_sum", "score_exponential", "score_sum", "probability_numerator", "[BLOCK_H,BLOCK_N]", "float32", 166, "unnormalized tile probabilities"),
        make_edge("e_sum_state", "score_sum", "online_state_update", "scores_sum", "[BLOCK_H]", "float32", 169, "tile probability sum"),
        make_edge("e_max_state", "running_max_merge", "online_state_update", "scores_scale", "[BLOCK_H]", "float32", 169, "historical softmax rescale"),
        make_edge("e_state_rescale", "online_state_update", "output_rescale", "scores_scale", "[BLOCK_H]", "float32", 170, "output rescale factor"),
        make_edge("e_exp_pv", "score_exponential", "pv_gemm", "probability_tile", "[BLOCK_H,BLOCK_N]", "float16/float32", 172, "tile probabilities"),
        make_edge("e_kv_pv", "kv_stage", "pv_gemm", "value_tile", "[BLOCK_N,D]", "float16", 172, "KV reused as value matrix"),
        make_edge("e_rescale_pv", "output_rescale", "pv_gemm", "rescaled_acc_o", "[BLOCK_H,D]", "float32", 172, "historical output accumulator"),
        make_edge("e_pv_norm", "pv_gemm", "output_normalize", "acc_o", "[BLOCK_H,D]", "float32", 174, "accumulated value output"),
        make_edge("e_state_norm", "online_state_update", "output_normalize", "logsum", "[BLOCK_H]", "float32", 174, "final online-softmax denominator"),
        make_edge("e_norm_no_store", "output_normalize", "no_split_store", "normalized_output", "[B,H,D]", "float16/float32", 176, "no-split normalized output", constraints=["num_split=1"], tag="num_split = 1", tag_position=0.62),
        make_edge("e_no_output", "no_split_store", "mla_output", "Output", "[B,H,D]", "float16", 176, "no-split output", constraints=["num_split=1"]),
        make_edge("e_state_lse", "online_state_update", "lse_encode", "logsum", "[B,H,SPLIT]", "float32", 87, "split denominator state", constraints=["num_split>1"], tag="num_split > 1", tag_position=0.72),
        make_edge("e_max_lse", "running_max_merge", "lse_encode", "scores_max", "[B,H,SPLIT]", "float32", 87, "split maximum state", constraints=["num_split>1"]),
        make_edge("e_norm_partial", "output_normalize", "partial_result_store", "partial_output", "[B,H,SPLIT,D]", "float16", 90, "normalized per-split output", constraints=["num_split>1"]),
        make_edge("e_lse_partial", "lse_encode", "partial_result_store", "split_lse", "[B,H,SPLIT]", "float16", 88, "encoded per-split LSE", constraints=["num_split>1"]),
        make_edge("e_partial_workspace", "partial_result_store", "partial_output_workspace", "Output_partial", "[B,H,SPLIT,D]", "float16", 90, "partial output workspace", edge_type="state"),
        make_edge("e_lse_workspace", "partial_result_store", "split_lse_workspace", "glse", "[B,H,SPLIT]", "float16", 88, "split LSE workspace", edge_type="state"),
        make_edge("e_lse_max", "split_lse_workspace", "split_lse_max", "glse", "[B,H,SPLIT]", "float16", 105, "split LSE values", edge_type="state"),
        make_edge("e_lse_sum_a", "split_lse_max", "split_lse_sum", "lse_max", "[B,H]", "float32", 108, "maximum for stable split reduction"),
        make_edge("e_lse_sum_b", "split_lse_workspace", "split_lse_sum", "glse", "[B,H,SPLIT]", "float16", 108, "split LSE values", edge_type="state"),
        make_edge("e_lse_weight", "split_lse_sum", "partial_weight", "lse_logsum", "[B,H]", "float32", 114, "combined LSE denominator"),
        make_edge("e_lse_weight_src", "split_lse_workspace", "partial_weight", "glse", "[B,H,SPLIT]", "float16", 114, "individual split LSE", edge_type="state"),
        make_edge("e_weight_acc", "partial_weight", "split_output_accumulate", "split_weights", "[B,H,SPLIT]", "float32", 116, "stable partial-output weights"),
        make_edge("e_partial_acc", "partial_output_workspace", "split_output_accumulate", "Output_partial", "[B,H,SPLIT,D]", "float16", 116, "partial outputs", edge_type="state"),
        make_edge("e_acc_split_store", "split_output_accumulate", "split_output_store", "combined_output", "[B,H,D]", "float32", 118, "combined split output"),
        make_edge("e_split_output", "split_output_store", "mla_output", "Output", "[B,H,D]", "float16", 118, "split-KV output", constraints=["num_split>1"]),
    ]
    groups = [
        {"id": "mla_decode_group", "label": "TileLang MLA Decode", "nodes": ["query", "position_query", "latent_kv", "position_key", "runtime_config", "kernel_dispatch", "mla_output"], "children": ["input_stage_group", "tile_loop_group", "finalize_group"], "colorKey": "sem:attention"},
        {"id": "input_stage_group", "label": "Input Staging", "nodes": ["q_stage", "kv_stage"], "children": [], "colorKey": "sem:comm"},
        {"id": "tile_loop_group", "label": "KV Tile Loop", "nodes": [], "children": ["score_compute_group", "online_softmax_group", "value_accumulation_group"], "badge": "repeat ceildiv(N, BLOCK_N)", "colorKey": "sem:attention"},
        {"id": "score_compute_group", "label": "QK + PE Score Compute", "nodes": ["qk_gemm", "pe_gemm", "score_block_max"], "children": [], "collapsed_by_default": True, "colorKey": "sem:attention"},
        {"id": "online_softmax_group", "label": "Online Softmax", "nodes": ["running_max_merge", "score_exponential", "score_sum", "online_state_update", "output_rescale"], "children": [], "collapsed_by_default": True, "colorKey": "sem:act"},
        {"id": "value_accumulation_group", "label": "Probability · Value", "nodes": ["pv_gemm"], "children": [], "collapsed_by_default": True, "colorKey": "sem:attention"},
        {"id": "finalize_group", "label": "Output Finalize", "nodes": ["output_normalize", "no_split_store"], "children": ["split_combine_group"], "colorKey": "sem:act"},
        {"id": "split_combine_group", "label": "Split-KV Combine", "nodes": ["lse_encode", "partial_result_store", "partial_output_workspace", "split_lse_workspace", "split_lse_max", "split_lse_sum", "partial_weight", "split_output_accumulate", "split_output_store"], "children": [], "badge": "conditional num_split > 1", "collapsed_by_default": True, "colorKey": "sem:attention"},
    ]
    mapping_ids = {item["id"] for item in mappings}
    referenced_mapping_ids = {mapping_id for node in nodes for mapping_id in node.get("attrs", {}).get("mapping_ids", [])}
    missing_mapping_ids = referenced_mapping_ids - mapping_ids
    if missing_mapping_ids:
        raise RuntimeError(f"nodes reference missing mappings: {sorted(missing_mapping_ids)}")
    return {
        "schema_version": "model_architecture.v1",
        "model": {"name": "AscendPort example_mla_decode MLA Kernel", "framework": "TileLang", "source_root": str(PROJECT_ROOT)},
        "extraction_scope": {
            "kind": "full_source",
            "full_main_layers": 1,
            "trace_main_layers": None,
            "notes": [
                "Complete source coverage of flashattn, including main_split, main_no_split, and the split combine kernel.",
                "The project default is num_split=1; the split-KV branch is retained as a conditional source branch.",
                "Operator mapping is associated by mapping_ids and delivered in operator_mapping.json.",
            ],
        },
        "sources": [
            {"id": "example_mla_decode", "kind": "source", "path": str(SOURCE_MIRROR), "role": "source_of_truth", "sha256": source_sha, "archive_path": str(ARCHIVE), "archive_member": ARCHIVE_MEMBER, "bundle_symbol": "CUDA"},
            {"id": "ascend_s4", "kind": "generated_source", "path": str(TARGET_S4_MIRROR), "role": "target_mapping_evidence", "bundle_symbol": "S4"},
            {"id": "ascend_s6", "kind": "generated_source", "path": str(TARGET_S6_MIRROR), "role": "target_mapping_evidence", "bundle_symbol": "S6"},
            {"id": "migration_analysis", "kind": "analysis", "path": str(ANALYSIS), "role": "target_mapping_evidence"},
            {"id": "model_graphviz_pattern", "kind": "pattern_contract", "path": str(PATTERN), "role": "render_contract"},
        ],
        "symbol_table": {
            "B": defaults.get("batch", 132), "H": defaults.get("heads", 128), "H_KV": defaults.get("kv_heads", 1),
            "N": defaults.get("kv_ctx", 8192), "D": defaults.get("dim", 512), "D_PE": defaults.get("pe_dim", 64),
            "BLOCK_N": 64, "BLOCK_H": 64, "SPLIT": 1,
        },
        "nodes": nodes,
        "edges": edges,
        "repeats": [{"id": "kv_tile_repeat", "template_node": "kv_stage", "range": "0..ceildiv(N,BLOCK_N)-1", "count": "ceildiv(N,BLOCK_N)", "provenance": source_provenance(152, "KV tile loop") }],
        "branches": [{
            "id": "split_choice", "condition": "num_split > 1", "true_target": "partial_result_store", "false_target": "no_split_store",
            "resolved_ranges": [{"target": "no_split_store", "range": "project default num_split=1"}],
            "provenance": source_provenance(178, "flashattn selects split or no-split kernel"),
        }],
        "visual_layout": {"direction": "vertical", "default_collapse_depth": 2, "nodes": visual_nodes, "groups": groups},
        "operator_mapping_ref": "operator_mapping.json",
        "warnings": [
            "The canonical example_mla_decode.py is embedded in the project archive; the top-level working-tree legacy JS currently contains a stale FlashAttention V2 payload and is not used.",
            "The default execution path hard-codes num_split=1, so split combine is source-complete but not emitted in the bundled S6 target prototype.",
            "S2 plans P·V for Cube Mmad, while S6 currently emits Vector Axpy; operator_mapping.json flags this prototype divergence.",
            "Target mappings are workbench design evidence and have not been validated on Ascend hardware in this artifact.",
        ],
    }


def validation_markdown(source_sha: str, schema: dict[str, Any], mappings: list[dict[str, Any]]) -> str:
    status_counts: dict[str, int] = {}
    for item in mappings:
        status = item["target"]["implementation_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    status_text = ", ".join(f"`{key}`={value}" for key, value in sorted(status_counts.items()))
    return f"""# AscendPort example_mla_decode.py MLA Validation

## Scope

- Source: the exact `const CUDA` payload bundled in `{ARCHIVE.name}` / `{ARCHIVE_MEMBER}`.
- Extracted mirror: `{SOURCE_MIRROR}`.
- SHA-256: `{source_sha}`.
- Coverage: `flashattn`, `main_split`, `main_no_split`, and the second-stage split combine kernel.
- Default path: `num_split=1`; the conditional split branch is retained but not presented as active by default.
- This is source architecture plus migration association data, not a profiling trace.

## Extraction result

- Canonical nodes: {len(schema['nodes'])}.
- Tensor/state edges: {len(schema['edges'])}.
- Operator mappings: {len(mappings)}.
- Mapping implementation states: {status_text}.

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
"""


def embed_modelviz_defaults() -> None:
    """Keep the file:// fallback synchronized with the generated JSON artifacts."""
    html = MODEL_VIZ.read_text(encoding="utf-8")
    documents = {
        "defaultSchemaJson": json.loads(SCHEMA_OUTPUT.read_text(encoding="utf-8")),
        "defaultGraphJson": json.loads(GRAPH_OUTPUT.read_text(encoding="utf-8")),
        "defaultMappingJson": json.loads(MAPPING_OUTPUT.read_text(encoding="utf-8")),
    }
    for element_id, document in documents.items():
        payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        pattern = re.compile(
            rf'(<script\s+type="application/json"\s+id="{re.escape(element_id)}">)(.*?)(</script>)',
            flags=re.DOTALL,
        )
        html, replacements = pattern.subn(rf"\g<1>{payload}\g<3>", html, count=1)
        if replacements != 1:
            raise RuntimeError(f"modelviz fallback slot is missing: {element_id}")
    MODEL_VIZ.write_text(html, encoding="utf-8")


def main() -> None:
    bundle, source, target_s4, target_s6 = load_bundle_sources()
    del bundle
    verify_source(source, target_s6)
    defaults = cli_defaults(source)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    mappings = build_mappings(source, target_s4, target_s6)
    schema = build_schema(source, mappings, defaults, source_sha)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_MIRROR.write_text(source, encoding="utf-8")
    TARGET_S4_MIRROR.write_text(target_s4, encoding="utf-8")
    TARGET_S6_MIRROR.write_text(target_s6, encoding="utf-8")
    SCHEMA_OUTPUT.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mapping_document = {
        "schema_version": "operator_mapping.v1",
        "source": {"name": "example_mla_decode.py", "framework": "TileLang", "sha256": source_sha},
        "target": {"platform": "Atlas A3 / Ascend 910C", "route": "Ascend C & PTO"},
        "mappings": mappings,
    }
    MAPPING_OUTPUT.write_text(json.dumps(mapping_document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, str(GRAPH_PROJECTOR), str(SCHEMA_OUTPUT), "-o", str(GRAPH_OUTPUT)],
        check=True,
    )
    VALIDATION_OUTPUT.write_text(validation_markdown(source_sha, schema, mappings), encoding="utf-8")
    embed_modelviz_defaults()
    print(f"source={SOURCE_MIRROR}")
    print(f"schema={SCHEMA_OUTPUT}")
    print(f"graph={GRAPH_OUTPUT}")
    print(f"mapping={MAPPING_OUTPUT}")
    print(f"nodes={len(schema['nodes'])}, edges={len(schema['edges'])}, mappings={len(mappings)}")


if __name__ == "__main__":
    main()
