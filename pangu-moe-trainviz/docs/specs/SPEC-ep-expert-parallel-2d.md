# SPEC · EP Expert Parallel 2D Side View · Pangu Pro MoE 72BA16B

> Target page proposal: `pangu-moe-trainviz/ep-expert-parallel-2d.html`
> Status: data spec, not implementation
> Tutorial source: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/df5d78ad4fed4a8f760d8990c28ae7a3c677c427/rlhf/sys-design/readme-4.md
> Pangu online source: https://arxiv.org/html/2505.21411v2 and https://gitcode.com/ascend-tribe/pangu-pro-moe
> Pangu implementation reference: https://ai.gitcode.com/ascend-tribe/openpangu-ultra-moe-718b-model
> Local reference page: `op-rank-time.html`
> Core rule: reuse existing Pangu MoE 72B data and TrainScope resources; simplify rendering to 2D side view only.

---

## 0. Goal

Build a branch page that explains and diagnoses Expert Parallelism for the existing Pangu MoE training visualization.

The page must not become a generic DeepSeek-only explainer. The tutorial article is used to enrich concepts, formulas, implementation details, and comparison objects, but the concrete model instance is a single Pangu version:

- `Pangu Pro MoE 72BA16B`
- 72B total parameters / 16.50B activated parameters
- 48 transformer layers
- 64 routed experts
- Top-8 activated experts per token
- 4 shared experts
- MoGE group-balanced routing
- training parallel shape `TP8 x EP2 x CP1 x PP5 x VPP5`
- training schedule view uses 50 pipeline slots because the paper appends 2 no-op layers to 48 transformer layers for PP/VPP load balancing

`op-rank-time.html` is a layout/interaction resource, not a second model source. Any local constants that still say `openPangu-R-72B-2512`, `50 decoder layers`, `L0-L3 Dense`, `L4-L49 MoE`, or `80 routed experts` are legacy values to replace with the arXiv v2 Pangu Pro MoE values above.

The view can be 2D. Preferred view: a side projection that keeps the model layer axis, PP stage ranges, EP token dispatch/combine, and rank groups aligned in one coordinate system.

---

## 1. Non-Negotiable Data Rules

1. Use arXiv v2 Pangu Pro MoE 72BA16B constants as the only target model constants. Do not use legacy `op-rank-time.html` values, historical `718B` comments, or `256 experts` labels from the reusable graph builder as target truth.
2. Reuse existing local data/resources where possible:
   - `data/graph-ultramoe-718b.js` as topology template only
   - `data/strict-1f1b-trace-sim.json` for 1F1B schedule
   - `data/timeseries.js` for synthetic training metrics
   - `js/analysis-data.js` for MoE/rank/card metrics
   - `pangu-palette.js` and PTO design tokens for semantic colors
3. Preserve source tutorial metrics and objects as a separate `theory` namespace. Do not silently overwrite Pangu constants with DeepSeek examples.
4. If an object count is large, store the detailed data but render aggregated charts by default. For Pangu Pro: `48 layers x 64 routed experts = 3072 expert cells`, feasible as a heatmap but too dense for labels.
5. Label synthetic data as synthetic or deterministic simulation. Do not present schedule or load metrics as measured profiler traces.
6. EP collective highlighting means "rank participates in token dispatch/combine", not "every highlighted rank is faulty".

---

## 2. Existing Resource Reuse

### 2.1 Topology Template

Source: `data/graph-ultramoe-718b.js`

Reusable node identities:

- inputs: `token_ids`, `position_ids`, `attn_mask`
- stem: `embedding`, `embedding_weight`, `dense_block`
- MoE representative layer: `mla`, `moe_prenorm`, `gate`, `w_gate`, `a2a_dispatch`, `expert_group_00..31`, `shared_expert`, `w_expert`, `a2a_combine`, `moe_residual`
- output: `final_norm`, `lm_head`, `lm_head_weight`, `logits_allgather`, `logits`

Required adapters:

```ts
type PanguPro72BTopologyAdapter = {
  sourceBuilder: 'buildUltraMoE718BGraph';
  targetModel: 'Pangu Pro MoE 72BA16B';
  overrideLabels: true;
  overrideExpertCount: 64;
  overrideMoeLayerCount: 48;
  overrideDenseLayers: [];
  addNoopPipelineSlots: 2;
  ignoreSourceCommentsAbout718B: true;
  ignoreSourceCommentsAbout256Experts: true;
};
```

### 2.2 Schedule

Source: `data/strict-1f1b-trace-sim.json`

Keep these fields:

- `schema`
- `fidelity`
- `defaultTick`
- `config.dp`
- `config.pp`
- `config.tp`
- `config.ep`
- `config.cp`
- `config.vpp`
- `config.microbatches`
- `config.stageRanges`
- `lanes`
- `ticks[].region`
- `ticks[].stages[stage].phase`
- `ticks[].stages[stage].micro`
- `ticks[].stages[stage].layerRange`
- `ticks[].stages[stage].opFocus.layer`
- `ticks[].stages[stage].opFocus.step`
- `ticks[].stages[stage].comm`
- `ticks[].stages[stage].dependsOn`
- `ticks[].stages[stage].produces`
- `ticks[].stages[stage].explain`

### 2.3 Training Metrics

Source: `data/timeseries.js`

Keep all current series:

- `train_loss`
- `val_loss`
- `eval_mmlu`
- `grad_norm`
- `load_balance_loss`

Keep event fields:

- `faultStep`
- `collapseStep`
- `defaultStep`
- `faultEvent.step`
- `faultEvent.what`
- `faultEvent.node`
- `anomalies.val_loss`
- `anomalies.grad_norm`
- `anomalies.load_balance_loss`

### 2.4 Analysis Metrics

Source: `js/analysis-data.js`

Keep all MoE arrays:

- `loadRatio`
- `assignedTokens`
- `acceptedTokens`
- `droppedTokens`
- `reroutedTokens`
- `routerProbMass`
- `allToAllSendTokens`
- `allToAllRecvTokens`

Keep layer summaries:

- `assignedTokens`
- `droppedTokens`
- `overloadedExperts`
- `idleExperts`
- `maxLoadRatio`
- `avgLoadRatio`
- `p95LoadRatio`
- `allToAllSendTokens`
- `allToAllRecvTokens`
- `allToAllSkew`

Keep rank/card metrics:

- rank: `computeUs`, `commUs`, `bubbleUs`, `utilRatio`, `commRatio`, `bubbleRatio`, `totalUs`
- card: `utilRatio`, `commRatio`, `bubbleRatio`, `pressure`, `state`

### 2.5 Online Source Data Snapshot

Fetched on 2026-07-01.

| source | status | use |
|---|---|---|
| `https://arxiv.org/html/2505.21411v2` | accessible | canonical Pangu Pro 72B architecture, MoGE routing, training parallelism, inference metrics |
| `https://gitcode.com/ascend-tribe/pangu-pro-moe` | page accessible; anonymous `git clone` unavailable | public source entry/provenance only; do not claim local source fields from it |
| `https://ai.gitcode.com/ascend-tribe/openpangu-ultra-moe-718b-model` | anonymous clone succeeded from AtomGit mirror | implementation object naming, config schema, vLLM Ascend MoE execution objects |

Clone details for implementation reference:

```ts
type PanguImplementationReferenceSnapshot = {
  repo: 'openpangu-ultra-moe-718b-model';
  cloneUrl: 'https://atomgit.com/ascend-tribe/openpangu-ultra-moe-718b-model.git';
  head: '562e3fb merge dev-update into main';
  files: [
    'config.json',
    'configuration_openpangu_moe.py',
    'modeling_openpangu_moe.py',
    'inference/runner_config/tp32.yaml',
    'inference/generate.sh',
    'inference/vllm_ascend/models/open_pangu.py',
    'inference/vllm_ascend/ops/fused_moe.py',
    'inference/vllm_ascend/utils.py',
    'doc/vllm_ascend_for_openpangu_ultra_moe_718b.md'
  ];
};
```

Important source-boundary rule:

- `Pangu Pro MoE 72B` is the target model for source-grounded metrics.
- `openPangu-Ultra-MoE-718B` is an implementation reference only; its 61 layers, 256 routed experts, 1 shared expert, 718B total params, and 39B active params must not overwrite Pangu Pro 72B target constants.

---

## 3. Target Model Constants

### 3.1 Canonical Online Pangu Pro 72B

Use this as the only target model config.

Sources:

- `Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity`, arXiv HTML v2: `https://arxiv.org/html/2505.21411v2`
- public source entry: `https://gitcode.com/ascend-tribe/pangu-pro-moe`

```ts
type PanguProMoe72BOnlineConfig = {
  id: 'panguProMoe72BA16B';
  name: 'Pangu Pro MoE';
  displayName: 'Pangu Pro MoE 72BA16B';
  totalParamsB: 71.99;
  activatedParamsB: 16.50;
  transformerLayers: 48;
  trainingPipelineSlots: 50; // 48 transformer layers + 2 no-op layers
  appendedNoopLayersForPpBalance: 2;
  routedExperts: 64;
  activatedExpertsPerTok: 8;
  sharedExperts: 4;
  vocabularySize: 153376;
  hiddenSize: 5120;
  intermediateSize: 1344;
  queryHeads: 40;
  keyValueHeads: 8;
  headSize: 128;
  routeArchitecture: 'MoGE group-balanced routing';
  groupRouting: {
    expertGroups: 8;
    expertsPerGroup: 8;
    activatedExpertsPerGroup: 1;
    idealTokenSharePerExpert: 0.125;
    imbalanceScoreTarget: 0;
  };
  numMtpLayers: 'used for speculative/MTP decode acceleration, exact count not disclosed in paper';
  renderMtpHead: true;
  fidelity: 'paper';
};
```

Canonical layer groups:

| group | layers/slots | kind | EP relevance |
|---|---:|---|---|
| transformer | L0-L47 | MoGE/MoE | Router, group-balanced local Top-1 per group, routed experts, shared experts |
| no-op pipeline balance | S48-S49 | no-op training slots | PP/VPP load-balancing placeholders only, no expert cells |
| output/head | final norm + LM head | op/tensor | TP/logits context, no EP dispatch |

---

## 4. Parallel Configs

### 4.1 Legacy Schedule Template Status

```ts
type LegacyScheduleTemplateStatus = {
  file: 'data/strict-1f1b-trace-sim.json';
  currentShape: 'DP2 x PP4 x TP2 x EP2';
  targetShape: 'Pangu Pro MoE training TP8 x EP2 x CP1 x PP5 x VPP5, DP not disclosed';
  useAs: 'interaction/timeline template only';
  requiredBeforeImplementation: [
    'regenerate PP stage ranges for 50 training slots over 5 pipeline stages',
    'label rank/timeline data as scaled-simulated if not rendering full 4K NPU context',
    'do not display DP2/PP4/TP2/EP2 as Pangu Pro paper configuration'
  ];
};
```

Rendering scale policy:

- Source facts show training uses 4K Ascend NPUs, but exact DP is not disclosed.
- The UI may render one representative PP/TP/EP slice or a scaled-down rank grid for readability.
- Any scaled rank grid must say `scaled simulation`, while model constants, EP buckets, and PP/VPP slots remain Pangu Pro v2 values.

### 4.2 Pangu Pro Paper/Training Config Context

Use this as the primary Pangu training context for a source-grounded page:

```ts
type PanguProTrainingParallelContext = {
  tp: 8;
  ep: 2;
  cp: 1;
  pp: 5;
  vpp: 5;
  transformerLayers: 48;
  pipelineSlots: 50;
  noopLayersForPpBalance: 2;
  pipelineStages: 5;
  virtualPipelineStages: 5;
  accelerationStrategies: [
    'Hierarchical EP All-to-All Communication',
    'Adaptive Pipeline Overlap Mechanism',
    'Fused operators'
  ];
  mfuRelativeIncrease: 0.35;
  loadImbalanceReduction: '>50% by max execution-time disparity for permute and gmm_up';
  noLongerRequiredFromPanguUltraForThisScale: [
    'fine-grained recomputation',
    'tensor swapping'
  ];
  dp: 'not explicitly disclosed';
  derivedDpApprox: 'about 50 if 4K Ascend NPUs are treated as about 4000 ranks and TP*EP*PP*CP = 80';
  fidelity: 'paper';
};
```

Rendering rule:

- Pangu Pro paper context is shown as the canonical training badge or inspector row
- `derivedDpApprox` must always be marked derived

### 4.3 Pangu Pro Inference Parallel Context

Use these objects to replace generic tutorial EP/TP examples for inference:

```ts
type PanguProInferenceParallelContext = {
  platform800IA2: {
    quantization: 'W8A8';
    prefill: {
      inputSeqLen: 2048;
      batchSize: 2;
      ttftMs: 424.21;
      inputThroughputTokensPerSecPerCard: 4828;
    };
    decode: [
      { batchSize: 1, tpotMs: 18.44, outputThroughputTokensPerSecPerCard: 14 },
      { batchSize: 456, tpotMs: 99.31, outputThroughputTokensPerSecPerCard: 1148 },
      { batchSize: 584, tpotMs: 95.56, outputThroughputTokensPerSecPerCard: 1528, usesMtpAcceleration: true }
    ];
    latencyBreakdown: {
      weightTransferRatio: 0.29;
      otherLatencySources: ['KV cache', 'computation', 'communication'];
    };
  };
  platform300IDuo: {
    quantization: 'W8A8';
    prefill: { batchSize: 2, inputSeqLen: 2048, latencyMs: 1940.3, throughputTokensPerSecPerCard: 1055 };
    decode: [
      { batchSize: 80, latencyMs: 99.5, throughputTokensPerSecPerCard: 201 },
      { batchSize: 128, latencyMs: 99.7, throughputTokensPerSecPerCard: 321, usesMtpAcceleration: true }
    ];
  };
  h2pStrategy: {
    attention: 'DP2+TP4';
    routedExperts: 'TP2+EP4';
    sharedExperts: 'TP8';
    communicationRewrite: [
      'attention AllReduce -> ReduceScatter + AllGather',
      'move AllGather after RMSNorm',
      'MoE global AllReduce -> global ReduceScatter + local AllGather'
    ];
    overlapOps: [
      'GMMRS = GroupedMatMul + ReduceScatter',
      'AGMM = AllGather + MatMul'
    ];
    expertComponentGlobalAgRsLatencyShare: 0.08;
  };
};
```

For a training-focused first version, display inference values in an optional "source metrics" tab. Do not mix throughput rows into simulated training timelines.

---

## 5. Tutorial Theory Namespace

The linked tutorial contributes concepts, formulas, and framework implementation objects. Store them separately:

```ts
type EpTheorySource = {
  sourceId: 'awesome-ml-sys-readme-4';
  sourceUrl: string;
  concepts: EpConcept[];
  formulas: EpFormula[];
  frameworkCases: FrameworkCase[];
  sourceExamples: SourceExample[];
};
```

### 5.1 Concepts To Preserve

| concept | required data fields |
|---|---|
| Sparse MoE | total experts, Top-k active experts, activated parameter intuition |
| Shared experts | always active, not routed, optional isolation from EP communication |
| Fine-grained experts | many smaller experts; TP can make GEMMs too skinny |
| No-EP baseline | every GPU stores all experts; memory explodes for large MoE |
| EP | shard experts along expert dimension, dispatch tokens physically across ranks |
| Dispatch All-to-All | distributed transpose from sequence/rank layout to expert/rank layout |
| Expert Compute | local FFN/group GEMM; no inter-rank sync during expert math |
| Combine All-to-All | return expert outputs to original token owner rank |
| TP | shard matrices inside an expert/layer; collective after partial results |
| ETP | first EP, then TP each expert; separate from "EP for experts + TP for non-MoE" |
| FSDP + EP | EP is explicit expert dim sharding; FSDP is parameter sharding/gathering |
| DeepEP | fused dispatch/combine, RDMA-oriented backend, handle-based combine |
| EPLB | expert redundancy/load balancing to reduce hotspot experts |
| Fused MoE | grouped/fused expert kernels when one GPU owns multiple experts |
| Prefetch | pre-gather next layer/block/experts to overlap FSDP communication |

### 5.2 Source Examples

The tutorial uses DeepSeek examples that should be shown as theory examples, not target constants:

```ts
type SourceExample = {
  modelFamily: 'DeepSeek MoE / DeepSeek V3 / DeepSeek R1';
  expertsExample?: 64 | 256;
  topKExample?: 8 | 16;
  note: string;
};
```

Use copy such as:

- "source example: DeepSeek-style fine-grained MoE can use many experts, e.g. 256."
- "target instance: Pangu Pro MoE 72BA16B uses 64 routed experts and Top-8."
- "source discusses both Top-8 examples and large-k/DeepEP discussion; do not infer Pangu k=16."

---

## 6. 2D Side View Coordinate Contract

### 6.1 Axes

```ts
type SideViewAxes = {
  x: 'model depth, left to right: input -> L0..L47 -> S48..S49 no-op training slots -> output';
  y: 'parallel/runtime lanes: hidden rail, Router/EP flow, rank groups, timeline rows';
  color: 'semantic object or metric heatmap';
};
```

Recommended bands from top to bottom:

1. `modelRail`: input, embedding, dense layers, MoE layers, final/head.
2. `moeLayerDetail`: only selected or hovered MoE layers expand Router, Dispatch, Experts, Combine.
3. `epShardMatrix`: EP0/EP1 buckets for selected layer or aggregated heatmap for all MoE layers.
4. `rankProjection`: DP/PP/TP/EP rank groups aligned under PP stage ranges.
5. `timeline`: 1F1B tick or runtime swimlane, optionally collapsed into communication bars.

### 6.2 Layer Rendering

Default render:

- L0-L47 as MoGE/MoE layers.
- S48-S49 as no-op training slots for PP/VPP load balancing.
- Every 5 or 10 layers/slots: label.
- PP boundaries: vertical dividers at stage ranges.
- Selected MoE layer: expand into the five EP phases.

Selected MoE layer side detail:

```text
hidden -> MLA -> PreNorm -> Router Top-8
                         -> Dispatch A2A -> EP0/EP1 expert shards -> Combine A2A -> PostNorm
                         -> Shared Expert -------------------------^
```

### 6.3 Large Count Simplification

Detailed data:

- store all expert cells: `48 x 64 = 3072`
- store per-layer summaries
- store per-EP-bucket summaries

Default view:

- show `model.moeLayerCount x model.routedExperts` heatmap if space allows
- otherwise show `model.moeLayerCount x ep` EP bucket heatmap, with drill-down to `model.routedExperts` experts for one layer

Aggregation:

```ts
expertBucket = Math.floor(expertId / (routedExperts / ep));
expertsPerEp = routedExperts / ep; // training EP2: 64 / 2 = 32
```

---

## 7. Object Catalog

### 7.1 Architecture Objects

```ts
type ArchitectureObject =
  | InputTensor
  | ParameterTensor
  | DenseLayer
  | MoeLayer
  | Router
  | DispatchComm
  | ExpertPool
  | RoutedExpert
  | ExpertShard
  | SharedExpert
  | CombineComm
  | CommunicationEdge
  | OutputOp;
```

Required fields:

```ts
type BaseObject = {
  id: string;
  label: string;
  kind: 'tensor' | 'op' | 'module' | 'comm' | 'rank' | 'metric';
  layer?: number;
  stage?: number;
  source: 'arxiv-v2' | 'pangu-implementation-reference' | 'legacy-local-page' | 'derived' | 'tutorial-theory' | 'simulated';
  fidelity: 'paper' | 'repo-source' | 'config' | 'topology-template' | 'derived-formula' | 'schedule-simulated' | 'metric-simulated' | 'theory';
  description: string;
};
```

### 7.2 EP Runtime Objects

```ts
type Router = BaseObject & {
  kind: 'op';
  layer: number;
  routedExperts: number; // Pangu Pro: 64
  topK: number; // Pangu Pro: 8
  routingMode: 'MoGE group-balanced' | 'standard TopK';
  outputFields: ['expertIds', 'expertScores', 'expertProbs', 'tokenToExpertMap'];
};

type DispatchComm = BaseObject & {
  kind: 'comm';
  commType: 'all-to-all-dispatch';
  group: EpGroupId;
  inputLayout: 'tokens grouped by source rank / sequence position';
  outputLayout: 'tokens grouped by destination expert shard';
};

type ExpertShard = BaseObject & {
  kind: 'module';
  layer: number;
  ep: number;
  experts: number[];
  expertCount: number; // training EP2: 32; inference routed expert TP2+EP4 uses a different partition
};

type CombineComm = BaseObject & {
  kind: 'comm';
  commType: 'all-to-all-combine';
  group: EpGroupId;
  inputLayout: 'expert outputs grouped by expert shard';
  outputLayout: 'expert outputs restored to token owner rank';
};
```

### 7.3 Hardware Objects

```ts
type Rank = {
  rank: number;
  dp: number;
  pp: number;
  tp: number;
  ep: number;
  stageRange: [number, number];
  group: string; // D0.PP1.TP0.EP1
};

type EpGroup = {
  id: string;
  dp: number;
  pp: number;
  tp: number;
  ranks: number[]; // EP0..EPn under fixed DP/PP/TP
  layers: [number, number];
};

type TpGroup = {
  id: string;
  dp: number;
  pp: number;
  ep: number;
  ranks: number[];
};

type DpGroup = {
  id: string;
  pp: number;
  tp: number;
  ep: number;
  ranks: number[];
};
```

### 7.4 Communication Objects

Every visible communication arrow, band, timeline event, and matrix cell must be generated from `CommunicationEdge`. Do not create communication-only SVG paths without a backing object.

```ts
type CommunicationKind =
  | 'ep-dispatch-a2a'
  | 'ep-combine-a2a'
  | 'ep-backward-a2a'
  | 'tp-allreduce'
  | 'tp-reducescatter'
  | 'tp-allgather'
  | 'pp-sendrecv'
  | 'dp-grad-sync'
  | 'fsdp-allgather'
  | 'fsdp-reducescatter'
  | 'h2p-attention-rs-ag'
  | 'h2p-moe-rs-ag'
  | 'gmmrs-overlap'
  | 'agmm-overlap';

type CommunicationPhase =
  | 'training-forward'
  | 'training-backward'
  | 'optimizer'
  | 'inference-prefill'
  | 'inference-decode'
  | 'theory-comparison';

type CommunicationEdge = BaseObject & {
  kind: 'comm';
  commKind: CommunicationKind;
  phase: CommunicationPhase;
  layer?: number; // 0..47 for real MoGE layers; no-op S48/S49 cannot own expert communication
  pipelineSlot?: number; // 0..49, for PP/VPP timeline and no-op slots
  sourceGroup: string;
  targetGroup: string;
  participants: string[];
  tensor: string;
  shapeFormula?: string;
  bytesFormula?: string;
  metricRefs: string[];
  renderViews: ('side-overview' | 'ep-flow-detail' | 'communication-matrix' | 'rank-timeline' | 'inspector')[];
  requiredForCoverage: boolean;
};
```

Required communication catalog:

| id | source/fidelity | expected coverage | render role |
|---|---|---|---|
| `ep_dispatch_forward` | derived from arXiv v2 MoGE + EP2 | one logical dispatch per real MoGE layer: `48` | Router to EP0/EP1 shards |
| `ep_combine_forward` | derived from arXiv v2 MoGE + EP2 | one logical combine per real MoGE layer: `48` | EP0/EP1 outputs back to token owner |
| `ep_backward_a2a` | derived/theory unless measured trace exists | backward dispatch/combine gradient path per real MoGE layer | timeline/inspector only by default |
| `training_hierarchical_ep_a2a` | arXiv v2 paper | present as training strategy object | summary badge and communication matrix group |
| `training_tp8_collectives` | derived from TP8 | TP communication rows, not per-op measured | formula/table and rank timeline |
| `training_pp5_vpp5_sendrecv` | derived from PP5/VPP5 | stage boundary send/recv across 50 slots | PP/VPP timeline lanes |
| `inference_attention_dp2_tp4` | arXiv v2 paper | attention `AllReduce -> ReduceScatter + AllGather` | source metrics tab |
| `inference_routed_expert_tp2_ep4` | arXiv v2 paper | routed expert TP2+EP4 communication | source metrics tab |
| `inference_shared_expert_tp8` | arXiv v2 paper | shared expert TP8 communication | source metrics tab |
| `gmmrs_agmm_overlap` | arXiv v2 paper | GMMRS and AGMM overlap objects | overlap annotation |
| `fsdp_ep_comparison` | tutorial-theory | FSDP all-gather/reduce-scatter objects | comparison tab only |
| `deepep_comparison` | tutorial-theory | fused dispatch/combine objects | comparison tab only |

Pangu-specific rendering rule:

- EP `Dispatch` and `Combine` must be shown for all 48 real MoGE layers, either as individual layer edges, aggregated layer bars, or heatmap cells.
- S48/S49 no-op slots can show PP/VPP scheduling but must not show Router, routed experts, shared experts, or EP token dispatch.
- Inference H2P communication is a source-metrics/reference view. It must not be mixed into the training timeline unless explicitly labeled `inference`.
- FSDP/DeepEP/VeOmni/Automodel/TorchTitan objects remain `tutorial-theory` or `external-comparison-only`, not Pangu source truth.

### 7.5 Pangu Implementation Source Objects

Replace the tutorial framework cases with Pangu objects whenever a Pangu source exposes equivalent objects.

Public Pangu Pro 72B source entry exists at `https://gitcode.com/ascend-tribe/pangu-pro-moe`, but anonymous `git clone` was not available during this source pass (`could not read Username`). Therefore:

- Pangu Pro 72B architecture/training/inference constants come from the paper.
- Pangu code object names and implementation schemas come from the public, anonymously cloneable `openpangu-ultra-moe-718b-model` repository.
- Pangu Ultra 718B values must not overwrite Pangu Pro 72B target constants.

```ts
type PanguSourceObject = {
  id: string;
  sourceModel: 'Pangu Pro MoE 72B' | 'openPangu-Ultra-MoE-718B';
  sourceKind: 'paper' | 'repo-file' | 'repo-doc' | 'legacy-local-page';
  sourceUrl?: string;
  sourcePath?: string;
  replacesTutorialObject?: string[];
  codeObjects: string[];
  metricsOrFields: string[];
  targetUse: 'canonical-data' | 'implementation-reference' | 'comparison-only';
};
```

Required Pangu rows:

| id | target use | objects/fields to include |
|---|---|---|
| `pangu_pro_moge_architecture` | canonical-data | `MoGE`, `Expert Partitioning`, `Group-Balanced Routing`, `Imbalance Score`, `N=64`, `K=8`, `M=8`, `K'=1`, `sharedExperts=4`, `idealTokenSharePerExpert=12.5%` |
| `pangu_pro_training_parallel` | canonical-data | `TP8`, `EP2`, `CP1`, `PP5`, `VPP5`, `48 transformer layers`, `2 no-op layers`, `50 pipeline slots`, `Hierarchical EP All-to-All`, `Adaptive Pipeline Overlap`, `Fused operators`, `MFU +35% relative` |
| `pangu_pro_inference_h2p` | canonical-data | `attention DP2+TP4`, `routed experts TP2+EP4`, `shared experts TP8`, `AllReduce -> ReduceScatter + AllGather`, `GMMRS`, `AGMM`, `W8A8`, `4828 input tokens/s/card`, `1148/1528 output tokens/s/card` |
| `pangu_ultra_config_schema` | implementation-reference | `config.json`, `PanguUltraMoEConfig`, `auto_map`, `num_dense_layers`, `hidden_size`, `num_routed_experts`, `num_experts_per_tok`, `num_shared_experts`, `moe_intermediate_size`, `routed_scaling_factor`, `norm_topk_prob` |
| `pangu_ultra_hf_modeling` | implementation-reference | `MoEGate`, `PanguUltraMoE`, `PanguUltraMoEDecoderLayer`, `topk_idx`, `topk_weight`, `tokens_per_expert`, `shared_experts`, `layer_idx >= num_dense_layers` |
| `pangu_ultra_vllm_moe` | implementation-reference | `OpenPanguMoE`, `AscendFusedMoE`, `FusedMoEState`, `AllGather`, `All2All`, `MC2`, `AllGatherEP`, `NaiveMulticast`, `expert_map`, `ExpertLoadBalancer`, `determine_expert_map` |
| `pangu_ultra_inference_runner` | implementation-reference | `tp32.yaml`, `attn_tp_size=32`, `moe_tp_size=32`, `embed_tp_size=32`, `generate.sh`, `WORLD_SIZE=32`, `4 nodes x 8 NPUs`, `HCCL_*`, `TASK_QUEUE_ENABLE=2` |

### 7.6 External Framework Comparison Objects

From the tutorial, preserve VeOmni/Automodel/TorchTitan only as comparison rows. These rows should never drive target Pangu counts, rank formulas, or metric values:

```ts
type ExternalFrameworkCase = {
  id: 'veomni' | 'automodel' | 'torchtitan';
  repo: string;
  epStrategy: string;
  fsdpStrategy: string;
  dispatcher?: string;
  prefetch?: string[];
  backend?: 'torch.distributed.all_to_all' | 'DeepEP fused dispatch/combine' | 'optional DeepEP';
  codeObjects: string[];
  importantFields: string[];
  targetUse: 'external-comparison-only';
};
```

Comparison rows:

| case | objects to include |
|---|---|
| VeOmni | `parallel_state.py`, `parallel_plan.py`, `torch_parallelize.py`, `parallelize_model_fsdp2`, `qwen3_moe/parallel_plan.py`, `preprocess`, `token_pre_all2all`, `tokens_post_all2all`, `input_splits`, `output_splits`, `tokens_sent_to_me` |
| Automodel | `FSDP2Manager`, `ExpertParallel`, `apply_ep`, `apply_fsdp`, `parallelize_model`, `MoEFlexTokenDispatcher`, `_DeepepManager`, `FusedDispatch`, `FusedCombine`, `handle`, `get_dispatch_layout`, `num_tokens_per_rank`, `num_tokens_per_rdma_rank`, `num_tokens_per_expert`, `is_token_in_rank` |
| TorchTitan | `expert_parallel.py`, `parallel_dims.py`, `deepep.py`, `llama4/infra/parallelize.py`, `fully_shard(transformer_block.moe.experts)`, `edp_mesh`, `efsdp`, `set_gradient_divide_factor`, forward prefetch next block + experts, backward prefetch previous block + experts |

---

## 8. Metric Catalog

### 8.1 Core Model/EP Metrics

| metric | type | target source | meaning |
|---|---|---|---|
| `transformerLayers` | int | arXiv v2 | real transformer/MoGE layers, `48` |
| `trainingPipelineSlots` | int | arXiv v2 | PP/VPP scheduling slots, `50` |
| `noopPipelineSlots` | int | arXiv v2 | no-op slots appended for PP balance, `2` |
| `routedExperts` | int | arXiv v2 | experts routed by Gate, `64` |
| `activatedExpertsPerTok` | int | arXiv v2 | Top-k active routed experts, `8` |
| `sharedExperts` | int | arXiv v2 | always-active shared expert count, `4` |
| `expertGroups` | int | derived from MoGE | `64 / 8 = 8` groups |
| `expertsPerGroup` | int | derived from MoGE | `8` experts per group |
| `activatedExpertsPerGroup` | int | arXiv v2 | `1`, local Top-1 per group |
| `expertsPerTrainingEp` | number | derived | `routedExperts / training.ep = 32` |
| `hiddenSize` | int | arXiv v2 | token hidden dimension, `5120` |
| `intermediateSize` | int | arXiv v2 | FFN/intermediate size, `1344` |
| `queryHeads` | int | arXiv v2 | `40` |
| `keyValueHeads` | int | arXiv v2 | `8` |

### 8.2 Routing Metrics

| metric | dimension | meaning |
|---|---|---|
| `routerProbMass[layer, expert]` | layer x expert | probability mass assigned by router |
| `assignedTokens[layer, expert]` | layer x expert | routed tokens before capacity/drop |
| `acceptedTokens[layer, expert]` | layer x expert | tokens accepted under capacity |
| `droppedTokens[layer, expert]` | layer x expert | capacity overflow/drop |
| `reroutedTokens[layer, expert]` | layer x expert | overflow tokens rerouted |
| `loadRatio[layer, expert]` | layer x expert | assigned/capacity style pressure |
| `overloadedExperts[layer]` | layer | count of experts over capacity |
| `idleExperts[layer]` | layer | experts receiving near-zero tokens |
| `maxLoadRatio[layer]` | layer | worst expert pressure |
| `p95LoadRatio[layer]` | layer | tail pressure |
| `avgLoadRatio[layer]` | layer | mean pressure |

### 8.3 All-to-All Metrics

| metric | dimension | meaning |
|---|---|---|
| `allToAllSendTokens[layer, expert]` | layer x expert | token volume sent out for dispatch/combine model |
| `allToAllRecvTokens[layer, expert]` | layer x expert | token volume received |
| `allToAllSendTokens[layer]` | layer | layer aggregate send |
| `allToAllRecvTokens[layer]` | layer | layer aggregate recv |
| `allToAllSkew[layer]` | layer | send/recv imbalance ratio |
| `inputSplits[rank]` | rank | tokens this rank sends to each EP rank |
| `outputSplits[rank]` | rank | tokens this rank receives from each EP rank |
| `tokensPerExpert[expert]` | expert | dispatch layout count |
| `tokensPerRank[rank]` | rank | DeepEP/dispatcher layout count |
| `tokensPerRdmaRank[rank]` | rank | RDMA-level layout count when available |
| `isTokenInRank[token, rank]` | sparse bool | token destination membership |

### 8.4 Rank/Timeline Metrics

| metric | dimension | source | meaning |
|---|---|---|---|
| `computeUs` | rank | `buildRankLoadViewModel` | local F/B compute time |
| `commUs` | rank | same | TP/EP/PP/DP communication time |
| `bubbleUs` | rank | same | pipeline waiting time |
| `utilRatio` | rank | same | compute fraction |
| `commRatio` | rank | same | communication fraction |
| `bubbleRatio` | rank | same | idle/wait fraction |
| `task.kind` | task | runtime | `F`, `B`, `tp`, `ep`, `pp`, `dp`, `bubble` |
| `task.microbatch` | task | runtime | micro-batch id |
| `task.startUs` | task | runtime | task start |
| `task.durUs` | task | runtime | task duration |

### 8.5 Training Outcome Metrics

| metric | dimension | source |
|---|---|---|
| `train_loss` | step | `TS_DATA` |
| `val_loss` | step | `TS_DATA` |
| `eval_mmlu` | step | `TS_DATA` |
| `grad_norm` | step | `TS_DATA` |
| `load_balance_loss` | step | `TS_DATA` |
| `collapseIntensity` | step | derived from `load_balance_loss` |

### 8.6 TP vs EP Formula Metrics

Theory variables from the tutorial:

```ts
type ParallelFormulaInputs = {
  N: number; // TP or EP group size
  B: number; // batch size
  L: number; // sequence length
  H: number; // hidden size
  k: number; // Top-k active experts
  S: number; // B * L * H
};
```

Formulas:

```ts
S = B * L * H
commTPBytesPerGpu = 2 * ((N - 1) / N) * S
commTPApprox = 2 * S
commEPBytesPerGpu = 2 * ((N - 1) / N) * (k * S / N)
commEPApprox = (2 * k / N) * S
epToTpCommRatioApprox = k / N
```

Additional theory metrics:

| metric | default source example | note |
|---|---:|---|
| `tpIntraNodeBandwidthGBps` | 900 | tutorial example for NVLink-class domain |
| `epInterNodeBandwidthGBpsMin` | 50 | tutorial example for RDMA domain |
| `epInterNodeBandwidthGBpsMax` | 100 | tutorial example for RDMA domain |
| `a2aConnectionScale` | `N^2` | small-connection/tail-latency risk |
| `gemmShapeRisk` | enum | TP can make expert GEMM skinny |
| `mfuImpact` | enum | TP lower, EP higher for fine-grained experts |

### 8.7 FSDP/EP Metrics

| metric/object | meaning |
|---|---|
| `epShardDim = 0` | experts are explicitly sharded along expert dimension |
| `fsdpExpertShardDim = 1` | hidden dimension sharding for expert params in VeOmni-style description |
| `fsdpRegularShardDim = 0` | normal FSDP parameter sharding |
| `allGatherCount` | parameter gathers before local expert FFN compute |
| `reduceScatterCount` | backward FSDP gradient reduce-scatter |
| `prefetchForwardTargets` | next block and/or next experts |
| `prefetchBackwardTargets` | previous block and/or previous experts |
| `reshardAfterForward` | whether FSDP releases full params after forward |
| `gradientDivideFactor` | TorchTitan-style expert gradient scaling field |

---

## 9. Main Data Schema

```ts
type EpExpertParallel2DSpec = {
  schema: 'pangu.ep-expert-parallel-2d.v1';
  model: PanguProMoe72BOnlineConfig;
  trainingParallelContext: PanguProTrainingParallelContext;
  inferenceParallelContext: PanguProInferenceParallelContext;
  legacyScheduleTemplate?: LegacyScheduleTemplateStatus;
  theory: EpTheorySource;
  axes: SideViewAxes;
  architecture: {
    objects: ArchitectureObject[];
    layers: LayerObject[];
    edges: ArchitectureEdge[];
  };
  communications: {
    edges: CommunicationEdge[];
    matrix: CommunicationMatrixCell[];
    catalog: CommunicationCatalog;
  };
  runtime: {
    ranks: Rank[];
    epGroups: EpGroup[];
    tpGroups: TpGroup[];
    dpGroups: DpGroup[];
    schedule: Strict1F1BSchedule;
  };
  metrics: {
    training: TrainingSeries;
    moe: MoeRuntimeMetrics;
    rankLoad: RankLoadViewModel;
    cardLoad: CardLoadViewModel;
    formulas: ParallelFormulaResult[];
  };
  views: Ep2DViewSpec[];
  coverage: CoverageReport;
  validation: ValidationRule[];
};
```

### 9.0 Completeness Contract

The data file must contain the full universe first, then views can aggregate it.

```ts
type PanguProUniverse = {
  realLayers: 48;
  pipelineSlots: 50;
  noopSlots: [48, 49];
  routedExpertsPerLayer: 64;
  sharedExpertsPerLayer: 4;
  expertGroups: 8;
  expertsPerGroup: 8;
  activatedExpertsPerGroup: 1;
  trainingEp: 2;
  expertsPerTrainingEp: 32;
  expertCellsExpected: 48 * 64;
  sharedExpertCellsExpected: 48 * 4;
};
```

Mandatory generated tables:

| table | required rows |
|---|---:|
| `layers` | `50` rows: 48 real MoGE layers + 2 no-op PP/VPP slots |
| `expertCells` | `3072` rows: `48 x 64` |
| `sharedExpertCells` | `192` rows: `48 x 4` |
| `epBucketSummaries` | `96` rows: `48 x EP2` |
| `routerObjects` | `48` rows |
| `dispatchCommEdges` | at least `48` logical forward dispatch rows |
| `combineCommEdges` | at least `48` logical forward combine rows |
| `ppSlotObjects` | `50` rows |
| `communicationEdges` | all communication rows listed in the communication catalog |

If the implementation renders a scaled rank grid, it may reduce visible rank rows, but it must not reduce model-layer/expert/communication coverage.

### 9.1 Layer Object

```ts
type LayerObject = {
  layer: number;
  kind: 'dense' | 'moe' | 'moge' | 'noop';
  ppStage: number;
  stageRange: [number, number];
  hasRouter: boolean;
  hasDispatchA2A: boolean;
  hasExpertPool: boolean;
  hasSharedExpert: boolean;
  hasCombineA2A: boolean;
  routedExperts: number;
  expertsPerTok: 0 | 8;
};
```

Rules:

- Dense layers have all EP fields false/zero.
- Canonical Pangu Pro MoGE/MoE layers have Router, Dispatch, Expert Pool, Shared Expert, Combine.
- Pangu Pro no-op training slots have all EP fields false/zero and `kind='noop'`.
- Every MoE layer maps to exactly one PP stage by `stageRanges`.

### 9.2 Expert Cell

```ts
type ExpertCell = {
  layer: number;
  expertId: number; // 0..63
  ep: number;
  localExpertId: number;
  loadRatio: number;
  assignedTokens: number;
  acceptedTokens: number;
  droppedTokens: number;
  reroutedTokens: number;
  routerProbMass: number;
  allToAllSendTokens: number;
  allToAllRecvTokens: number;
  state: 'idle' | 'normal' | 'overloaded' | 'dropped';
};
```

### 9.3 EP Bucket Summary

```ts
type EpBucketSummary = {
  layer: number;
  ep: number;
  expertIds: number[];
  expertCount: number;
  assignedTokens: number;
  acceptedTokens: number;
  droppedTokens: number;
  reroutedTokens: number;
  maxLoadRatio: number;
  avgLoadRatio: number;
  p95LoadRatio: number;
  allToAllSendTokens: number;
  allToAllRecvTokens: number;
  allToAllSkew: number;
};
```

### 9.4 Communication Matrix Cell

```ts
type CommunicationMatrixCell = {
  id: string;
  commKind: CommunicationKind;
  phase: CommunicationPhase;
  sourceGroup: string;
  targetGroup: string;
  edgeIds: string[];
  logicalEdgeCount: number;
  tokenCount?: number;
  bytes?: number;
  skew?: number;
  source: 'arxiv-v2' | 'derived' | 'simulated' | 'tutorial-theory';
  fidelity: 'paper' | 'derived-formula' | 'schedule-simulated' | 'theory';
};
```

Matrix requirements:

- `ep-dispatch-a2a` and `ep-combine-a2a` must be groupable by `layer`, `PP stage`, `EP source`, and `EP target`.
- `tp-*` rows must be groupable by `TP group` and phase.
- `pp-sendrecv` rows must be groupable by `PP stage boundary` and `VPP slot`.
- theory rows must be filterable off by default.

### 9.5 Communication Catalog

```ts
type CommunicationCatalog = {
  requiredKinds: CommunicationKind[];
  panguTraining: {
    epDispatchForward: { expectedLogicalEdges: 48; required: true };
    epCombineForward: { expectedLogicalEdges: 48; required: true };
    hierarchicalEpAllToAll: { expectedObjectsMin: 1; required: true };
    tp8Collectives: { expectedObjectsMin: 1; required: true; fidelity: 'derived-formula' };
    pp5Vpp5SendRecv: { expectedObjectsMin: 5 * 5; required: true; fidelity: 'derived-formula' };
  };
  panguInference: {
    attentionDp2Tp4: { expectedObjectsMin: 1; required: true };
    routedExpertTp2Ep4: { expectedObjectsMin: 1; required: true };
    sharedExpertTp8: { expectedObjectsMin: 1; required: true };
    gmmrs: { expectedObjectsMin: 1; required: true };
    agmm: { expectedObjectsMin: 1; required: true };
  };
  comparison: {
    fsdpAllGather: { expectedObjectsMin: 1; required: false };
    fsdpReduceScatter: { expectedObjectsMin: 1; required: false };
    deepEpFusedDispatchCombine: { expectedObjectsMin: 1; required: false };
  };
};
```

### 9.6 Coverage Report

```ts
type CoverageStatus = 'pass' | 'warn' | 'fail';

type CoverageItem = {
  id: string;
  expected: number | string;
  actual: number | string;
  status: CoverageStatus;
  source: 'arxiv-v2' | 'derived' | 'simulated' | 'tutorial-theory';
  message: string;
};

type CoverageReport = {
  overall: CoverageStatus;
  items: CoverageItem[];
  missingRequiredIds: string[];
  generatedAt?: string;
};
```

Required coverage items:

| id | expected |
|---|---:|
| `coverage.layers.real` | `48` |
| `coverage.pipeline.slots` | `50` |
| `coverage.pipeline.noop_slots` | `2` |
| `coverage.expert_cells` | `3072` |
| `coverage.shared_expert_cells` | `192` |
| `coverage.ep_buckets` | `96` |
| `coverage.router_objects` | `48` |
| `coverage.dispatch_edges.forward` | `48` |
| `coverage.combine_edges.forward` | `48` |
| `coverage.noop_has_no_ep_comm` | `0 violations` |
| `coverage.comm.required_catalog` | `all required Pangu communication ids present` |
| `coverage.theory_filterable` | `all tutorial/FSDP/DeepEP comparison objects filterable` |

---

## 10. View Specs

### 10.1 `side-overview`

Purpose: show the whole Pangu 72B model in 2D side projection.

Required data:

- all 48 transformer/MoGE layers
- 2 no-op training slots for 50-slot PP/VPP balance
- PP/VPP stage ranges
- output head
- rank projection bands

Default indicators:

- selected step
- selected tick
- current PP stage
- current op focus
- MoE layers with overload badges

### 10.2 `ep-flow-detail`

Purpose: show one selected MoE layer's EP flow.

Required objects:

- `Router Top-8`
- `A2A Dispatch`
- `EP0 shard`
- `EP1 shard`
- `Shared Expert`
- `A2A Combine`
- `Post-MLP RMSNorm`

Required metrics:

- Top-k = 8
- routed expert count = `model.routedExperts`
- experts per EP = `model.routedExperts / training.ep` when training EP is used
- assigned/accepted/dropped/rerouted tokens
- A2A send/recv/skew
- load ratio summary

### 10.3 `expert-heatmap`

Purpose: detailed Pangu expert utilization.

Modes:

- `layer-by-expert`: `model.moeLayerCount x model.routedExperts` cells
- `layer-by-ep`: `model.moeLayerCount x ep` buckets
- `selected-layer-experts`: `1 x model.routedExperts` detailed cells

Metric selector:

- `loadRatio`
- `assignedTokens`
- `droppedTokens`
- `reroutedTokens`
- `routerProbMass`
- `allToAllSendTokens`
- `allToAllRecvTokens`

### 10.4 `communication-matrix`

Purpose: prove every communication family has a data object and a visual representation.

Default filters:

- show Pangu training communication
- hide tutorial comparison communication
- hide inference communication unless `Source Metrics` tab is active

Rows/columns:

- EP source/target groups for `ep-dispatch-a2a` and `ep-combine-a2a`
- PP stage boundaries for `pp-sendrecv`
- TP groups for `tp-*`
- source metrics groups for inference H2P

Required interactions:

- click matrix cell -> inspector lists `CommunicationEdge[]`
- hover layer in side view -> highlight matching communication matrix row/cell
- click EP bucket -> filter matrix to dispatch/combine touching that EP bucket
- missing required communication -> show red `missing` cell, not an empty space

### 10.5 `coverage-panel`

Purpose: make completeness visible and auditable.

Required rows:

- `48 / 48 real layers`
- `50 / 50 pipeline slots`
- `3072 / 3072 routed expert cells`
- `192 / 192 shared expert cells`
- `96 / 96 EP buckets`
- `48 / 48 Router objects`
- `48 / 48 EP Dispatch edges`
- `48 / 48 EP Combine edges`
- `0 no-op EP communication violations`
- `required Pangu communication catalog present`

Status colors:

- pass: all expected counts match
- warn: scaled/simulated rank or timeline data
- fail: missing layer, expert, EP bucket, or required communication edge

### 10.6 `tp-vs-ep-formula`

Purpose: explain why EP is different from TP using tutorial formulas.

Inputs:

- `N`
- `B`
- `L`
- `H`
- `k`

Outputs:

- `S`
- `commTPBytesPerGpu`
- `commEPBytesPerGpu`
- `epToTpCommRatioApprox`
- bandwidth domain note
- GEMM shape note

Preset rows:

1. Canonical Pangu Pro training: `TP=8`, `EP=2`, `k=8`, `H=5120`
2. Canonical Pangu Pro inference: attention `DP2+TP4`, routed experts `TP2+EP4`, shared experts `TP8`
3. Tutorial large EP example: `N=64/256`, `k=8 or 16`, marked theory only

### 10.7 `fsdp-ep-implementation`

Purpose: explain EP + FSDP implementation details from the tutorial.

Rows:

- Pangu forward sequence
- Pangu backward sequence
- Pangu implementation references
- external comparison: VeOmni
- external comparison: Automodel
- external comparison: TorchTitan

Required forward sequence:

```text
gate/router -> A2A Dispatch -> routed expert compute -> shared expert compute -> A2A Return/Combine -> merge -> output
```

Generic FSDP comparison sequence:

```text
expert param/FSDP context -> all-gather -> Expert FFN forward/backward -> reduce-scatter -> release
```

Clarification:

- FSDP backward reduce-scatter is a FSDP/DP action, not EP group gradient aggregation.
- EP experts optimize their own routed expert gradients; no EP-wide expert gradient averaging is implied.

### 10.8 `rank-timeline`

Purpose: reuse the existing schedule/runtime resources in 2D.

Rows:

- paper config badge: `TP8 x EP2 x CP1 x PP5 x VPP5`, DP not disclosed
- scaled rank lanes if needed for readability, explicitly marked `scaled simulation`
- aggregate by PP stage, TP group, EP group, or communication kind

Task kinds:

- `F`
- `B`
- `tp`
- `ep`
- `pp`
- `dp`
- `bubble`

EP-specific overlay:

- highlight `task.kind === 'ep'`
- show Dispatch/Combine relation when tick op focus is `a2a_dispatch` or `a2a_combine`

### 10.9 `inspector`

Purpose: one selected object, with all relevant metrics.

Inspector sections:

1. identity: object id, layer, stage, DP/PP/TP/EP
2. Pangu constants: `48 transformer layers + 2 no-op training slots`, `64 routed experts`, `Top-8`, `4 shared experts`
3. selected metrics: load, tokens, A2A, rank ratios
4. theory note: EP/TP/FSDP concept from tutorial
5. source/fidelity: config, simulated, theory
6. actions: focus heatmap, focus timeline, compare TP/EP, show FSDP path

---

## 11. Derivation Rules

### 11.1 Expert To EP Mapping

```ts
expertsPerEp = routedExperts / ep; // training: 64 / 2 = 32
ep = Math.floor(expertId / expertsPerEp);
localExpertId = expertId % expertsPerEp;
```

If `routedExperts % ep !== 0`, use ragged buckets and store explicit expert id arrays.

### 11.2 Layer To PP Mapping

```ts
ppStageForLayer(layer, stageRanges) =
  stage where stageRanges[stage][0] <= layer <= stageRanges[stage][1]
```

### 11.3 Rank Mapping

```ts
rank = (((dp * ppCount + pp) * tpCount + tp) * epCount + ep)
```

### 11.4 EP Bucket Aggregation

```ts
bucket.assignedTokens = sum(expert.assignedTokens)
bucket.acceptedTokens = sum(expert.acceptedTokens)
bucket.droppedTokens = sum(expert.droppedTokens)
bucket.reroutedTokens = sum(expert.reroutedTokens)
bucket.maxLoadRatio = max(expert.loadRatio)
bucket.avgLoadRatio = avg(expert.loadRatio)
bucket.p95LoadRatio = p95(expert.loadRatio)
bucket.allToAllSendTokens = sum(expert.allToAllSendTokens)
bucket.allToAllRecvTokens = sum(expert.allToAllRecvTokens)
bucket.allToAllSkew = max(send, recv) / max(1, min(send, recv))
```

### 11.5 State Classification

```ts
expert.state =
  droppedTokens > 0 ? 'dropped' :
  loadRatio >= 1.0 ? 'overloaded' :
  assignedTokens < capacityTokens * 0.08 ? 'idle' :
  'normal'
```

### 11.6 Collapse Intensity

Keep existing rule from `op-rank-time.html`:

```ts
collapseIntensity(step) = clamp01((0.42 - load_balance_loss[step]) / 0.38)
```

This is synthetic-demo logic and must be marked as such.

---

## 12. Validation Rules

```ts
type ValidationRule = {
  id: string;
  severity: 'error' | 'warn';
  test: string;
  message: string;
};
```

Required checks:

| id | severity | rule |
|---|---|---|
| `model.layers.total` | error | `transformerLayers === 48` and `trainingPipelineSlots === 50` |
| `model.moe.range` | error | no-op slots are excluded from expert heatmap |
| `model.experts.target` | error | Pangu Pro uses 64 routed experts; do not use tutorial 256, Pangu Ultra 256, or legacy local 80 |
| `model.topk.target` | error | target Pangu instance uses Top-8 |
| `ep.bucket.count` | error | EP bucket summaries cover all `model.routedExperts` experts per selected MoE layer |
| `coverage.layers.real` | error | exactly 48 real MoGE layers |
| `coverage.pipeline.slots` | error | exactly 50 PP/VPP training slots |
| `coverage.expert.cells` | error | exactly 3072 routed expert cells |
| `coverage.shared.cells` | error | exactly 192 shared expert cells |
| `coverage.ep.buckets` | error | exactly 96 EP bucket summaries |
| `coverage.router.objects` | error | exactly 48 Router objects |
| `coverage.comm.dispatch.forward` | error | exactly 48 required forward dispatch communication edges |
| `coverage.comm.combine.forward` | error | exactly 48 required forward combine communication edges |
| `coverage.comm.catalog` | error | all required Pangu communication catalog ids present |
| `coverage.noop.ep` | error | no-op slots S48/S49 contain no Router/Expert/EP token communication |
| `rank.count` | warn | full paper-scale DP is not disclosed; scaled views must be labeled scaled-simulated |
| `rank.formula` | error | rank ids match selected scaled/paper group formula |
| `pp.stage.coverage` | error | stage ranges cover 50 training slots without gaps |
| `fidelity.schedule` | warn | schedule marked simulated |
| `fidelity.metrics` | warn | MoE/rank/card metrics marked synthetic if generated locally |
| `collective.semantic` | warn | EP highlight copy says participant, not faulty rank |
| `source.namespace` | error | DeepSeek examples stay under `theory.sourceExamples`, not target model constants |

---

## 13. Copy/Label Requirements

Use precise labels:

- `EP token dispatch/combine`
- `All-to-All Dispatch`
- `All-to-All Combine`
- `Router Top-8`
- `64 routed experts`
- `32 experts per EP shard` for training EP2
- `Shared Expert`
- `FSDP all-gather`
- `FSDP reduce-scatter`
- `TP All-Reduce`
- `PP send/recv`
- `DP gradient sync`

Avoid these misleading labels:

- `EP layer`
- `EP card`
- `all ranks failed`
- `measured trace` for simulated data
- `Pangu has 256 experts`
- `Pangu k=16`

---

## 14. Minimum Data Artifact

If a separate data file is created later, name it:

```text
data/ep-expert-parallel-2d-spec.js
```

It should export:

```ts
window.PANGU_EP_2D_SPEC = {
  schema: 'pangu.ep-expert-parallel-2d.v1',
  model,
  trainingParallelContext,
  inferenceParallelContext,
  legacyScheduleTemplate,
  theory,
  communications,
  buildExpertCells(metrics),
  buildSharedExpertCells(metrics),
  buildEpBucketSummaries(expertCells),
  buildCommunicationEdges(inputs),
  buildCoverageReport(inputs),
  buildFormulaRows(inputs),
  coverage,
  validationRules,
};
```

Do not duplicate `TS_DATA`, `strict-1f1b-trace-sim.json`, or `analysis-data.js` arrays unless the page needs an offline snapshot.

---

## 15. Suggested First Screen

Data-driven first screen:

1. top bar: `Pangu Pro MoE 72BA16B · EP Expert Parallel · TP8 x EP2 x CP1 x PP5 x VPP5`
2. left tree: `MoE Forward`, `EP Flow`, `Communication Matrix`, `Coverage`, `TP vs EP`, `FSDP + EP`, `Framework Cases`
3. center: 2D side model, L0-L47 plus S48-S49 no-op slots, PP/VPP ranges, selected MoGE layer expanded
4. right inspector: selected object and metrics
5. bottom: EP heatmap, communication matrix, coverage panel, or rank timeline, switchable

Default selected object:

```ts
{
  layer: 0,
  object: 'a2a_dispatch',
  explanation: 'first Pangu Pro MoGE layer dispatch from Router Top-8 into EP0/EP1 expert shards'
}
```

Default selected metric:

```ts
metric = 'allToAllSendTokens'
```

Rationale: this makes the tutorial's central point visible immediately: EP turns logical expert routing into physical All-to-All token movement.
