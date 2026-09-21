# hw-native-sys 层级坐标规范（L7 → L0）——所有视图统一遵循（v2 · 严格版）

> 参照 hw-native-sys（compute-graph-viewer）递归层级。本仓库唯一层级编号来源为
> `src/scene/data.ts` 中的 `HW_LEVELS` / `HW_BY_KEY` / `UB_COORD` / `LAYER_INFO`
> （helper：`levelTag/levelName/levelFull`）。

## 层级表（这就是全部层级，任何视图的层级轴必须与此完全一致、级数一致、顺序一致）

| L | key（代码内部） | 名称 | 英文 | 通往下一级的互联 | sibling 样例 |
|----|----------------|------|------|------------------|--------------|
| L7 | `global`（UB_COORD 里为 `job`） | 全球调度 | Global | **DCN** | Global A / C |
| L6 | `cluster` | 集群 | Cluster | **Scale-Out** | Cluster A / C |
| L5 | `pool` | 服务池 | Service Pool | **Pool 内互联** | Pool 1 / 3（`PODS_PER_POOL = 4`） |
| L4 | `super` | **Pod**（UBL128） | Pod · UBL128 | **Scale-Up** | Pod α / γ |
| L3 | `node` | Host | Host · 1 OS | **PCIe / UB** | Host 1 / 3 |
| L2 | `card` | Chip · NPU | Chip · NPU | **封装互连** | NPU 1 / 3 |
| L1 | `die` | Die（**可选级**） | Die | **NoC** | die 0 |
| L0 | `core` | 核组 Core-Group | Core-Group | 内部：MTE/FixPipe 流水 | AIV · AIC · AICPU |

## 硬性规则（v2 与 v1 的区别：不再有“并入级”行）

1. **“超节点”“机柜”不再是层级**。L4 的名字就是 **Pod**。机柜（`cab`）只是 Pod 内的
   物理分组：可以作为 3D 物理场景（机房/机柜视图）和物理分组框存在，但**绝不出现在**
   层级轴、漏斗、Smartscape 层级、面包屑层级链、层级图例中。文案需要提到时写
   “机柜（物理分组）”。`TOK.supernode`/Atlas 产品名只能出现在产品规格文案中，不得作层级名。
2. **Tile 不是层级**。`tile` 是 L0 Core-Group 的内部粒度（pattern 缓冲/swimlane lane），
   不出现在层级轴/面包屑；L0 的内部组织统一复用 memory-architecture pattern
   （`src/view/CoreGroupPattern.tsx`），标题一律 “L0 内部 / L0 Core-Group 内部”。
3. **所有视图的层级轴完全相同**：运行状态、联动控制台、平面视图（层级图）、3D 漏斗
   全部是同一条链：全球 L7 → 集群 L6 → 服务池 L5 → Pod L4 → Host L3 → Chip·NPU L2 →
   Die L1（可选）→ Core-Group L0，级间互联标注 DCN / Scale-Out / Pool 内互联 / Scale-Up /
   PCIe·UB / 封装互连 / NoC。
4. **L1 Die 标注“可选”**（单 die 芯片可省略）。
5. 旧 UB 互联层级（`UB_LEVELS`）用互联名 id（`NoC·D2D/UB·Host/SU·柜内/SU·Pod/SO`），
   不占用 L 编号；“L1/L2 交换”是交换芯片产品名，保留。
6. **并行维度落位**：TP=L3 Host 内 · EP/PP=L4 Pod 内 · DP=跨 Pod（L5/L6）。
7. `LAYER_INFO`（data.ts）现为 8 条、L7→L0 按位有序，PlaneView LAY.defs 必须与其对齐。
