# NCCL 参考图 · 分析与落点

> 对象：`public/cube-cockpit.html`（座舱：左侧 层级剖面/整网分隔映射/算法展开 + 右侧 物理层级/逻辑魔方/层级堆叠 + 底部 泳道回放）。
> 目的：把三批参考图里的逻辑，按"能落到哪个视图/视角"归类，指导实现。约定：**形状/线型/位置管"是什么"，颜色只管"忙不忙"**（全站红线）。

---

## 0. 一句话结论

三批图正好覆盖通信五层里**站里最薄的两层**：
- **L4 原语 / L3 算法**（图 5–12 的集合原语 before→after 块图）→ 新增 **「原语数据流」面板**（方案 B）。
- **L2 拓扑 / 需求1 路径**（图 1–3、13–23 的物理拓扑 + PXN 换轨）→ **物理楼层图的路径投影 + rail**（方案 option 1）。

其余（NCCL 建图四段、1GPU→多机）作为**来源标注/引导语**，价值低但便宜。

---

## 1. 按组分析（23 张去重成 6 组）

### 组 A — NCCL 建图四段（图1 Topology→Graph search→Graph connect→Kernels；图15 三步"构建PCIe树→加网卡→加NvLink"）
- **提取逻辑**：ring/tree **不是凭空的**，是 NCCL 从物理拓扑（GPU/NIC/PCIe/NVLink）**搜索**出的最优环/树，再"连成 FIFO 管道"跑 reduction。
- **落点**：算法展开面板**顶部一句话来源标注**——"本环/树 = NCCL 按物理拓扑搜出的最优路径"。可选：一个折叠的"建图四段"迷你流程条。
- **优先级**：P3（点睛，非必须）。

### 组 B — 节点内物理拓扑（图2 PCIe树 t-bar；图3/23 8卡环+NIC出口；图13/21/22 NVLink全连+QPI；图14/17 NvSwitch fabric）
- **提取逻辑**：
  1. 节点内 GPU 之间是 **NVLink/NVSwitch any-to-any 网（UB-Mesh），不是星型 hub**（红线④）。
  2. GPU 不直接连网：**GPU → PCIe Switch / NVSwitch → NIC** 才出网。
  3. **rail = NIC 号 = GPU 槽位**（GPU_i 挂 NIC_i）。
  4. CPU/PCIe/QPI 是管理/存储面（对应 VPC 灰）。
- **落点**：
  - 物理层级 **3D Host(L3)/Chip(L2)** 的 UB-Mesh 连线校准（any-to-any，别画成星型）。
  - **option 1 路径投影的"UB 段"（紫）**：域内通信 + 出网前的节点内一跳都走这里。
- **优先级**：P1（喂给 option 1）。

### 组 C — rail 平面 / 超节点（图16 超节点 ACS/AccLink；图18/19/20 DGX spine-leaf S0-3/L0-3 + NIC0-7）
- **提取逻辑**：
  1. **rail-optimized**：GPU_i → NIC_i → Leaf_i；同号 NIC 归一条 rail。**Leaf(L)/Spine(S) = rail 平面的交换层**。
  2. **超节点(ACS/AccLink)** = scale-up 域 = 我们的 Pod/UB 域；域内走 AccLink/UB，跨超节点才上 RDMA rail。
  3. RDMA 平面**只承载跨 Pod/超节点**流量（红线①）。
- **落点**：
  - 三张网 lens 的 **RDMA 平面按 rail 分色**（已在 3D，可在物理楼层图加"网络域"着色）。
  - **option 1 路径投影的"rail 段"（蓝，标 rail N）** + 出口(青)。
- **优先级**：P0/P1（option 1 主体）。

### 组 D — 换轨 / PXN（图18/20 双色路径 DGX-A→DGX-B）
- **提取逻辑**：
  - **不换轨（深紫）**：GPU3 → 自己的 NIC3 → rail3 直出 → 对端 NIC3 → GPU3（源目标同 rail 时最短）。
  - **换轨/PXN（浅紫）**：GPU3 →**节点内经 NVSwitch 搬到与目标 rail 对齐的卡 GPU0** → NIC0 → rail0 出 → 对端 → GPU3。即**先在 UB 域内换轨、再出网**，避免 spine 层跨 rail 多跳。
- **落点**：**option 1 路径的 ◇ 换轨(PXN)点 + "换轨→rail N" 标签**；需求1 验收里"换轨段必须显式、不许画成卡直连远端"。
- **优先级**：P0（需求1 硬要求）。

### 组 E — 集合原语（图5 Broadcast / 6 Scatter / 7 Gather / 8 AllGather / 9 AllToAll / 10 Reduce / 11 ReduceScatter / 12 AllReduce）
- **提取逻辑**：每个原语 = **NPU 列 + 彩色数据块的 before→after**，语义一目了然：
  | 原语 | 数据搬运语义 | 泳道 phase |
  |---|---|---|
  | Broadcast | 1 卡的块 → 全卡各一份（复制） | — |
  | Scatter | 1 卡的 N 块 → 各卡各 1 块（切分下发） | — |
  | Gather | 各卡 1 块 → 1 卡收齐 | — |
  | AllGather | 各卡 1 块 → 全卡都收齐（拼接） | AllReduce 后半 |
  | Reduce | 各卡的块 → 1 卡求和 | — |
  | **ReduceScatter** | 各卡求和后**每卡留 1 段**（对角） | AllReduce 前半 |
  | **AllReduce** | 全卡都得全和 = ReduceScatter + AllGather | **TP·前向 / DP·梯度** |
  | **AllToAll** | N×N **转置**（每卡把第 j 块发给卡 j） | **EP·MoE** |
- **落点**：**方案 B —— 算法展开 tab 新增「原语数据流」子视图**，随选中 phase 显示对应原语的 before→after 块图（纯 2D、无交叉、契合度最高）。这是图片主体（8/23 直接命中），补齐文档标注的 L4"缺 phase 语义可视化"。
- **优先级**：P1（方案 B 主体）。

### 组 F — 扩展概念（图4 1 GPU → multi-GPU/node）
- **提取逻辑**：NCCL 把单卡计算扩到多机多卡。纯引导概念。
- **落点**：算法展开空态的引导文案，价值低。
- **优先级**：P3。

---

## 2. 落点汇总表

| 组 | 图 | 提取逻辑 | 落到哪 | 优先级 |
|---|---|---|---|---|
| A | 1,15 | 环/树从物理拓扑搜出 | 算法展开顶部来源标注 | P3 |
| B | 2,3,13,14,21,22,23 | UB-Mesh any-to-any + GPU→NIC 出口 + rail=NIC号 | 物理 3D 连线校准 + option1 UB段(紫) | P1 |
| C | 16,18,19,20 | rail-optimized + Leaf/Spine + 超节点=UB域 | 三张网 rail 分色 + option1 rail段(蓝) | P0 |
| D | 18,20 | PXN=先UB域内换轨再出网 | option1 ◇换轨点 | P0 |
| E | 5–12 | 8 个集合原语 before→after 语义 | **方案 B 原语数据流面板** | P1 |
| F | 4 | 单卡→多机 | 引导文案 | P3 |

---

## 3. 实施顺序（已与需求方确认：先 option 1 再 B）

1. **option 1 —— 物理楼层图路径投影**（组 B/C/D）：选中卡 → 楼层图上画 **同 Pod=紫(UB域内) · 跨 Pod=蓝(rail N 出网) · ◇=换轨(PXN)点**，机柜级；配"网络域(超节点)"着色 lens。
2. **方案 B —— 原语数据流面板**（组 E）：算法展开加子视图，随 phase 显示 ReduceScatter/AllGather/AllToAll/… 的 NPU 列 before→after 块图。
3. （可选 P3）组 A 来源标注、组 F 引导文案。

---

*参考：NVIDIA NCCL / PXN 博客、DGX NVLink 拓扑、超节点 AccLink 资料；与项目内《NCCL 通信平面图与 3D 视图逻辑整理》《通信概念层级与昇腾超节点映射》一致。*
