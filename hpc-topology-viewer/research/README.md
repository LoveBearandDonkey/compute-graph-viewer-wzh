# AI Infra 统一驾驶舱（原型 v22）

大模型训练/推理运行时监控诊断工具的交互原型：一个 Atlas 950 超节点（8192 NPU）的
物理集群 ⇄ 逻辑魔方 ⇄ 层级堆叠统一 3D 视图 + Smartscape 式下钻 + 整网分隔映射 + 软硬结合连线镜头（互联矩阵联动）+ 模式 B 深潜（泳道/busbw/通信矩阵）。

> 版本分支说明：v22 自 v19 数据版分支迭代（会话交付文件名 `ai_infra_可视化原型_v20.html`），
> 特性为暗色实现；合入 v21 light 主线时需适配主题 token。

## 使用

浏览器直接打开 `index.html`（需联网加载 three.js CDN）。建议体验路径：

1. 场景剧本依次点 🩺巡检 → 🚨MoE 越界诊断 → 🐢找慢副本 → 🔬单机深潜；
2. 顶栏「🎛 视图」面板切连线镜头到「互联流量」→ 悬停右侧矩阵卡的格子/介质条，看矩阵⇄3D 双向联动；
3. hover 右下 HUD 的 step 时间分解「通信」行，看软件预期⇄硬件连线对照；
4. 左侧层级行从 L4 一路点到 L2，看连线粒度带换挡（全景 → 本机视角 → 卡角色），重点试 PP 和 EP 镜头；
5. 再试魔方形态的五种排列切换。

## 目录

```
index.html          当前版本（=history/v22）
vendor/pto-design-system/  PTO 设计系统 token/css（foundation·semantic·components·style）
CLAUDE.md           给 Claude Code 的工程上下文（架构图、状态机、约定、陷阱）
docs/
  上游设计文档_训练监控可视化.md    产品/领域设计（12 章：并行策略、NCCL/HCCL、双模式…）
  交互与联动设计文档.md            交互事实源（原则、状态机、粒度带、联动矩阵、版本决策、路线图）
  设计系统与数据接入规划.md        下一阶段施工图：Token/组件化/工程化路径 + DataSource 接口与真实数据替换顺序
history/            v2(Gemini)→v22 全部版本，docs 第 10 章有各版关键决策
tools/harness.js    jsdom 运行时回归测试（npm i jsdom three@0.128.0 && node tools/harness.js）
```

## 版本脉络（细节见 docs/交互与联动设计文档.md 第 10 章）

v4 统一 3D 单相机 · v5 Smartscape 下钻 · v6 层级反馈+L4 机柜 · v7 模式 B+SP/CP ·
v8 任务透镜+魔方多排列 · v9-v10 联动修复+L5 语义 · v11-v12 IDE 分屏+泳道动态化+遮挡治理 ·
v13 魔方通信线 · v14 场景剧本 · v15 TP/PP 专属排列 · v16 软硬形态映射 · v17 业界范式（计数器轨/通信矩阵/Top5 分诊） · v18 NCCL 语义深化（PXN 消息路径镜头/散射对比/双 Channel/算法 HUD） · v19 真实数据注入（Pangu Pro MoE 72B-A16B，arXiv:2505.21411）+ 排列选型论证 · v20 接入 PTO 设计系统（默认 light）+ 顶栏按钮分组下拉 · v21 彻底 light 化（3D 光照/网格/机柜/Host 材质、浮层玻璃底、Mode-B 画布、压暗色）+ 关系筛选选中态实心强化 · **v22 软硬结合连线镜头**——五镜头重做（粗管宽度=流量、统一状态色、方向粒子、语义标签、Manhattan 走线、PXN 粒子速度=时延）+ 互联矩阵叠加卡（Host×Host×介质条 ⇄ 3D 交叉高亮/过滤）+ HUD 工况 step 分解联动 + 层级 LOD 粒度带（pod/field/cab/chip/die，L3=本机视角、L2=卡角色）+ 顶栏单行化（场景剧本直出 + 🎛视图集体配置面板 + 🧪演练）+ 泳道回放归位时间轴 + 左栏画布 HiDPI + Pod/SN 选中态状态驱动
