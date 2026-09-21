# 开源 3D 模型下载 · 转换 · 归档（HPC Topology Viewer）

这个目录是各视图的**模型仓库**。`model-registry.ts` 用
`import.meta.glob('./models/*.glb')` 自动扫描本目录，**约定文件名 = 零件的
`CatalogPart.id`**（见 `../parts-catalog.ts`）。把命名正确的 `.glb` 丢进来，对应
视图就会自动用它替换程序化几何——**无需改任何代码**，删掉文件即回退程序化几何。

> 命名为什么不带品牌？本仓库刻意**不在源码/产物里出现明文品牌名**（见根
> README 的「Content encoding」）。所以 part id 一律用**通用、去品牌**的名字，
> 模型文件名也只用这些 id，保持仓库 scrape-clean。

---

## 0. 一句话工作流

```
下载模型 → (必要时)转 GLB → 重命名为 <part-id>.glb → 丢进本目录 → 刷新页面
```

- **尺寸不用管**：加载器自动把模型等比缩放并居中到该零件在场景里的槽位。
- **朝向**要大致对：世界轴 **Y 上**、**Z = 前→后**、X = 左右。歪了就在
  `parts-catalog.ts` 给该零件加 `modelRotationDeg: [x, y, z]`（加载时在缩放前应用）。

---

## 1. 转换命令（在仓库根目录）

```bash
# glb / gltf / obj / stl —— 通用
node scripts/convert-to-glb.mjs <输入文件> <part-id>

# step / stp —— 全自动（OpenCASCADE WASM，无需 FreeCAD）
node scripts/step-to-glb.mjs <输入.step> <part-id>

# iges / x_t(Parasolid) / FCStd 等 —— 先在 FreeCAD/CAD 另存 STEP 或导出 GLB，再用上面两条
```
默认**不开 Draco**（运行时无需外部解码器，契合本项目离线 / noindex 部署）；
想压体积加 `--draco`。

---

## 2. 零件下载对照表（清单）

> 文件名列 = 你要保存到本目录的名字。来源优先勾选 **Downloadable + CC-BY / CC0**。
> ★ = 出现频率最高、最值得先做。

| ★ | 零件 | 文件名 | 真实参考尺寸 mm (w×h×d) | 用到的视图 | 推荐来源 / 关键词 |
|---|---|---|---|---|---|
| ★ | NPU 加速模组（OAM/夹层卡） | `npu-accelerator-module.glb` | 102×50×165 | 节点 · UB层级(L0/L1) · 邻接矩阵 · 阵列全景 | Sketchfab/GrabCAD：`OAM module`,`AI accelerator OAM`,`GPU module` |
| ★ | 服务器 CPU 封装（LGA） | `cpu-server-package.glb` | 75×8×75 | 节点 | Sketchfab/GrabCAD：`server CPU`,`LGA package`,`datacenter processor` |
| ★ | 计算刀片 / 液冷节点托盘 | `compute-blade.glb` | 880×130×740 | 机柜 · UB层级(L2) · 阵列全景 | GrabCAD：`server blade`,`liquid cooled tray`,`GPU server tray` |
| ★ | 计算机柜（19" 整柜） | `cabinet-compute.glb` | 600×2235×1200 | 全景总览 · UB层级(L3) | GrabCAD/OCP：`server rack 42U`,`ORV3 rack`（**低面数**，总览有上百实例） |
|  | 通信柜（UB 交换整柜） | `cabinet-switch.glb` | 600×2235×1200 | 全景总览 | 同上，可同款换面板配色 |
|  | DDR5 RDIMM 内存条 | `mem-ddr5-rdimm.glb` | 133×31×4 | 节点 | GrabCAD：`DDR5 RDIMM`/`DDR4 DIMM` STEP（KiCad packages3D 实测无 DIMM 库）|
|  | 光模块 / OSFP 收发器 | `optic-osfp-module.glb` | 100×17×22 | 节点(光口) · UB交换 | TraceParts/GrabCAD：`OSFP`,`QSFP-DD`,`optical transceiver 800G` |
|  | DPU / 网卡（PCIe 卡） | `dpu-nic-card.glb` | 167×19×69 | 节点 | GrabCAD/OCP：`PCIe NIC half height`,`OCP 3.0 mezzanine` |
|  | UB 交换托盘 / 线卡 | `ub-switch-line-card.glb` | 500×88×700 | 机柜(通信柜交换单元) | GrabCAD：`switch line card`,`1U switch tray` |
|  | 电源框 / PSU（CRPS） | `psu-crps-shelf.glb` | 440×88×700 | 机柜(供电单元) | GrabCAD：`CRPS power supply`,`server PSU`,`power shelf` |
|  | 液冷分集水器 / CDU 歧管 | `cdu-liquid-manifold.glb` | 500×200×150 | 机柜(液冷单元) | GrabCAD：`liquid cooling manifold`,`CDU`,`coolant distribution` |

> 上面这张表与 `../parts-catalog.ts` 一一对应；改了那里记得同步这里。

---

## 3. 下载源

| 来源 | 登录 | 格式 | 适合 |
|---|---|---|---|
| **Sketchfab** | 免费 | 直接 GLB | 外观好看的整件（CPU/模组/机柜），筛 CC-BY/CC0 |
| **GrabCAD** | 免费 | STEP/IGES | 工业件齐全（刀片/电源/机柜/导轨/歧管），逐件看许可 |
| **TraceParts / 3DContentCentral** | 免费 | STEP/GLB | 真实型号连接器、光模块、滑轨 |
| **KiCad kicad-packages3D** (GitLab) | 免登录 | STEP/WRL | 连接器 / 芯片封装（注意：**无 DIMM/DDR 库**）|
| **OCP** opencompute.org | 免登录 | STEP | OCP 网卡 / ORV3 机架 / 托盘 |

---

## 4. 归档规范

把原始下载件 + 许可证存到 `_sources/<part-id>/`（本目录 glob **不递归**，不会误加载）：

```
models/
  npu-accelerator-module.glb        ← 运行时加载（自动检测）
  _sources/
    npu-accelerator-module/
      original.step                 ← 原始 CAD
      LICENSE.txt                   ← 许可证 / 来源 URL / 作者
      notes.md                      ← 缩放 / 朝向调整记录
```

许可优先级：**CC0**（随便用）> **CC-BY**（需署名）> CC-BY-SA（传染）。厂商官方
CAD 通常允许设计用途，但不可二次售卖模型本身——逐个看模型页。

---

## 5. 验证 & 排错

1. 放好文件 → 刷新页面 → 对应视图该零件应显示真实模型并与周围协调。
2. 仍是程序化几何 → 检查文件名是否与第 2 节 `part-id` **完全一致**（大小写、连字符）。
3. 朝向歪了 / 太扁太长 → 在 `parts-catalog.ts` 给该零件加 `modelRotationDeg`，
   或把该零件的 `<ModelOr fit="stretch">`（默认 `contain` 等比，`stretch` 撑满槽位）。
4. 总览 / 阵列全景很卡 → 机柜模型面数太高；换 **低面数**版本（这两个视图会实例化上百个）。
