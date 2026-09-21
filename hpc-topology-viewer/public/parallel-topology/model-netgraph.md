# 模型整网视图（真图展开）· Pattern

> 构建产物：`/patterns/model-netgraph/pattern.html`（自包含，带 `?v=<短SHA>` 版本戳）
> 源：`public/parallel-topology/demo.html`，默认落在 `view=chain · cuts=pcte · rank=0 · vtab=side · embed=1`
> 契约：同级 `pattern.json`

## 它回答什么

**一张整网是怎么被切开的。** PP / TP·SP / EP 都落下、**不落 DP**、不进卡维度——
所以画面上是「切分本身」，不是切完之后的 world 张卡。

默认侧视：全部层沿层深铺开、PP Stage 分段、Dense→MoE 换挡带、mHC residual 主干；
每个算子旁标着它被哪把刀沿哪根张量轴切（SP 切 seq、TP 切 hidden、EP 切 expert），
以及这一层这一算子的读数。切到正视（`vtab=front`）就是**一层之内**的算子 DAG。

它**不**回答「这一份落到哪张卡上」——那是 `/patterns/rank-topology-3d/` 的题面。

## 口径（这一条最要紧）

| 这一屏的东西 | 是不是真的 |
|---|---|
| 算子清单、层级、连边 | **真数据**：openPangu 真图 |
| 每个算子被哪把刀沿哪根轴切、通信原语 | **真数据**：net-sharding 的切分表 |
| 算子框的**宽度** | 按类别的粗略权重（线性层 ≫ norm/逐元素），图上标明口径 |
| σ / Amax / ms / GB/s 这些读数 | **示意口径**，表达量级与相对关系，不是实测 profiling |

## 标注的取舍

标注按可读性分档出现，规则只有一条：**框装不下名字时优先写值**。
名字在悬停与明细卡里都拿得到，而数值不画在图上就真的看不到；再窄就只剩形状与颜色。
放大时字跟着图一起长（到 2.4 倍封顶），不会出现「越放大字越小」。

## 常用 URL

| 想看什么 | URL |
|---|---|
| 默认（按层深） | `pattern.html` |
| 一层之内的算子 DAG | `?vtab=front` |
| 收成功能块（不逐算子展开） | `?mods=1` |
| 只看结构（关掉所有数据标注） | `?mv=` |
| 第 29 层住在哪几张卡上 | `?layer=29` |
| 把 DP 也落下（不再是「一张整网」） | `?cuts=dpcte` |

## 约束与不适用

- **不落 DP 是这一屏的题面**。要看「同一份被复印了多少遍」，去 `rank-topology-3d`。
- 层数很多时（48 层以上）侧视会超出画布宽度，需要横向拖动——那是取景，不是渲染缺陷。
- 不是 profiling 视图：这里的 ms / GB/s 用来比较量级，不能拿去做性能结论。
- 换模型架构需要同时更换 `model-graphviz` 的真图与 `net-sharding` 的切分表，两者必须同一份来源，否则算子对不上切法。

## 嵌入

```html
<iframe src="/patterns/model-netgraph/pattern.html?embed=1&vtab=side"
        style="width:100%;height:100%;border:0"></iframe>
```

宿主可用 `postMessage({type:'pto:state', view, sel, theme})` 推状态；
不认识的参数一律忽略，非法值退回默认，不报错。
