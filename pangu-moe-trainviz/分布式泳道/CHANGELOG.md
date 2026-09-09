# 更新记录

- 2026-09-09：升级 training-timeline.v2，保留 ST 的 reduce-scatter / clip / optimizer / params-all-gather 参考形状；新增参数组、optimizer state shard、collective group 和独立训练运行时图，完成 Optimizer 事件双向联动。
- 2026-09-09：Rank 展开后移除重复摘要通信；Optimizer 保留真实时长并设 4px 最低可见宽度；补齐 Pro 可见节点到演示事件的映射；dropdown 改用 input tokens，不再伪装为按钮。
- 2026-09-09：事件按绘制区间分轨，激活驻留独立展示；展开的 DP/PP 不再叠画所有 Rank。删除默认状态摘要，hover 缩为一句，info 改用浅色 panel-shell 与系统链接按钮。
- 2026-09-09：默认全部 DP；修复 Pro 的 L1 / L2 结构展开；泳道行头去除 tag 边框、完整显示 Rank 名称并对齐展开按钮；补充行头与事件 / 并集 hover 说明。
- 2026-09-09：按轻量 demo 范围收敛数据：仅 L2 典型算子，其他层 / 输出头为概览；保留 8 MB/DP 的完整 step 调度、通信和交互。撤销全层明细方案，保留 schema + 离线生成器 + JSON 消费前端。
- 2026-09-09：恢复完整 step / 全部 MB 视角，拆分 MB 焦点与结构选择，修复并集边界及通信参与者投影；ST 保留为研究证据。
