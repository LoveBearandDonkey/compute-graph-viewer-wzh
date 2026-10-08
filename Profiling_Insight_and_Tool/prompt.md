# 队列

1、为什么我手动打开“内存观测”就要暂停播放，理论上，我调整画布里的视角并不改变配置，不应影响播放；2、取消rank选中后，cs3-side  也要消失。反过来说，点击关闭cs3-side  也意味要取消rank选中


[rank-intro.html](d:/Projects/compute-graph-viewer-wzh/Profiling_Insight_and_Tool/rank-intro/rank-intro.html) 8·内存 中OOM页签，右侧详情栏：1、EP0、EP1代表什么？和rank什么关系？2、EP3怎么是 4.8× 均值  ，我看EP3的 4916是其他463的10倍而不是4.8倍。这里的4915是什么？3、rankmem   中冒出来的MoE dispatch·临时区  、L7 · 路由倾斜异常增量  、MoE 路由后中间激活  3个，到底归属非oom时rankmem的哪个类？总不能说oom时我说我的内存又多几个类型这么随意吧，如果属于某个类，就要像attn.kv归到权重一样有缩进线；4、MoE 中间激活  和 路由倾斜异常增量  分别是什么，怎么产生的，用于做什么？5、 allocator 碎片  是什么意思？；6、OOM、内存峰值分析、GAP 分析页签的右上角播放按钮不需要时红色


第 4 份走到 L7 时跨过 64 GB。但是，memCycleCanvas为什么把红线画在L7的左侧
 
9·通信 中：1、桑基线不需要闪动，现在不需要体现通信前后关系，因此只需要静态展示桑基线即可；2、comm-timeline默认折叠，不展示下面的色点表；3、这些桑基线和训练常见的问题有什么关系，能用于体现什么问题场景吗？






增加锁定某表单、单改模式



# 规范遵从
请完整阅读并遵从 D:\Projects\pto-design-system-wzh

包括
D:\Projects\pto-design-system-wzh\SKILL.md
D:\Projects\pto-design-system-wzh\references\DESIGN.md

页面必须使用'patterns/ide-frame'作为整体框架

必须优先复用现有pattern：
D:\Projects\pto-design-system-wzh\patterns\pattern.json

最后，按D:\Projects\pto-design-system-wzh\SKILL.md中的“工作流 B”优化 @xxx

如果任务过重，可以先列出步骤，我们一步一步完成



