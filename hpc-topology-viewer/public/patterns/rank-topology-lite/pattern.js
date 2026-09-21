/* rank-topology-lite · pattern.js
   规模够大才值得用逻辑魔方当默认层（世界卡数 world ≤ 64 时，矩阵本体自己
   一屏就够清楚——64 张卡的 SVG 阵列不存在密度/性能问题，逻辑魔方那层
   "先看形状"的价值这时反而是多绕一圈）。按这条规则分两条路：

   world ≤ 64：直接铺满并行拓扑矩阵本体的原页（不传 fastcard/mono/solo，
     就是 rank-topology-3d 独立打开的样子），没有逻辑魔方、没有返回按钮、
     没有三档——矩阵自己的"选中/取消选中"手势已经够用，不必再包一层。

   world > 64（?preset=pangu/moe718b128k/incident2048，非默认——见下面
   「默认预置」那段注释，规模先收回到 64 卡，等交互形式定下来再考虑扩大）：
   "无限画布"分三档取景，档位越往里，画得越少、看得越细：
     1 集群       —— 逻辑魔方铺满，没有选中任何卡。
     2 同组定位   —— 逻辑魔方里点一张方块，**留在逻辑魔方自己身上**：它自带
                     的"卡内魔方"局部聚焦（选中卡与它所在的行列一起被摄像机
                     框住）已经是干净的效果，不必换到矩阵那一屏再画一遍——
                     矩阵在这个规模下要给每张兄弟卡都摆通信芯片/容量告警，
                     牌子挤在一起反而更花；但 cc=0/cclabels=0 把逻辑魔方
                     画布里那份"层/算子构成"的显示关掉了（见 rubikParams
                     的注释），宿主右下角这张浮卡要接住这句话，不能真的
                     只剩一句邀请——选中的瞬间先摆邀请（rank/坐标/层区间，
                     不留空白），紧接着借矩阵本体一次不铺屏的 ?brief=1&sel=
                     请求（requestTier2Brief），回信一到就原地升级成跟
                     第三档一样的层区间/显存构成明细，读者不用先点"下钻"
                     才看得到这些数字。
     3 单卡下钻    —— 点那句邀请，才真正换到并行拓扑矩阵本体（fastcard=1&
                     solo=1）：连兄弟卡也隐去，看的是这一张卡内部的填充版
                     ——真实容量读出、显存构成明细（保留原本的彩色：权重/
                     梯度/优化器态/专家/激活各自的颜色在说"这一块字节是
                     什么"，现在只服务一张卡，不会跟别的卡的颜色打架）、
                     卡内那一级通信。
   退档不靠宿主自己另起一颗按钮：点空白就是每一档自带的手势——逻辑魔方
   点空白取消选中（第 2→1 档），矩阵点空白退出 soloCard（第 3→2 档，
   一句 pto:tier 指令，不重新加载 iframe）。"← 返回…"那颗按钮删了（反馈
   原话"有了面包屑不要这个了"）：档位已经写在面包屑里（逻辑魔方顶栏的
   模型名 + rank 后缀、矩阵自己的画布名字），退档交给两个 iframe 原生就有
   的手势，不必再摆一层复述"你在哪/怎么回去"的按钮。tier2 完全不涉及
   iframe 切换，天然没有闪烁——切换的"丝滑"就是靠少切一次做到的，不是靠
   更长的过渡动画补出来的。
*/
(function () {
  'use strict';

  var qs = new URLSearchParams(location.search);

  var rubikFrame = document.getElementById('rubikFrame');
  var matrixFrame = document.getElementById('matrixFrame');
  var briefCard = document.getElementById('briefCard');
  var clusterBadge = document.getElementById('clusterBadge');
  var universeStage = document.getElementById('universeStage');
  var universeToggle = document.getElementById('universeToggle');

  /* 两档预置，world = tp×pp×dp（EP 折在 DP 内部，不进世界卡数——两个本体
     的 README 都确认过这个口径）。默认盘古 ProMoE，world=4000，走三档取景；
     ?preset=dense64 是 demo.html 自己现成的 64 卡预置，用来验证"world ≤ 64
     直接显示矩阵原页"这条规则确实会触发，不是摆着不用的死分支。
     modelName：全部命名/面包屑的唯一来源——不编一个新名字，直接抄 demo.html
     自己 PRESETS 数组里给这个 preset 起的名字（那份是"模型叫什么"这件事的
     权威出处），这一层不重复维护第二份。 */
  var PRESETS = {
    pangu: { tp: 8, pp: 5, dp: 100, ep: 2, matrixPreset: 'pangu', modelName: '盘古 ProMoE' },
    dense64: { tp: 4, pp: 4, dp: 4, ep: 1, matrixPreset: 'dense64', modelName: '稠密预置' },
    /* incident2048：demo.html 那份「2048卡·Router溢出复盘」预置的桥接条目——
       tp/pp/dp/ep 取值与 demo.html PRESETS 里 incident2048 的注释同一份推算
       （world=pp×edp×ep=4×8×64=2048 → dp=edp×ep=512、ep=64），两边用同一组
       数字，rank 号才能对得上。见下面 INCIDENT 数据块与 renderIncident()。 */
    incident2048: { tp: 1, pp: 4, dp: 512, ep: 64, matrixPreset: 'incident2048', modelName: '2048卡·Router溢出复盘' },
    /* moe718b128k：demo.html 那份 PRESETS.moe718b128k 的桥接条目，tp/cp/pp/dp/ep
       逐位照抄那边的 cfg（world=tp·cp·pp·dp=8·16·16·4=8192；dp=4 不是 1——
       逻辑魔方自己的模型要求 EP 必须整除 DP 本身，dp=1 时 ep(4) 除不尽会
       直接抛异常建模失败，dp=4 是两边约束都满足的最小值，demo.html 那份
       PRESETS 的注释里有完整推导）。这是第一个 cp>1 的桥接预置，之前
       pangu/dense64/incident2048 都是 cp=1（省了这个字段也一样），这档
       必须显式给 cp，不然逻辑魔方按 cp=1 建模型，跟矩阵本体的四维结构
       对不上、rank 换算全错。世界卡数公式与 rubikParams/
       rubikSelToMatrixSel 里补的 cp 项，见下面对应位置的注释。 */
    moe718b128k: { tp: 8, cp: 16, pp: 16, dp: 4, ep: 4, matrixPreset: 'moe718b128k', modelName: 'MoE 718B(A39B)·128K序列' }
  };
  /* 面包屑第二段：反馈「面包屑应该是3层」「这一层没有对应的面包屑」——
     原来选中之后不管第二档（留在逻辑魔方，选中卡与它所在的并行组）还是第三档
     （下钻到矩阵），面包屑都只写"模型名 / rank N"两段，第二档这一步在面包屑里
     直接被跳过了。这里补成第三段，两处（逻辑魔方招牌的 tierlabel、矩阵
     stitle 的对应写法）用同一个字符串，读起来是同一句话。
     原来叫"选中+兄弟"，反馈"给一个更好的名称，这个太随意了"——那句读着
     像随手写的笔记（还带个"+"号），换成"同组定位"：这一档做的事就是把
     选中卡摆进它所在的 TP/PP/DP 并行组里定个位，还没下钻到字节级详情，
     名字直说这件事，不再是简写。 */
  var TIER2_LABEL = '同组定位';
  /* 默认预置：先后改过两次。第一次反馈「改这里的默认配置」，从 pangu
     （4000卡演示规格）换成 moe718b128k（pangu_sophon_pytorch 项目里体量
     最大的一档真实 MoE）；随后反馈"这个的rank数量太多了……回退一步回到
     之前只用64个rank的时候，也就是并行拓扑本身的pattern的配置"+"等到
     我们的形式确定之后再扩大rank的数量"——8192 卡（尤其宇宙视图一屏 16
     段×9 叶子）密度已经压过"先把交互形式定下来"这个当下的目的，退回到
     dense64（world=64，demo.html 自己现成的稠密预置，就是并行拓扑矩阵
     本体自己那份配置，不是这一层另起的）。moe718b128k 不是删掉，只是不
     再是默认——?preset=moe718b128k 仍旧可以直接打开看那档真实数据，形式
     定下来之后再考虑要不要重新扩大默认规模。旧链接 ?preset=pangu/dense64/
     incident2048/moe718b128k 都照样认得。 */
  var PS = PRESETS[qs.get('preset')] || PRESETS.dense64;
  /* world 公式补上 cp：原来只有 tp×pp×dp，pangu/dense64/incident2048 都是
     cp=1（省了这个乘数结果一样），moe718b128k 是第一个 cp>1（=16）的桥接
     预置，不补的话这里算出的卡数只有真实 world 的 1/16，逻辑魔方与矩阵
     本体从一开始就对不上。PS.cp 缺省仍按 1 处理，旧预置的 world 逐位不变。 */
  var world = PS.tp * (PS.cp || 1) * PS.pp * PS.dp;

  /* ══════════════════════════════════════════════════════════════════════
     真实故障复盘数据（仅 preset=incident2048 时出现）——反馈「这里真实监控
     数据卡片放到哪个屋里」选了"总览+单卡下钻都要，带时间线联动"。
     逐字抄自 /incident-canvas/index.html 的 GROUPS / TRAIN_METRICS / BOARD /
     TRAIN_HOOKS（那份本身照抄 compute-graph-viewer 的 INCIDENT_GROUPS，数值
     一个没改）——这里只挑了卡片要用的字段（不带 mechanism 分相/chart 曲线，
     那部分是 incident-canvas 自己的画布长项，这一层不重新实现一遍），事件
     顺序、conclusion 原文、BOARD 每格的 v/s/why 逐位照抄，没有改写。
     这是**另一次独立的 2048 卡训练**的真实事故（见 incident2048 预置注释），
     不是"当前选中卡正在发生的事"——两侧显存构成卡（renderMemCards）用的是
     本页自己按架构字段估出的假设值，跟这里引用的事故原文数字并不是同一套
     口径，两者一起出现时不要混着读。 */
  var INCIDENT_PROBLEMS = [
    { id: 'problem-2', name: '问题2 · Router 溢出与通信死锁',
      lede: '报错点在 HCCL 通信超时，震中却在 Layer 38 的 Router——一次 FP8 数值溢出，经 EP barrier 与 PP 依赖扩散成 2048 卡停摆。',
      events: [
        { id: 'p1-warning', time: '15k', dim: '数值·预警', sev: 'warn', title: 'Loss scale 连续衰减',
          conclusion: 'Layer 38 的数值健康已提前恶化，AMP scaler 从 65536 衰减到 4096。' },
        { id: 'p1-nan', time: '15203', dim: '耗时·数值', sev: 'bad', title: 'Loss NaN / grad_norm Inf',
          conclusion: '异常只在多卡复现，Layer 38 是首个数值病灶候选。' },
        { id: 'p1-log', time: '+8ms', dim: '通信·日志', sev: 'bad', title: 'Plog 暴露 buffer 失配', rank: 1559,
          conclusion: '运行时 EP rank 23 的 send=0、recv=9832；通信报错同时携带 router_logits Inf 证据。' },
        { id: 'p1-a2a', time: '+30s', dim: '通信·耗时', sev: 'bad', title: 'All-to-all 超时，63 rank 空等', rank: 1559,
          conclusion: 'EP rank 23 是首个阻塞者，其余 63 个 EP rank 是 barrier 受害者，不应被判为 64 个独立根因。' },
        { id: 'p1-root', time: '-30s', dim: '数值·负载', sev: 'bad', root: true, title: 'Router FP8 溢出，E193 吸收 98% token',
          conclusion: '这是问题2的根因事件：FP8 softmax 溢出导致路由塌缩，而不是 HCCL 自身故障。' },
        { id: 'p1-spread', time: '+30.1s', dim: '通信·扩散', sev: 'bad', title: 'PP3 断裂，2048 NPU hang',
          conclusion: '报错点是通信 timeout，异常震中却在 Layer 38 Router；单点经 EP barrier 和 PP 依赖扩散至整网。' }
      ] },
    { id: 'problem-1', name: '问题1 · 显存峰值与碎片 OOM',
      lede: '显存从 55 GB 一路爬到顶：12 层激活的存活区间在前向末尾全部重叠，叠上 LM Head 的 logits 把 64 GB 顶满，最后死在一次 0.5 GB 的临时申请上。',
      events: [
        { id: 'p2-rise', time: '8000+', dim: '显存·趋势', sev: 'warn', title: '显存从 55 GB 持续爬升',
          conclusion: 'PP stage 3 的显存不再回落，吞吐同期下降 12.5%。' },
        { id: 'p2-cost', time: '12000', dim: '耗时·显存', sev: 'warn', title: '分配/释放 API 占时 7.4%',
          conclusion: '显存管理耗时 890 ms，明显高于正常值 2%；带宽利用率 78%，可排除纯带宽瓶颈。' },
        { id: 'p2-peak', time: '12000', dim: '显存·容量', sev: 'bad', title: '激活值占用 36.2 GB',
          conclusion: '激活值占峰值的 56.6%，是唯一可大幅缩减的组成。' },
        { id: 'p2-layer', time: '12000', dim: '显存·Layer', sev: 'warn', title: 'L38 单层激活达到 1.2 GB',
          conclusion: 'Layer 38 比普通 Dense 层高 1.7 倍，额外占用来自 expert dispatch buffer。' },
        { id: 'p2-oom', time: '12003', dim: '显存·OOM', sev: 'bad', root: true, title: 'EP rank 17（global rank 1553）触顶并发生碎片 OOM', rank: 1553,
          conclusion: '64/64 GB 容量不足是主因，83% 碎片率让 0.5 GB 临时 buffer 更早申请失败。' }
      ] }
  ];
  var INCIDENT_METRICS = [
    { k: 'loss',    name: 'lm loss',            want: '↓',    src: 'loss_func() → training_log()' },
    { k: 'gnorm',   name: 'grad_norm',          want: '稳定',  src: 'training_log()' },
    { k: 'lscale',  name: 'loss_scale',         want: '不触发', src: 'logger_and_track_metrics_callback.py:74' },
    { k: 'zeros',   name: 'num_zeros_in_grad',  want: '↓',    src: 'training_log()' },
    { k: 'tflops',  name: 'throughput',         want: '↑',    src: 'PretrainMetricConfig._compute_throughput' },
    { k: 'mfu',     name: 'MFU',                want: '↑',    src: 'PretrainMetricConfig._compute_mfu' },
    { k: 'tokday',  name: 'throughput_per_day', want: '↑',    src: '_build_log_dict:1505' },
    { k: 'steptime',name: 'elapsed time / iter',want: '↓',    src: '_build_log_dict:1491' },
    { k: 'lr',      name: 'learning_rate',      want: '按计划', src: 'lr scheduler（cosine / WSD）' },
    { k: 'mem',     name: 'mem_reserved_bytes', want: '平稳',  src: 'NPU 保留显存 / theoretical_memory' }
  ];
  var INCIDENT_HOOKS = [
    { k: 'nan',       name: 'NaN / Inf 检测',   src: 'check_for_nan_in_loss_and_grad', act: '任一 rank 的 loss 出现 NaN 直接报错退出' },
    { k: 'spike',     name: 'Loss Spike 监控',  src: 'loss_spike_monitor_callback',    act: '损失突然飙升时触发回调' },
    { k: 'heartbeat', name: 'Heartbeat 监控',   src: 'init_heartbeat_monitor_pid',     act: '训练进程无响应则重启' },
    { k: 'dataspeed', name: '数据生产速度告警', src: '_warn_data_production_speed',    act: '数据加载速度接近训练速度时告警' },
    { k: 'oom',       name: 'OOM 前兆',         src: 'mem_reserved_bytes 增长趋势',    act: '保留显存持续增长，容量见底' }
  ];
  var INCIDENT_BOARD = {
    'p1-warning': { hooks: ['spike'], m: {
      lscale: { v: '65536 → 4096', s: 'warn', why: '连续四次减半，越过三级预警线 8192' },
      loss: { v: '仍在正常区间', s: 'ok', why: '距崩溃尚有 53 step——这正是它作为预警的价值' } } },
    'p1-nan': { hooks: ['nan', 'spike'], m: {
      loss: { v: 'NaN', s: 'bad', why: '本轮梯度整段作废' },
      gnorm: { v: 'Inf', s: 'bad', why: '末段每 step 涨约一个数量级，越界在同层反向' },
      lscale: { v: '已退到 4096', s: 'warn', why: '再退也救不回来，溢出的是 logits 不是梯度尺度' } } },
    'p1-log': { hooks: [], m: {
      steptime: { v: '+8 ms 起', s: 'warn', why: '收发失配刚发生，还没变成等待' } } },
    'p1-a2a': { hooks: ['heartbeat'], m: {
      steptime: { v: '+30 000', s: 'bad', why: '等满 HCCL 超时阈值 30 s' },
      tflops: { v: '0（63 卡）', s: 'bad', why: '空等期间算力零产出，而日志上什么都不报' },
      mfu: { v: '0', s: 'bad', why: '同上——这正是「看起来通信很慢」最容易骗人的地方' },
      tokday: { v: '0', s: 'bad', why: '累计空转 63 × 30 s ≈ 1890 卡·秒' } } },
    'p1-root': { hooks: [], m: {
      zeros: { v: '247 / 256 专家无梯度', s: 'bad', why: '路由塌缩后它们再没收到过 token' },
      loss: { v: '—', s: 'na', why: '本事件采的是路由份额与 logits，不在这十格里' } } },
    'p1-spread': { hooks: ['heartbeat'], m: {
      steptime: { v: 'hang', s: 'bad', why: '依赖环闭合，4 个 stage 全停在等待上' },
      tflops: { v: '0（2048 卡）', s: 'bad', why: '99.95% 的卡只是被链条拖住的' },
      mfu: { v: '0', s: 'bad' }, tokday: { v: '0', s: 'bad' } } },
    'p2-rise': { hooks: ['oom'], m: {
      mem: { v: '55 → 63.7', s: 'bad', why: '4000 step 未回落，被留住的是一直活着的激活' },
      tokday: { v: '3200 → 2800 tokens/s', s: 'warn', why: '同期吞吐下降 12.5%' },
      tflops: { v: '同比 −12.5%', s: 'warn', why: '原文给的是 tokens/s，这一格按同一口径读' } } },
    'p2-cost': { hooks: [], m: {
      steptime: { v: '12 000', s: 'warn', why: '其中 890 ms（7.4%）花在显存分配/释放上，正常水位约 2%' },
      mem: { v: 'HBM 带宽 78%', s: 'ok', why: '可排除纯带宽瓶颈——贵在碎片整理与换页，不在搬数据' } } },
    'p2-peak': { hooks: ['oom'], m: {
      mem: { v: '64.0 / 64', s: 'bad', why: '激活占 56.6%，安全余量 0 GB' } } },
    'p2-layer': { hooks: [], m: {
      mem: { v: 'L38 单层 1.2', s: 'warn', why: '同段普通层 0.71 GB，多出来的 0.5 GB 来自 expert dispatch buffer' } } },
    'p2-oom': { hooks: ['oom'], m: {
      mem: { v: '已分配 60.1 / 64', s: 'bad', why: '碎片率 83%，最大连续块只有 0.32 GB' },
      steptime: { v: '中断', s: 'bad', why: '它一崩 PP3 就断，全网跟着停在等待上' } } }
  };
  var INCIDENT_SEVC = { ok: '#3FB950', warn: '#D29922', bad: '#F85149', na: '#6E6E6E' };
  var INCIDENT_SEVN = { ok: '正常', warn: '预警', bad: '告警', na: '未采' };

  // ── 真实故障复盘面板：直接摆在最外层拓扑上，不必先跳一次预置 ─────────────
  // 原来只在 ?preset=incident2048 才渲染，默认屏幕上只留一条「⚠ 真实故障
  // 复盘…」链接，点了才整页跳到 incident2048（另一份拓扑）才看得到数据。
  // 反馈「不希望问题定位的那一块儿和本身的拓扑是分离的，现在必须要点击
  // 左上角的告警才能进入有数据的界面，我希望这个界面直接显示在最外层的
  // 拓扑上」——这份数据本身（时间线/十格指标卡）跟当前正在看哪个预置的
  // 拓扑无关，是另一起独立训练的历史复盘，没有理由非要先跳转页面才能看到，
  // 所以这段渲染逻辑挪到 `world ≤ 64` 分流之前，任何预置打开都会显示（默认
  // 收起，跟以前一样，只是不再需要一次页面跳转才能展开）。
  //
  // 会跟着预置变的只有「下钻」按钮：事件里的 rank 号（1559/1553 这些）是
  // incident2048 那份 2048 卡拓扑自己坐标系里的真实 rank，当前预置不是
  // incident2048 时，这个数字在当前这张拓扑里根本不存在（比如默认的
  // dense64 只有 64 张卡）——不能假装它能在当前页面内下钻到同一张卡，那是
  // 编数据。所以按钮改成看当前预置：是 incident2048 就地下钻（跟以前
  // 一样）；不是的话，按钮改一句更诚实的说法并整页跳转到 incident2048（带
  // 上这个 rank 号），把读者带到这个数字真正有意义的那张拓扑上，不在当前
  // 页面里硬凑一个假坐标。
  var incidentPanel = document.getElementById('incidentPanel');
  var incidentSel = null;
  function incidentEventById(id) {
    for (var i = 0; i < INCIDENT_PROBLEMS.length; i++) {
      var evs = INCIDENT_PROBLEMS[i].events;
      for (var j = 0; j < evs.length; j++) if (evs[j].id === id) return evs[j];
    }
    return null;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function renderIncidentPanel() {
    if (!incidentPanel) return;
    var ev = incidentSel ? incidentEventById(incidentSel) : null;
    var board = incidentSel ? INCIDENT_BOARD[incidentSel] : null;
    var timelineHtml = INCIDENT_PROBLEMS.map(function (prob) {
      var dots = prob.events.map(function (e) {
        var on = e.id === incidentSel;
        return '<button type="button" class="ip-dot' + (on ? ' is-on' : '') + '" data-ev="' + e.id + '"'
          + ' style="--ip-sevc:' + INCIDENT_SEVC[e.sev] + '" title="' + esc(e.time + ' · ' + e.title) + '">'
          + '<span class="ip-dotmark"></span><span class="ip-dottime">' + esc(e.time) + '</span></button>';
      }).join('');
      return '<div class="ip-prob"><span class="ip-probname">' + esc(prob.name) + '</span><div class="ip-events">' + dots + '</div></div>';
    }).join('');
    // 十格指标卡只在选中了具体事件时才画——没选中事件时这十格清一色是「—」
    // 占位，不是数据。反馈「默认展开数据只展开有的数据」：面板默认展开（见
    // 下面去掉 is-collapsed），但只展开"有数据"的那部分——时间线本身随时
    // 都是真数据（11 个真实事件），默认就露出来；十格卡片在没选中事件之前
    // 一格数字都没有，展开了也只是十个「—」，不算「有数据」，索性不摆这块
    // 空壳，选中事件之后再补上，那时才真的有内容可展开。
    var cardsHtml = ev ? INCIDENT_METRICS.map(function (m) {
      var cell = board && board.m && board.m[m.k];
      var v = cell ? cell.v : '—';
      var sevKey = cell ? cell.s : 'na';
      return '<div class="ip-card" style="--ip-sevc:' + INCIDENT_SEVC[sevKey] + '">'
        + '<div class="ip-k">' + esc(m.name) + '<span class="ip-want">要求 ' + esc(m.want) + '</span></div>'
        + '<div class="ip-v">' + esc(v) + '</div>'
        + (cell && cell.why ? '<div class="ip-why">' + esc(cell.why) + '</div>' : '<div class="ip-src">' + esc(m.src) + '</div>')
        + '</div>';
    }).join('') : '';
    var onIncident = PS.matrixPreset === 'incident2048';
    var headHtml = ev
      ? '<span class="ip-evtitle">' + esc(ev.title) + '</span><span class="ip-evsev" style="--ip-sevc:' + INCIDENT_SEVC[ev.sev] + '">' + INCIDENT_SEVN[ev.sev] + '</span>'
        + (ev.rank != null ? '<button type="button" class="ip-drill" data-act="ip-drill" data-rank="' + ev.rank + '">'
          + (onIncident ? '下钻 rank ' + ev.rank + ' →' : '查看 rank ' + ev.rank + ' 所在的真实拓扑 →') + '</button>' : '')
      : '<span class="ip-evtitle ip-evtitle-empty">先在时间线上点一个事件——十格读数按那一刻的原文填，没采到的写「—」</span>';
    var concHtml = ev ? '<div class="ip-conc">' + esc(ev.conclusion) + '</div>' : '';
    incidentPanel.innerHTML =
      '<div class="ip-hd">' + headHtml + '<button type="button" class="ip-collapse" data-act="ip-collapse" title="收起/展开">' + (incidentPanel.classList.contains('is-collapsed') ? '▲' : '▼') + '</button></div>'
      + concHtml
      + '<div class="ip-timeline">' + timelineHtml + '</div>'
      + (ev ? '<div class="ip-cards">' + cardsHtml + '</div>' : '')
      + '<div class="ip-foot">口径来自 pangu_sophon_pytorch · 这十条是真的会被打印、画成曲线的那几个；另一次独立 2048 卡训练的真实复盘，与当前预置的架构字段无关，不替它编一个'
      + (onIncident ? '' : '——当前预置不是 incident2048，事件里的 rank 号在这张拓扑上并不存在，点"查看真实拓扑"会整页跳转')
      + '。</div>';
    incidentPanel.classList.remove('is-hidden');
  }
  incidentPanel && incidentPanel.addEventListener('click', function (ev) {
    var dot = ev.target.closest('[data-ev]');
    if (dot) { incidentSel = dot.getAttribute('data-ev'); renderIncidentPanel(); return; }
    if (ev.target.closest('[data-act="ip-collapse"]')) { incidentPanel.classList.toggle('is-collapsed'); renderIncidentPanel(); return; }
    var drill = ev.target.closest('[data-act="ip-drill"]');
    if (!drill) return;
    var rank9 = parseInt(drill.getAttribute('data-rank'), 10);
    if (PS.matrixPreset === 'incident2048') showDetail(rank9);
    else location.href = '?preset=incident2048&sel=' + rank9;
  });
  renderIncidentPanel();

  if (world <= 64) {
    /* 规模小：矩阵本体自己一屏就是全部——不铺逻辑魔方、不裁剪它的任何交互，
       与直接打开 /patterns/rank-topology-3d/ 逐字节相同。stitle 换成模型
       名称，跟三档取景那条路用的是同一个名字来源，不是另起一套说法。
       分隔符用 "/"：反馈「都放成面包屑用/分隔」，与下面 tier3 那条、
       逻辑魔方自己的招牌（见 pattern.js 的 syncBrand）三处统一成同一套
       写法，不是"这条 · 那条 /"各写各的。 */
    var plainP = new URLSearchParams({
      embed: '1', theme: 'dark', preset: PS.matrixPreset, card: '1', view: 'chain', vtab: '3d',
      stitle: PS.modelName + ' / ' + world + ' 卡'
    });
    matrixFrame.src = '../rank-topology-3d/pattern.html?' + plainP.toString();
    matrixFrame.classList.remove('is-hidden');
    rubikFrame.classList.add('is-hidden');
    return;
  }


  // ── 逻辑魔方：固定当前预置，深色主题 ──────────────────────────────────
  // color=neutral：默认就是素色（中性灰），不是负载热力橙→粉——这个简洁版要的
  // 默认态是"先看形状、不看颜色"，颜色留给选中/告警这些真正需要强调的状态。
  // 逻辑魔方自己独立打开（/rubik-pattern.html）默认仍是负载热力，这个参数只在
  // 这里传，不改它自己的默认值。
  // groupgap=3：拉开 tp/pp/dp/ep 各组之间的缝，这种规模下"这是几段/几片"才
  // 读得出来——独立打开的 /rubik-pattern.html 默认 1（原样间距），这个参数
  // 只在这里传，呼应"默认状态下参考并行拓扑拉大间距、让分组更明显"那条反馈。
  // brand=：逻辑魔方顶栏那块"逻辑魔方"招牌换成模型名称——同一条"用模型
  // 名称做全部命名"的规矩，这一层管得到的每一处都不留生造的产品名。
  // cclabels=0：收起"卡内魔方"那两枚钉在 3D 世界坐标上的字牌（行末算子名 +
  // 顶部"卡内 · L.."标题）——它们跟着相机转，规模一大会飘到这一层自己的
  // 悬浮数据卡/返回按钮那片地界上，跟已经在讲同一句话的右侧详情卡叠在一起
  // （反馈原话"不再这里显示只显示右侧卡片就好"）。彩色小格阵列本身照常画，
  // 少的只是文字；独立打开 /rubik-pattern.html 不受影响，默认还画这两枚牌。
  // axsel=0：选中一张卡时收起"贴在几何体上"那类轴刻度字牌（TP0/PP3 这种，
  // 世界尺寸固定）——第二档的镜头贴得极近（"局部聚焦"），这类字牌会占满
  // 大半个画布、糊住选中卡自己的坐标读出。坐标信息本来就写在 DOM 侧栏与
  // 悬浮数据卡里，画布里不用再重复一遍。
  // cc=0：选中一张卡时画布里那圈"卡内魔方"彩色小格阵列整个不画了——不只是
  // 字牌（cclabels 管那个），是格子本身。反馈原话"不是说去色的问题，是
  // 不要在画布中显示"：同一句话（rank / 层区间 / 对象持有情况）右侧详情卡
  // 已经摆得清清楚楚，画布这层不用再重复一份彩色阵列；"卡片还是保留彩色"
  // 指的是右侧详情卡与装载清单的颜色，那两处不受这条影响，独立打开
  // /rubik-pattern.html 也不受影响，默认还画这圈格子。
  // tierlabel=：面包屑第三段，见上面 TIER2_LABEL 的注释——独立打开
  // /rubik-pattern.html 不传这个参数，默认还是"模型名 / rank N"两段。
  // zoomsel=0.5：反馈「在这一步就做一个小的zoomin」附图是盘古预置选中一张
  // 卡后，那一列在 4000 卡满屏阵列里只有几个像素——选中时镜头往那张卡
  // 推近一半（不是矩阵那种贴近单卡的"局部聚焦"，这里镜头还是全景机位，
  // 只是缩小取景范围），取消选中飞回原机位。独立打开 /rubik-pattern.html
  // 不传这个参数，选中不受影响。
  // chrome=0：反馈「点击单卡会卡在这里」「去掉标签，下面的内容放到标题后面去」——
  // 附图是逻辑魔方自带的选中卡侧栏（.prc-info，"RANK"kicker+标题+键值表那一整套）
  // 在窄屏媒体查询下挪到画面底部，跟这一层自己的 briefCard/角标/corner-link 叠在
  // 一起，还用它的透明留白盖住了舞台——点上去点在了这张看不见的卡上，画布本身
  // 反而没反应，看着像"卡住了"。这一层右上角的 briefCard 早就把"选中的是哪张卡、
  // 什么坐标"说清楚了，.prc-info 与顶栏那一整套（形态/视角按钮、更多抽屉、图例）
  // 全是重复的第二份 chrome——直接让逻辑魔方自己那套别画，不止是这一个面板的
  // 样式问题。独立打开 /rubik-pattern.html 不传这个参数，默认还画，不受影响。
  var rubikParams = new URLSearchParams({
    /* cp：缺省按 1（PS.cp||1）——pangu/dense64/incident2048 没有这个字段，
       String(undefined) 会变成字面量 "undefined" 传出去，||1 兜底成
       rubik-pattern.html 自己的默认值，三档旧预置的取景逐位不变；
       moe718b128k 第一次真的用上非 1 的 cp。 */
    theme: 'dark', tp: String(PS.tp), cp: String(PS.cp || 1), pp: String(PS.pp), dp: String(PS.dp), ep: String(PS.ep),
    color: 'neutral', groupgap: '3', brand: PS.modelName, cclabels: '0', axsel: '0', cc: '0',
    tierlabel: TIER2_LABEL, zoomsel: '0.5', chrome: '0'
  });
  rubikFrame.src = '../../rubik-pattern.html?' + rubikParams.toString();

  // ── 宇宙视图：第一档的第二种画法，内联 SVG 径向星图（不是 iframe） ─────
  // 中心 = 模型本身；第一圈 = 各条 PP 段（沿用这个仓库里"PP流水=段"的既有
  // 心智模型，跟逻辑魔方 PP流水形态、并行拓扑矩阵段落条讲的是同一件事，
  // 不是另起一套分类）；每段外沿撒开一批采样出的**真实** rank 当叶子点——
  // 叶子的 tp/cp/rep 坐标是在 tp×cp×dp 这条扁平轴上等距抽样出来的，不是
  // 摆拍凑数量，点开哪一颗都能换算出一个真实存在的 rank。
  // 这里不重新发明"选中之后干什么"：叶子点点击算出 sel 直接调
  // showTier2/rubikSelToMatrixSel，跟逻辑魔方 postMessage 报上来的走的
  // 是同一条路径、落的是同一张 briefCard——两种视图只是"从哪触发选中"
  // 不同，选中之后的下钻/详情渲染一个字不重复实现。
  var universeMode = false;
  var universeBuilt = false;
  // 参考图那种暖金/紫青撞色，圈内按段循环取色，同一段的叶子跟着它所在的
  // hub 同色（深浅由 CSS 的 hover/dim 状态区分，不再按叶子逐个换色）。
  var HUB_PALETTE = ['#8B7CF6', '#4FC3D9', '#E8637A', '#F2B84B', '#5FD3A5', '#C77DFF', '#4FA6E8', '#F28B5B'];
  // 第二档副标题的公共写法：逻辑魔方 postMessage 上报的选中（见下面 message
  // 监听里的 rubik-select 分支）与宇宙视图叶子点击共用同一句拼法，唯一的
  // 差别是前者能带上 rubik-cube 自己算好的层区间（L{lo}-L{hi}），宇宙视图
  // 这条路径没有 rubik-cube 的模型可查，就不编一段假的层区间——宁可这一档
  // 副标题短一截，也不摆一个编出来的数字。
  function tier2SubLine(sel) {
    return 'tp' + sel.tp + ((PS.cp || 1) > 1 ? ' cp' + sel.cp : '') + ' pp' + sel.pp + ' rep' + sel.rep;
  }
  /* 把 count 个点铺进一段扇形楔子（原点 ox,oy · 中心角 baseAngle · 半张角
     fanHalf · 半径 [rNear,rFar]），按行铺开的网格，不是全挤在一条半径线上。
     反馈「这个宇宙视图中间的rank也要按照真实的数量来……现在的数量是远远
     不够的」——第一版把整段的叶子都摆在同一个半径上，count 一大（比如
     pangu 预置一个子组 100 颗）扇面里那点角宽度根本不够摊开，100 个点挤
     成了肉眼看着像 1 个点的一团——「数字是真的」但看不出「真的有这么多」，
     没解决反馈要的问题。这里改成二维网格：径向分成几"环"，同一环内再按
     角度摊开，两个维度一起摊，同样的角宽能摆下多得多的点、彼此还分得开。 */
  function layoutWedge(ox, oy, baseAngle, fanHalf, count, rNear, rFar) {
    var cols = Math.max(1, Math.min(count, Math.round(Math.sqrt(count * 2.4))));
    var rows = Math.ceil(count / cols);
    var pts = [];
    for (var k = 0; k < count; k++) {
      var row = Math.floor(k / cols);
      var rowStart = row * cols;
      var colsInRow = Math.min(cols, count - rowStart);
      var col = k - rowStart;
      var colT = colsInRow > 1 ? (col / (colsInRow - 1) - 0.5) * 2 : 0;
      var rowT = rows > 1 ? row / (rows - 1) : 0;
      var a = baseAngle + colT * fanHalf;
      var r = rNear + rowT * (rFar - rNear);
      pts.push({ x: ox + Math.cos(a) * r, y: oy + Math.sin(a) * r });
    }
    return pts;
  }
  function buildUniverseSvg() {
    var W = 1600, H = 1000, CX = W / 2, CY = H / 2;
    var PPN = PS.pp, TPN = PS.tp, CPN = PS.cp || 1, DPN = PS.dp;
    var R1 = 250, RSUB = 340, R2 = 460;
    // 真实数量，不抽样：外圈 hub = PP 段（不变），每段内再按 TP×CP 分出
    // 子组（groupCount，TP/CP 都是 1 时退化成没有子组，直接进内层），子组
    // 内的叶子 = 这个 (pp,tp,cp) 组合下**全部** DPN 个 rep，一个不少——
    // tp·cp·pp·dp 四个因子相乘正好等于 world，这一屏画的就是全部 world
    // 张卡，不是取景。
    var groupCount = TPN * CPN;
    // 叶子数一多，每颗都连一条到 hub/子 hub 的线只会糊成一团黑（100 条线
    // 挤在几十像素宽的楔子里，比不画还难看），加上每条 <line> 都是一个新
    // DOM 节点——数量一大直接翻倍。超过这个阈值就只画点、不画连线，靠点
    // 本身的聚簇位置读出"这是哪个子组的"，小数量（≤12，比如 moe718b128k
     // 一个子组只有 4 个 dp）继续画线，读起来更直接。
    var THREAD_MAX = 12;
    var hubsHtml = '', leavesHtml = '', linksHtml = '';
    for (var i = 0; i < PPN; i++) {
      var theta = (i / PPN) * Math.PI * 2 - Math.PI / 2;
      var hx = CX + Math.cos(theta) * R1, hy = CY + Math.sin(theta) * R1;
      var color = HUB_PALETTE[i % HUB_PALETTE.length];
      linksHtml += '<line class="u-ray" data-pp="' + i + '" x1="' + CX + '" y1="' + CY + '" x2="' + hx.toFixed(1) + '" y2="' + hy.toFixed(1) + '" stroke="' + color + '"/>';
      hubsHtml += '<g class="u-hub" data-pp="' + i + '">'
        + '<circle class="u-hubglow" cx="' + hx.toFixed(1) + '" cy="' + hy.toFixed(1) + '" r="24" fill="' + color + '"/>'
        + '<circle class="u-hubcore" cx="' + hx.toFixed(1) + '" cy="' + hy.toFixed(1) + '" r="13" fill="' + color + '"/>'
        + '<text class="u-hublabel" x="' + hx.toFixed(1) + '" y="' + (hy + 34).toFixed(1) + '" text-anchor="middle">PP' + i + '</text>'
        + '</g>';
      // 扇形半张角按"这一段跟相邻段隔多远"来定（相邻 hub 的夹角是
      // 2π/PPN），封顶在那个夹角的 42%——留出至少约 16% 的空隙，扇面才会
      // 读成"N 段各喷一束"而不是糊成一整条连续的外圈圆环。
      var fanHalf = Math.min(0.34, (Math.PI / PPN) * 0.42);
      if (groupCount <= 1) {
        // TP=CP=1：这一段没有第二层可分（比如 incident2048），DPN 个叶子
        // 铺进 hub 直接张开的那一整个楔子（径向 R1+50…R2）。
        var pts0 = layoutWedge(CX, CY, theta, fanHalf, DPN, R1 + 50, R2);
        for (var r0 = 0; r0 < DPN; r0++) {
          var p0 = pts0[r0];
          if (DPN <= THREAD_MAX) linksHtml += '<line class="u-thread" data-pp="' + i + '" x1="' + hx.toFixed(1) + '" y1="' + hy.toFixed(1) + '" x2="' + p0.x.toFixed(1) + '" y2="' + p0.y.toFixed(1) + '" stroke="' + color + '"/>';
          var sel0 = { tp: 0, cp: 0, pp: i, rep: r0 };
          leavesHtml += '<circle class="u-leaf" data-pp="' + i + '" data-tp="0" data-cp="0" data-rep="' + r0 + '"'
            + ' cx="' + p0.x.toFixed(1) + '" cy="' + p0.y.toFixed(1) + '" r="4" fill="' + color + '">'
            + '<title>' + esc(tier2SubLine(sel0)) + '</title></circle>';
        }
      } else {
        // 有 TP/CP 结构：段内再分 groupCount 个子组（子 hub），子组之间的
        // 角距跟外圈"段与段之间留缝"是同一个道理——子扇半张角封顶在"这个
        // 子组自己的角位槽宽"的 42%，组与组之间才不会糊在一起。子组本身
        // 复用 .u-hub 这个类（只是多一个 .u-subhub 标记做小尺寸样式），
        // 点击时走跟点外圈 hub 一样的"聚焦这一整个 PP 段"逻辑——子组不是
        // 唯一 rank，点了下钻没有意义，只聚焦讲得通。
        var groupSpacing = groupCount > 1 ? (2 * fanHalf) / (groupCount - 1) : 2 * fanHalf;
        var subFanHalf = Math.min(groupSpacing * 0.42, 0.15);
        for (var g = 0; g < groupCount; g++) {
          var tp9 = g % TPN, cp9 = Math.floor(g / TPN) % CPN;
          var gt = groupCount > 1 ? (g / (groupCount - 1) - 0.5) * 2 * fanHalf : 0;
          var ga = theta + gt;
          var sx = CX + Math.cos(ga) * RSUB, sy = CY + Math.sin(ga) * RSUB;
          linksHtml += '<line class="u-ray is-sub" data-pp="' + i + '" x1="' + hx.toFixed(1) + '" y1="' + hy.toFixed(1) + '" x2="' + sx.toFixed(1) + '" y2="' + sy.toFixed(1) + '" stroke="' + color + '"/>';
          hubsHtml += '<circle class="u-hub u-subhub" data-pp="' + i + '" cx="' + sx.toFixed(1) + '" cy="' + sy.toFixed(1) + '" r="4" fill="' + color + '">'
            + '<title>' + esc('pp' + i + ' tp' + tp9 + ((CPN > 1) ? ' cp' + cp9 : '')) + '</title></circle>';
          var pts = layoutWedge(CX, CY, ga, subFanHalf, DPN, RSUB + 30, R2);
          for (var r = 0; r < DPN; r++) {
            var p = pts[r];
            if (DPN <= THREAD_MAX) linksHtml += '<line class="u-thread" data-pp="' + i + '" x1="' + sx.toFixed(1) + '" y1="' + sy.toFixed(1) + '" x2="' + p.x.toFixed(1) + '" y2="' + p.y.toFixed(1) + '" stroke="' + color + '"/>';
            var sel = { tp: tp9, cp: cp9, pp: i, rep: r };
            leavesHtml += '<circle class="u-leaf" data-pp="' + i + '" data-tp="' + tp9 + '" data-cp="' + cp9 + '" data-rep="' + r + '"'
              + ' cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="3" fill="' + color + '">'
              + '<title>' + esc(tier2SubLine(sel)) + '</title></circle>';
          }
        }
      }
    }
    // 稀疏星点只做氛围，不承载数据——数量固定、每次重建（理论上只建一次，
    // 见 renderUniverse 的 universeBuilt 守卫）位置会不一样，纯装饰，不影响
    // 任何可读信息。
    var stars = '';
    for (var s = 0; s < 140; s++) {
      var sx = Math.random() * W, sy = Math.random() * H, sr = Math.random() * 1.1 + 0.2;
      stars += '<circle class="u-star" cx="' + sx.toFixed(1) + '" cy="' + sy.toFixed(1) + '" r="' + sr.toFixed(2) + '"/>';
    }
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
      + '<defs>'
      + '<radialGradient id="uCoreGrad" cx="40%" cy="35%" r="65%">'
      + '<stop offset="0%" stop-color="#FFF6DD"/><stop offset="55%" stop-color="#E8C468"/><stop offset="100%" stop-color="#8A6A1E"/>'
      + '</radialGradient>'
      + '<filter id="uGlow" x="-200%" y="-200%" width="500%" height="500%">'
      + '<feGaussianBlur stdDeviation="9" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
      + '</filter>'
      + '</defs>'
      + '<g class="u-stars">' + stars + '</g>'
      + '<g class="u-links">' + linksHtml + '</g>'
      + '<g class="u-leaves" filter="url(#uGlow)">' + leavesHtml + '</g>'
      + '<g class="u-hubs" filter="url(#uGlow)">' + hubsHtml + '</g>'
      + '<circle class="u-core" cx="' + CX + '" cy="' + CY + '" r="46" fill="url(#uCoreGrad)" filter="url(#uGlow)"/>'
      + '<text class="u-corelabel" x="' + CX + '" y="' + (CY + 74) + '" text-anchor="middle">' + esc(PS.modelName) + '</text>'
      + '<text class="u-coresub" x="' + CX + '" y="' + (CY + 96) + '" text-anchor="middle">' + world + ' 卡 · ' + PPN + ' 段流水线</text>'
      + '</svg>';
  }
  function renderUniverse() {
    if (universeBuilt) return;
    universeStage.innerHTML = buildUniverseSvg();
    universeBuilt = true;
  }
  // 点一个 hub（段本身）= 聚焦这一段、把其余段的 hub/射线/叶子调暗，不下钻
  // （一段里有好几张卡，hub 本身不对应唯一 rank）；ppIdx=null 时全部复原。
  function focusHub(ppIdx) {
    if (!universeBuilt) return;
    var sel9 = ppIdx == null ? null : String(ppIdx);
    universeStage.querySelectorAll('.u-hub, .u-leaf, .u-ray, .u-thread').forEach(function (el) {
      el.classList.toggle('is-dim', sel9 != null && el.getAttribute('data-pp') !== sel9);
    });
  }
  universeStage.addEventListener('click', function (ev) {
    var leaf = ev.target.closest('.u-leaf');
    if (leaf) {
      var sel = { tp: +leaf.getAttribute('data-tp'), cp: +leaf.getAttribute('data-cp'), pp: +leaf.getAttribute('data-pp'), rep: +leaf.getAttribute('data-rep') };
      focusHub(sel.pp);
      showTier2(rubikSelToMatrixSel(sel), tier2SubLine(sel));
      return;
    }
    var hub = ev.target.closest('.u-hub');
    if (hub) { focusHub(hub.getAttribute('data-pp')); return; }
    focusHub(null);
    showOverview();
  });
  if (universeToggle) {
    universeToggle.classList.remove('is-hidden');
    universeMode = qs.get('view') === 'universe';
    universeToggle.classList.toggle('is-on', universeMode);
    universeToggle.setAttribute('aria-pressed', String(universeMode));
    universeToggle.addEventListener('click', function () {
      universeMode = !universeMode;
      universeToggle.classList.toggle('is-on', universeMode);
      universeToggle.setAttribute('aria-pressed', String(universeMode));
      showOverview();
    });
    /* HTML 默认铺的是逻辑魔方（no is-hidden）；?view=universe 打开时要在
       第一帧就换成宇宙视图，不能等用户点一次切换钮才生效。showOverview()
       在这里调用是安全的——pendingMatrixSel/briefCard 这一刻本来就是初始
       态，不会覆盖掉任何还没发生的状态；下面 ?sel= 深链分支如果命中，会
       在此之后再调 showDetail() 把它换成第三档，两次调用顺序不冲突。 */
    if (universeMode) showOverview();
  }

  // ── 集群总览角标：一开场就借矩阵本体算一遍「多少张卡超容」，不用等读者
  //    下钻到第三档才看到真实数字 ────────────────────────────────────────
  // 容量/显存那套判定（capVerdict/memParts）全在矩阵共用的 demo.html 里，这
  // 一层不重算一遍（重算会有两套数）。矩阵本体现在只在第三档才真的打开、
  // 画满屏 SVG——但只要「算一遍聚合、报个数」，不需要真的铺开那张画。给
  // matrixFrame 先借用一次，带 ?brief=1：demo.html 收到这个参数会跳过
  // urlLoad 之后那一整套 render()，只算 ptoClusterBrief() 就地 postMessage
  // 报完，不建 SVG、不占那几秒的渲染开销。matrixFrame 这一刻仍然是
  // is-hidden（CSS 已经收着），第三档真正下钻时 showDetail() 照常把它的
  // src 换成真正的详情页——两次导航互不冲突，只是多一次不可见的加载。
  var clusterWorstRank = null;
  (function () {
    var bp = new URLSearchParams({ embed: '1', preset: PS.matrixPreset, brief: '1' });
    matrixFrame.src = '../rank-topology-3d/pattern.html?' + bp.toString();
  })();
  var CLUSTER_CAP_LABEL = { oom: '⚠ 超出容量', red: '⚠ 逼近红线', amber: '临界（黄线）' };
  function renderClusterBadge(brief) {
    if (!brief || !clusterBadge) return;
    var n = brief.n, level = n.oom > 0 ? 'oom' : n.red > 0 ? 'red' : n.amber > 0 ? 'amber' : 'ok';
    clusterWorstRank = brief.worst;
    var text = level === 'ok' ? (brief.world + ' 卡 · 全部正常')
      : CLUSTER_CAP_LABEL[level] + ' · ' + n[level] + '/' + brief.world + ' 卡';
    clusterBadge.textContent = text;
    clusterBadge.classList.toggle('is-alert', level !== 'ok');
    clusterBadge.classList.remove('is-hidden');
  }
  /* 点一下角标直接下钻到最惨那张卡（ratio 最高，聚合时顺手记下的）——不用先
     经过「随便选一张再看是不是这张最严重」。角标是全局状态，跟当前在哪一档
     无关，点开永远落在第三档，跟从档二点"↓ 单卡下钻"一致。 */
  clusterBadge && clusterBadge.addEventListener('click', function () {
    if (clusterWorstRank != null) showDetail(clusterWorstRank);
  });

  // ── 并行拓扑矩阵：只在第三档才加载，固定带 fastcard=1&solo=1 ─────────────
  // fastcard=1：矩阵共用的 demo.html 里的可选参数，默认关闭——这个简洁版传了它，
  // 详情态才会「不画坐标轴/EP 组框、无关联的卡直接不画、选中即飞焦」；不传就是
  // rank-topology-3d 自己原本的样子（标签/群组色照常画）。
  // solo=1：连兄弟卡也隐去，只看这一张卡内部——矩阵现在只在第三档才被打开，
  // 打开就直接是这一档，不必先落在"同组定位"再等一次点击才往里走。
  // 不传 mono：早先给这一档也传过 mono=1（整屏收黑白灰阶），但反馈原话
  // "选中之后的内部填充维持之前的彩色"——显存构成（权重/梯度/优化器态/
  // 专家/激活…）各自的颜色不是装饰，是在说"这一块字节是什么"，solo 视图
  // 现在只服务一张卡（不再是一整条兄弟行），色相不会跟别的卡打架，彩色
  // 反而比黑白更好读。mono 继续保留给 rank-topology-3d/net-slicing/
  // model-netgraph 这类会同屏画很多卡、需要收敛色相的场景用。
  // stitle：矩阵原生的画布名字接管这块地时，续用同一个模型名称（"用模型
  // 名称来做全部的命名和面包屑"）——不换一套说法，读者从第二档点进来，
  // 左上角那行字只是从这一层渲染的换成矩阵自己渲染的，内容不跳。
  // 分隔符用 "/" 不用 "·"："都放成面包屑用/分隔"——与逻辑魔方招牌
  // （syncBrand）、上面 world≤64 那条 stitle 统一成同一套写法。三段式
  // （模型名 / TIER2_LABEL / rank N）跟逻辑魔方那边的 tierlabel 拼法
  // 完全一致：从第二档点"下钻"换到这一屏时，左上角那行字只是从逻辑魔方
  // 渲染的换成矩阵渲染的，字面上一个字不跳。
  function matrixSrcFor(matrixSel) {
    var p = new URLSearchParams({
      embed: '1', theme: 'dark', preset: PS.matrixPreset, fastcard: '1', solo: '1',
      view: 'chain', card: '1', vtab: '3d', sel: String(matrixSel),
      stitle: PS.modelName + ' / ' + TIER2_LABEL + ' / rank ' + matrixSel
    });
    return '../rank-topology-3d/pattern.html?' + p.toString();
  }

  /* 第二档悄悄问矩阵本体要这张卡自己的显存构成——跟集群角标那次 ?brief=1
     借用是同一条"不铺满屏 SVG、只要 ptoRankBrief() 那份已经算好的摘要"的
     路，多带一个 sel=矩阵 rank。反馈「现在看不到rank中间的层和分片了还有
     显存」：cc=0/cclabels=0 把逻辑魔方画布里"卡内魔方"那份显示去掉之后
     （见 rubikParams 的注释），第二档原来只留一句"↓ 单卡下钻"邀请、不摆
     数字——那时候数字确实拿不到（矩阵没打开，编不出来），现在矩阵本体能
     在不渲染整屏的前提下就把这张卡的层区间/显存构成算完发回来（demo.html
     那边的改动同样是 opt-in，只在已有的 brief=1 分支里加一步，其他消费
     这份 demo.html 的 pattern 不传 sel 就不会触发，行为不变），没理由再让
     读者多点一次"下钻"才看到。选中就立刻发起这次请求，回信之前浮卡先
     显示邀请那版（不留空白，见 renderDrillInvite），回信到了再原地升级
     成真数据——这一步不换档，读者仍在第二档，"下钻"按钮还在，点了才真的
     飞到矩阵那一屏（solo）。 */
  function requestTier2Brief(matrixSel) {
    var bp = new URLSearchParams({ embed: '1', preset: PS.matrixPreset, brief: '1', sel: String(matrixSel) });
    matrixFrame.src = '../rank-topology-3d/pattern.html?' + bp.toString();
  }

  /* 逻辑魔方与并行拓扑矩阵各自实现了一遍"rank ↔ (tp,cp,pp,dp) 坐标"的换算，
     内部打包顺序不一样，同一个数字在两边指的不是同一张卡：
       逻辑魔方（pattern.js）  rankOf = ((rep*PP + pp)*CP + cp) * TP + tp
       并行拓扑（demo.html）   rankOf = ((pp*DP + dp)*CP + cp) * TP + tp
     只有 tp 在两边都是最内层（同一个 %TP），pp/dp(rep) 的打包顺序不同，
     所以必须按坐标三元组换算，不能把 rank 数字直接抄过去——实测验证过：
     逻辑魔方 rank 830（tp6·pp3·rep20）对应并行拓扑 rank 2566，矩阵本体
     读出的坐标正是 tp6·cp0·dp20·pp3，与逻辑魔方报的坐标逐位一致。
     cp 那一项：pangu/dense64/incident2048 都是 CP=1，sel.cp 恒为 0、
     PS.cp 缺省按 1，这一项乘完加完等于没有，公式跟改动前逐位相同；
     moe718b128k（CP=16）是第一个用上它的预置——逻辑魔方的 onSelect
     payload 补了 cp 字段（见 vendor/rubik-cube/pattern.js 的注释）才有
     这个数可用。 */
  function rubikSelToMatrixSel(sel) {
    return ((sel.pp * PS.dp + sel.rep) * (PS.cp || 1) + (sel.cp || 0)) * PS.tp + sel.tp;
  }

  var pendingMatrixSel = null;   // 第二档选中的那张卡，换算好的矩阵 rank——第三档就是拿它去开矩阵
  var pendingSubLine = null;     // 第二档那行坐标副标题——brief 回信之后原地升级要用同一句

  /* 三档的"这是什么"这句话，全部交给当前显示的那个 iframe 自己的原生标题说，
     这一层不再另起一块牌子重复一遍：第一/二档是逻辑魔方自己的顶栏招牌
     （见 rubikParams 的 brand=，已经从它自己的默认名"逻辑魔方"换成模型
     名称）与它选中后自己浮出的"RANK / rank N"身份面板；第三档是矩阵自己的
     画布名字（见 matrixSrcFor 的 stitle=）。三处名字同一个来源（PS.modelName），
     读起来是一句话，不是宿主外挂一层跟原生标题抢地、还经常撞在一起的重复牌子。 */

  /* 第一档有两种画法（逻辑魔方 iframe / 宇宙视图内联 SVG），由 universeMode
     决定当前显示哪一个——showOverview/showTier2 共用这一个开关函数，不必
     各自重复一遍"显哪个、藏哪个"。matrixFrame 两处都要藏：从第三档退回来
     时它还开着。 */
  function showTier1Visual() {
    matrixFrame.classList.add('is-hidden');
    if (universeMode) {
      renderUniverse();
      universeStage.classList.remove('is-hidden');
      rubikFrame.classList.add('is-hidden');
    } else {
      universeStage.classList.add('is-hidden');
      rubikFrame.classList.remove('is-hidden');
    }
  }

  function showOverview() {
    showTier1Visual();
    pendingMatrixSel = null;
    hideBrief();
    focusHub(null);
  }

  /* 第二档：留在第一档那个视图身上（逻辑魔方或宇宙视图，看 universeMode），
     只换宿主自己这层的 chrome——右下角浮出"下钻"邀请。选中态是那个视图
     自己的事（这一刻画面早就是对的，来路无关：可能是刚刚报上来的新选中，
     也可能是从第三档退回来、本来就还停在原地没变过），这个函数只管 sel
     （换算好的矩阵 rank，供下钻按钮用）与 subLine（下钻邀请那一行副标题，
     各来路按自己手上的坐标格式拼好再传进来，见 tier2SubLine）。 */
  function showTier2(matrixSel, subLine) {
    showTier1Visual();
    pendingMatrixSel = matrixSel;
    pendingSubLine = subLine;
    renderDrillInvite(matrixSel, subLine, null);
    requestTier2Brief(matrixSel);
  }

  /* 第三档：真正换到矩阵那一屏，solo=1 直接落在"只看这一只"。 */
  function showDetail(matrixSel) {
    matrixFrame.src = matrixSrcFor(matrixSel);
    matrixFrame.classList.remove('is-hidden');
    rubikFrame.classList.add('is-hidden');
    universeStage.classList.add('is-hidden');
    hideBrief();
  }

  // ── 接逻辑魔方自己上报的选中事件：rubik-select 是它页内换选中卡时主动发的——
  //    选中就是第二档，取消选中（点空白，它自己原有的手势）就退回第一档。 ──
  // ── 接矩阵本体上报的换档事件：pto:tier 带着 {tier, sel, brief}。矩阵现在
  //    只在第三档才被打开，收到 tier<3（矩阵里点空白退出 soloCard）就说明
  //    读者要退回第二档——切回逻辑魔方（它一直还停在原地、选中态没变过），
  //    副标题这时改用矩阵自己上报的 brief.coord/layers 拼（跟逻辑魔方自己
  //    的 tp/pp/rep 是两套坐标格式，不能混用同一个拼法）。 ──
  window.addEventListener('message', function (ev) {
    var d = ev.data;
    if (!d) return;
    if (ev.source === rubikFrame.contentWindow) {
      if (d.type === 'rubik-drill') {
        /* 再点一次已经选中的那张方块 = 下钻——逻辑魔方自己报的坐标已经够
           换算出矩阵 rank，不用等 pendingMatrixSel（用户可能从深链或退档
           回来，那个变量这一刻不一定是这张卡），直接算一遍最准。 */
        if (d.sel && d.sel.rank != null) showDetail(rubikSelToMatrixSel(d.sel));
        return;
      }
      if (d.type !== 'rubik-select') return;
      if (d.sel && d.sel.rank != null) {
        var st9 = d.sel.stage;
        /* cp 只在 PS.cp>1 时才显示——d.sel.cp===0 是合法坐标（CP>1 时也有
           第 0 段），不能拿它的真假值判断"要不要显示"，得看这份预置本身
           有没有 CP 这根轴。 */
        showTier2(rubikSelToMatrixSel(d.sel), 'tp' + d.sel.tp
          + ((PS.cp || 1) > 1 ? ' cp' + d.sel.cp : '') + ' pp' + d.sel.pp + ' rep' + d.sel.rep
          + (st9 ? ' · L' + st9.lo + '–L' + st9.hi : ''));
      } else showOverview();
      return;
    }
    if (ev.source === matrixFrame.contentWindow) {
      if (d.type === 'pto:cluster') { renderClusterBadge(d.brief); return; }
      if (d.type === 'pto:rank-brief') {
        // 这次借用可能是为了一张早就不再选中的卡（读者点得快，回信滞后）——
        // 只在还是当前这张卡时才拿去升级浮卡，旧回信直接丢弃。
        if (d.brief && d.brief.rank === pendingMatrixSel) renderDrillInvite(pendingMatrixSel, pendingSubLine, d.brief);
        return;
      }
      if (d.type !== 'pto:tier') return;
      if (d.tier === 3) {
        renderBrief(d.brief);
      } else if (d.sel != null && d.brief) {
        showTier2(d.sel, 'tp' + d.brief.coord.tp + ' cp' + d.brief.coord.cp + ' dp' + d.brief.coord.dp
          + ' pp' + d.brief.coord.pp + (d.brief.coord.ep != null ? ' ep' + d.brief.coord.ep : '')
          + ' · L' + d.brief.layers.lo + '–L' + d.brief.layers.hi);
      } else {
        showOverview();
      }
    }
  });

  function hideBrief() {
    briefCard.classList.remove('is-cta');
    briefCard.classList.add('is-hidden');
    briefCard.innerHTML = '';
  }

  /* rank 详情卡的正文（容量徽标 + 坐标/层区间 + 显存构成 + 合计）——第二档
     升级之后与第三档共用同一份拼法：两边的数字都来自矩阵本体同一个
     ptoRankBrief()（见 requestTier2Brief 与 matrixSrcFor 各自怎么问它要），
     这里只拼一次版式，不为两档各写一份、读出两套数。容量告警只用文字/
     底色深浅分挡，不引入色相，呼应"默认关掉颜色只有黑白"那条反馈。 */
  var CAP_LABEL = { oom: '⚠ 超出容量', red: '⚠ 逼近红线', amber: '临界（黄线）', ok: '正常' };
  function gbFmt(v) { return (Math.round(v * 10) / 10) + ' GB'; }
  function coordSubLine(brief) {
    return 'tp' + brief.coord.tp + ' cp' + brief.coord.cp + ' dp' + brief.coord.dp
      + ' pp' + brief.coord.pp + (brief.coord.ep != null ? ' ep' + brief.coord.ep : '')
      + ' · L' + brief.layers.lo + '–L' + brief.layers.hi;
  }
  function memBriefHtml(brief) {
    var capBadge = '<span class="brief-badge' + (brief.cap.level === 'ok' ? '' : ' is-alert') + '">'
      + (CAP_LABEL[brief.cap.level] || brief.cap.level) + '</span>';
    return '<div class="brief-h">rank ' + brief.rank + capBadge + '</div>'
      + '<div class="brief-sub">' + coordSubLine(brief) + '</div>'
      + brief.segs.map(function (s) {
        return '<div class="brief-row"><span>' + s.label + '</span><b>' + gbFmt(s.gb) + '</b></div>';
      }).join('')
      + '<div class="brief-row brief-total"><span>合计 / ' + brief.hbm + ' GB</span><b>' + gbFmt(brief.cap.totGB) + '</b></div>';
  }

  /* 第二档的浮卡：选中的瞬间先摆一句邀请（brief 还没回来，不留空白）；
     requestTier2Brief 那次借用回信之后（brief 参数非空、rank 对得上），
     原地升级成跟第三档一样详细的卡片——层区间/显存构成不再是编不出来的
     数字。"↓ 单卡下钻"按钮两种状态都留着：这一步升级的只是内容详细度，
     不是换档，点了才真的飞到矩阵那一屏（solo）。 */
  function renderDrillInvite(matrixSel, subLine, brief) {
    if (brief && brief.rank === matrixSel) {
      briefCard.innerHTML = memBriefHtml(brief)
        + '<button type="button" class="brief-cta" data-act="drill">↓ 单卡下钻 · 查看填充详情</button>';
    } else {
      briefCard.innerHTML = '<div class="brief-h">rank ' + matrixSel + '</div>'
        + '<div class="brief-sub">' + subLine + '</div>'
        + '<button type="button" class="brief-cta" data-act="drill">↓ 单卡下钻 · 查看填充详情</button>';
    }
    briefCard.classList.add('is-cta');
    briefCard.classList.remove('is-hidden');
  }
  briefCard.addEventListener('click', function (ev) {
    if (ev.target.closest('[data-act="drill"]') && pendingMatrixSel != null) showDetail(pendingMatrixSel);
  });

  /* 第三档的浮卡：不再留"下钻"按钮（已经在这一档了），正文跟第二档升级后
     共用同一个 memBriefHtml。 */
  function renderBrief(brief) {
    if (!brief) { hideBrief(); return; }
    briefCard.classList.remove('is-cta');
    briefCard.innerHTML = memBriefHtml(brief);
    briefCard.classList.remove('is-hidden');
  }

  // ── URL 深链：?sel=<并行拓扑矩阵自己的 rank 编号> 打开时直接进第三档 ─────
  var qsel = parseInt(qs.get('sel'), 10);
  if (isFinite(qsel) && qsel >= 0 && qsel < world) showDetail(qsel);
})();
