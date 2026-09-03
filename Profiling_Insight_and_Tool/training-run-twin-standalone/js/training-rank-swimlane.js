/*
 * 训练步泳道（按 Rank） —— training-monitoring-v2.html 底部 Timeline 面板的泳道图
 * ════════════════════════════════════════════════════════════════════════════
 * 为什么另起一个文件而不是改 js/timeline-swimlane.js：
 *   原来那张图是「算子/通信事务级」的 1F1B trace（每个 rank 一条泳道，条上密密麻麻
 *   是 AR / A2A / PP / DP 这些 kernel 与集合通信），站在**训练**的角度太细——看图的人
 *   想先知道「这一步里各 rank 的前向、反向、梯度同步、参数更新分别发生在什么时候、
 *   谁把谁堵住了」，而不是先看到几百根 3px 宽的柱子。原文件仍被「问题一 · 通信调度层」
 *   的定位链引用（renderProblemOneTimeline，只画 rank 23 一条），保持不动。
 *
 * 架构照搬 hpc-topology-viewer 的 combo-workbench/swimlane.html（微批次生命周期泳道）：
 *   · 行 = 一棵可展开的树：训练步 → PP stage → Rank → Rank 内三条流（计算/通信/更新）；
 *     行模型就是 { id, label, events, depth, parentRow, rowType, meta, detailEvents }，
 *     可见行 = 父链全展开的行（visibleRows）。
 *   · 画布 = 单张 canvas，左侧 gutter 画行头（sticky，随横向滚动固定），右侧画事件条；
 *     命中区统一压进 hitRects，反向遍历取最上层（后压入的先命中）。
 *   · 事件条复用 window.PtoSwimlaneTaskPattern.drawTaskBar / createTooltip / showTooltip，
 *     前向条叠向右的 chevron 底纹、反向条叠向左的，方向一眼可读。
 *   · 离散缩放 1×…64×，横向滚动；行头 tooltip 说明该行在层级里的位置。
 * 相对上游的取舍：**不下钻到 kernel 级**（上游那块 attn/router/experts 色带整段没有搬），
 * 最细停在「Rank 内的计算流 / 通信流 / 更新流」三轨。
 *
 * ── 内容按本页的故事重排 ─────────────────────────────────────────────────────
 * 并行配置与 v2 其余部分一致：openPangu-2.0-Flash · 32 Rank · DP2 × PP4 × TP2 × EP2 ·
 * 1F1B · 8 micro-batch。Rank 编号取 Megatron 默认序（tp 最内、pp 最外）：
 *     rank = stage×8 + dp×4 + tp×2 + ep      →  PP0:0–7  PP1:8–15  PP2:16–23  PP3:24–31
 * 层切分故意不均匀（首尾 stage 要带 Embedding / LM Head，层数少一些）：
 *     PP0 = Embedding + L0–L9 ／ PP1 = L10–L23 ／ PP2 = L24–L39 ／ PP3 = L40–L45 + LM Head
 * 这样 **L38 的 MoE 落在 PP2，而 PP2 正好是 rank 16–23（node2 那 8 张卡）** —— 与定位链里
 * 「NCCL trace: node2 ranks 16-23, rank 23 all-to-all timeout」「EP rank 23 / PP stage 3」
 * 「rank 17 OOM」三条既有文案对上，不用再各说各的。
 *
 * 两个场景（顶部段控切换，就是「构造问题」的那一下）：
 *   · 健康步 15202：F/B 交错的 1F1B 梯形 → DP 梯度 AllReduce → Optimizer Step，闭环完整；
 *   · 事故步 15203：rank 23 在 B m3 的 L38 expert dispatch 处发起 EP All-to-All 后再没回来
 *     （send=0 / recv≈9832，buffer 失配死锁）。图上按「谁先被堵住」逐圈扩散：
 *       EP 组同伴 rank 22 立刻空等 → PP2 其余 rank 做完手上的 micro-batch 后空等 EP 栅栏 →
 *       PP1/PP0 收不到 PP2 回传的梯度 → PP3 的 send 队列积压 → 全局 DP AllReduce 与
 *       Optimizer Step 一直没能开始（图上以虚线轮廓画在「本该发生」的位置）。
 * 时间是确定性模拟（同一份输入永远画出同一张图），不是真实 profiling 采样。
 *
 * ── 配色与「定位聚焦」──────────────────────────────────────────────────────
 * 配色按色相环铺开（见下方 COLORS 注释），**红色只留给错误**：反向条从上游那支
 * #ff4b7b 粉红改成紫，事故步的 NaN loss 反过来归到 fault 上红。
 * 聚光灯（js/training-spotlight.js 问题二 · 步④「通信调度层」）照到本图时会调
 * focusFault()：切到事故步 + 事故窗口、把 rank 23 那行滚到视野中间、整图去色，
 * 只留那一条红色的 all-to-all 死锁条 —— 光洞把视线引到面板，面板再把视线引到那一条。
 *
 * 对外暴露 window.PtoTrainingRankSwimlane.{render, focusFault, clearFocus}。
 */
(function registerTrainingRankSwimlane(global) {
  'use strict';

  /* ══ 1. 世界配置与故事常量 ══════════════════════════════════════════════ */

  var WORLD = {
    worldSize: 32, dp: 2, pp: 4, tp: 2, ep: 2, micro: 8,
    schedule: '1F1B', model: 'openPangu-2.0-Flash'
  };

  var STAGES = [
    { stage: 0, scope: 'Embedding + L0–L9', layers: 'L0–L9', weight: 0.92, note: '含 token embedding 查表' },
    { stage: 1, scope: 'L10–L23', layers: 'L10–L23', weight: 1.00, note: 'dense + MoE 混合层段' },
    { stage: 2, scope: 'L24–L39', layers: 'L24–L39', weight: 1.14, note: 'MoE 密集层段，L38 router 在此' },
    { stage: 3, scope: 'L40–L45 + LM Head', layers: 'L40–L45', weight: 0.98, note: '含 final norm / LM Head / Loss' }
  ];

  // 事故三要素：哪张卡、第几个 micro-batch 的反向、卡在哪个算子上。
  var FAULT = {
    rank: 23, stage: 2, micro: 3, layer: 'L38',
    op: 'EP All-to-All · expert dispatch',
    reason: 'router softmax FP8 溢出 → 98% token 塌缩到 expert 193 → send=0 / recv≈9832 buffer 失配',
    watchdog: 'HCCL watchdog 30 s'
  };
  var OOM_RANK = 17;                 // 问题一（显存 OOM）的那张卡，同在 PP2
  var HEALTHY_STEP = 15202;
  var INCIDENT_STEP = 15203;

  // 基准耗时（ms）。整步 ≈ 470 ms，与 v2 时光机里的 step 耗时同量级。
  var F_BASE = 11.4, B_BASE = 22.1, PP_COMM = 2.2;
  var ALLREDUCE_MS = 43.6, OPTIMIZER_MS = 18.4;

  /* 配色：把「红」腾出来只给错误。
     原来是 前向蓝 + 反向 #ff4b7b 粉红 + 琥珀 + 橙红 —— 暖端挤了一堆，一屏下来到处像在
     报警，真正的故障条反而不跳。现在按色相环把各类事件铺开，每两类之间至少隔一个可辨
     的色相段：红 0° → 琥珀 38° → 黄绿 77° → 翠绿 160° → 青 187° → 天蓝 199° → 蓝 228°
     → 紫 271° → 玫红 330°。取值都来自设计系统已有的语义色与 combo-workbench 那张图的
     ARCH_NODE_COLORS，不新造色板。**红色 #DC2626 全图只有 fault 一种用途。**  */
  var COLORS = {
    step: '#38BDF8',      // 天蓝 · 训练步总包络（只出现在最顶一行）
    forward: '#4369EF',   // 蓝   · 前向
    backward: '#A855F7',  // 紫   · 反向（原 #ff4b7b 粉红：反向是正常流程，不该长得像报警）
    comm: '#04D793',      // 翠绿 · PP / TP / EP 通信
    reduce: '#F59E0B',    // 琥珀 · DP 梯度 AllReduce
    update: '#87C80F',    // 黄绿 · Optimizer Step 参数更新
    hold: '#22D3EE',      // 青   · 激活驻留（半透明带 + 描边，形状上已与实心条不同）
    loss: '#EC4899',      // 玫红 · Loss（事故步的 NaN loss 归到 fault，见 buildRows）
    wait: '#6B7280',      // 灰   · 空等 / 流水线气泡
    fault: '#DC2626'      // 红   · 全图唯一的红，只给真正的错误
  };
  // 「定位聚焦」（聚光灯照到本图时）下，非故障条一律压成这一色 —— 让红色独占注意力
  var DIM_COLOR = '#5A6472';
  var DIM_ALPHA = 0.34;

  var LAYOUT = {
    rowHeight: 24, eventHeight: 17, expandedRankHeight: 64,
    trackPitch: 19, trackTop: 3, gutter: 252, headerH: 46
  };
  var ZOOM_STEPS = [1, 2, 4, 8, 16, 32, 64];
  var TRACK_LABELS = ['计算流', '通信流', '更新流'];

  /* ══ 2. 小工具 ═════════════════════════════════════════════════════════ */

  function rankOf(stage, dp, tp, ep) { return stage * 8 + dp * 4 + tp * 2 + ep; }
  function round2(v) { return Math.round(v * 100) / 100; }
  function fmtMs(v) { return Number(v).toFixed(2) + ' ms'; }
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  // 确定性抖动：同样的 (a,b,c) 永远得到同一个 0..1，保证图每次都一模一样。
  function hash01(a, b, c) {
    var h = ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) >>> 0;
    h ^= h >>> 13;
    h = Math.imul(h, 1274126177) >>> 0;
    return h / 4294967295;
  }
  function rgba(hex, alpha) {
    var clean = String(hex || '').replace('#', '');
    var full = clean.length === 3 ? clean.split('').map(function (c) { return c + c; }).join('') : clean;
    var v = parseInt(full, 16);
    if (!isFinite(v)) return 'rgba(255,255,255,' + alpha + ')';
    return 'rgba(' + ((v >> 16) & 255) + ',' + ((v >> 8) & 255) + ',' + (v & 255) + ',' + alpha + ')';
  }

  function event(id, label, kind, startMs, endMs, metrics) {
    return {
      id: id, label: label, kind: kind,
      startMs: round2(startMs), endMs: round2(endMs),
      durationMs: round2(endMs - startMs),
      metrics: metrics || {}
    };
  }

  function row(id, label, events, options) {
    var o = options || {};
    return {
      id: id, label: label, events: events || [],
      depth: o.depth || 0,
      parentRow: o.parentRow || null,
      rowType: o.rowType || 'row',
      meta: o.meta || null,
      detailEvents: o.detailEvents || null
    };
  }

  /* ══ 3. 1F1B 排布模拟 ══════════════════════════════════════════════════ */

  // 每个 stage 一条指令流：warmup 阶段连发 F，稳态 F/B 交替，drain 阶段连发 B。
  // 依赖：F 要等上游 stage 同 micro 的 F 传过来；B 要等下游 stage 同 micro 的 B 回传。
  function simulate1F1B() {
    var pp = WORLD.pp, micro = WORLD.micro;
    var compF = [], compB = [], seq = [], idx = [], stageFree = [];
    var s, k;
    for (s = 0; s < pp; s += 1) {
      compF.push(new Array(micro).fill(null));
      compB.push(new Array(micro).fill(null));
      stageFree.push(0);
      idx.push(0);
      var warmup = pp - 1 - s;
      var steady = micro - warmup;
      var ops = [], f = 0, b = 0;
      for (k = 0; k < warmup; k += 1) ops.push({ type: 'F', m: f++ });
      for (k = 0; k < steady; k += 1) { ops.push({ type: 'F', m: f++ }); ops.push({ type: 'B', m: b++ }); }
      while (b < micro) ops.push({ type: 'B', m: b++ });
      seq.push(ops);
    }

    var scheduled = 0, total = pp * micro * 2, guard = 0;
    while (scheduled < total && guard++ < 20000) {
      var progressed = false;
      for (s = 0; s < pp; s += 1) {
        if (idx[s] >= seq[s].length) continue;
        var op = seq[s][idx[s]];
        var dep = 0;
        if (op.type === 'F') {
          if (s > 0) {
            if (!compF[s - 1][op.m]) continue;
            dep = compF[s - 1][op.m].end + PP_COMM;
          }
        } else if (s < pp - 1) {
          if (!compB[s + 1][op.m]) continue;
          dep = compB[s + 1][op.m].end + PP_COMM;
        } else {
          if (!compF[s][op.m]) continue;
          dep = compF[s][op.m].end + 1.5;   // 最后一个 stage：F 完 → Loss → 立刻 B
        }
        var base = op.type === 'F' ? F_BASE : B_BASE;
        var dur = base * STAGES[s].weight * (0.88 + hash01(s, op.type === 'F' ? 1 : 2, op.m) * 0.24);
        var start = Math.max(stageFree[s], dep);
        var rec = { start: start, end: start + dur, type: op.type, m: op.m, stage: s };
        (op.type === 'F' ? compF : compB)[s][op.m] = rec;
        stageFree[s] = rec.end;
        idx[s] += 1;
        scheduled += 1;
        progressed = true;
      }
      if (!progressed) break;
    }

    var pipelineEnd = 0;
    for (s = 0; s < pp; s += 1) {
      for (k = 0; k < micro; k += 1) {
        pipelineEnd = Math.max(pipelineEnd, (compF[s][k] || {}).end || 0, (compB[s][k] || {}).end || 0);
      }
    }
    return { compF: compF, compB: compB, pipelineEnd: pipelineEnd };
  }

  var SCHED = simulate1F1B();
  var GRAD_SYNC_START = round2(SCHED.pipelineEnd + 1.8);
  var GRAD_SYNC_END = round2(GRAD_SYNC_START + ALLREDUCE_MS);
  var OPT_START = round2(GRAD_SYNC_END + 0.6);
  var OPT_END = round2(OPT_START + OPTIMIZER_MS);
  var STEP_END = OPT_END;

  // 每张卡的个性：TP/EP 分片让它比同组同伴早一点或晚一点开始，收尾还要等最慢的那片。
  function rankProfile(stage, dp, tp, ep) {
    var rank = rankOf(stage, dp, tp, ep);
    var startOffset = round2(hash01(rank, 11, 0) * 1.6);
    var syncWait = round2(hash01(rank, 17, 0) * 2.1);
    return {
      rank: rank, stage: stage, dp: dp, tp: tp, ep: ep,
      startOffset: startOffset, syncWait: syncWait,
      role: 'DP' + dp + ' · TP' + tp + ' / EP' + ep,
      hiddenShard: 'hidden[' + (tp * 1280) + ':' + (tp * 1280 + 1280) + ']',
      localExperts: 'expert[' + (ep * 128) + '–' + (ep * 128 + 127) + ']',
      expertLoad: (58 + Math.round(hash01(rank, 23, 0) * 34)) + '%'
    };
  }

  /* ══ 4. 事件构造 ═══════════════════════════════════════════════════════ */

  // 事故场景下每张卡「做到哪一步就再也推不动了」。返回 null = 不受影响（健康场景）。
  // 分四圈，越靠近 rank 23 停得越早：
  //   ① rank 23 本人  ② 同 EP 组同伴  ③ PP2 其余 rank  ④ 其它 stage
  function faultCutoff(profile) {
    var faultB = SCHED.compB[FAULT.stage][FAULT.micro];
    // 死锁的引信：B m3 跑到 28% 处发起 L38 的 expert dispatch all-to-all，之后再没回来。
    var a2aStart = round2(faultB.start + (faultB.end - faultB.start) * 0.28);
    if (profile.rank === FAULT.rank) {
      return { at: a2aStart, ring: 'origin' };
    }
    if (profile.stage === FAULT.stage && profile.dp === 1 && profile.tp === 1) {
      // 同一个 EP 组（只差 ep 分片）：all-to-all 是组内集合操作，同一刻一起挂住。
      return { at: a2aStart, ring: 'ep-peer' };
    }
    if (profile.stage === FAULT.stage) {
      // PP2 其余 rank：手上的 B m3 还能做完，再往下就撞上 EP 组栅栏。
      return { at: round2(SCHED.compB[FAULT.stage][FAULT.micro].end), ring: 'stage-peer' };
    }
    if (profile.stage < FAULT.stage) {
      // PP0 / PP1：PP2 不再回传梯度，B m4 起就没有输入了。
      return { at: round2(SCHED.compB[profile.stage][FAULT.micro].end), ring: 'upstream' };
    }
    // PP3：还能往前跑两个 micro-batch，直到 send 队列被下游堵满。
    var lastMicro = Math.min(WORLD.micro - 1, FAULT.micro + 2);
    return { at: round2(SCHED.compB[profile.stage][lastMicro].end), ring: 'downstream' };
  }

  var RING_TEXT = {
    'ep-peer': ['空等 · EP 组栅栏', '与 rank ' + FAULT.rank + ' 同属一个 EP all-to-all 组，集合操作未返回，本卡在同一时刻挂住'],
    'stage-peer': ['空等 · 等待 EP 组', '手上的 micro-batch 做完后撞上 PP2 的 EP 组栅栏，无法进入下一个 micro-batch'],
    'upstream': ['空等 · 等待 PP2 梯度', 'PP2 不再回传激活梯度，本 stage 的后续反向没有输入'],
    'downstream': ['空等 · PP send 阻塞', '下游 PP2 不再取走梯度，send 队列积压后本 stage 也停住']
  };

  // 一张卡在一个 step 里的全部事件。detail=true 的只在该 rank 行展开后出现，
  // 折叠态只看「关键前向 / 反向 / 更新 / 故障」，这正是本图和旧图的分水岭。
  function buildRankEvents(profile, incident) {
    var stage = profile.stage, rank = profile.rank;
    var id = 'r' + rank;
    var events = [];
    var cutoff = incident ? faultCutoff(profile) : null;
    var windowEnd = incident ? INCIDENT_WINDOW_END : STEP_END;
    var m;

    function alive(t) { return !cutoff || t < cutoff.at - 0.01; }

    for (m = 0; m < WORLD.micro; m += 1) {
      var f = SCHED.compF[stage][m];
      var b = SCHED.compB[stage][m];
      if (f && alive(f.start)) {
        var fs = round2(f.start + profile.startOffset);
        var fe = round2(Math.min(f.end, cutoff ? cutoff.at : f.end));
        events.push(event(id + '-f' + m, 'F m' + m, 'forward', fs, Math.max(fs + 0.4, fe), {
          track: 0, phase: 'Forward', microbatch: 'm' + m, rankId: 'Rank ' + rank,
          stage: 'PP' + stage, layers: STAGES[stage].layers, tpShard: 'TP' + profile.tp, epShard: 'EP' + profile.ep,
          localTensorShard: profile.hiddenShard, localExperts: profile.localExperts, expertLoad: profile.expertLoad
        }));
        // 前向里的 TP AllGather（展开后才画）
        if (alive(f.start + (f.end - f.start) * 0.18)) {
          events.push(event(id + '-f' + m + '-tp', 'TP AG', 'comm',
            f.start + (f.end - f.start) * 0.18, f.start + (f.end - f.start) * 0.30, {
              track: 1, detail: true, collective: 'TP AllGather', group: 'TP' + WORLD.tp + ' · rank ' + rankOf(stage, profile.dp, 0, profile.ep) + '/' + rankOf(stage, profile.dp, 1, profile.ep),
              payload: '20 MiB', microbatch: 'm' + m
            }));
        }
      }
      if (b && alive(b.start)) {
        var bs = round2(b.start + profile.startOffset);
        var be = round2(Math.min(b.end, cutoff ? cutoff.at : b.end));
        events.push(event(id + '-b' + m, 'B m' + m, 'backward', bs, Math.max(bs + 0.4, be), {
          track: 0, phase: 'Backward', microbatch: 'm' + m, rankId: 'Rank ' + rank,
          stage: 'PP' + stage, layers: STAGES[stage].layers, tpShard: 'TP' + profile.tp, epShard: 'EP' + profile.ep,
          weightGrad: fmtMs((b.end - b.start) * 0.46), inputGrad: fmtMs((b.end - b.start) * 0.38)
        }));
        // 反向里的 EP All-to-All combine（展开后才画）—— 故事的引信就在这里
        var a2aStart = b.start + (b.end - b.start) * 0.28;
        if (stage === FAULT.stage && alive(a2aStart)) {
          events.push(event(id + '-b' + m + '-a2a', 'EP A2A', 'comm', a2aStart, a2aStart + 3.1, {
            track: 1, detail: true, collective: 'EP All-to-All', layer: FAULT.layer,
            group: 'EP' + WORLD.ep + ' · rank ' + rankOf(stage, profile.dp, profile.tp, 0) + '/' + rankOf(stage, profile.dp, profile.tp, 1),
            payload: '9832 token', microbatch: 'm' + m
          }));
        }
      }
      // PP 交接（展开后才画）
      if (stage < WORLD.pp - 1 && SCHED.compF[stage][m] && alive(SCHED.compF[stage][m].end)) {
        events.push(event(id + '-pps' + m, 'PP→', 'comm', SCHED.compF[stage][m].end, SCHED.compF[stage][m].end + PP_COMM, {
          track: 1, detail: true, transactionId: 'P2P-S' + INCIDENT_STEP + '-PP' + stage + '-PP' + (stage + 1) + '-F-m' + m,
          direction: '发送激活 → PP' + (stage + 1), payload: '40 MiB', microbatch: 'm' + m
        }));
      }
      if (stage > 0 && SCHED.compB[stage][m] && alive(SCHED.compB[stage][m].end)) {
        events.push(event(id + '-ppg' + m, '←PP', 'comm', SCHED.compB[stage][m].end, SCHED.compB[stage][m].end + PP_COMM, {
          track: 1, detail: true, transactionId: 'P2P-S' + INCIDENT_STEP + '-PP' + stage + '-PP' + (stage - 1) + '-B-m' + m,
          direction: '发送梯度 → PP' + (stage - 1), payload: '40 MiB', microbatch: 'm' + m
        }));
      }
    }

    if (!incident) {
      // 健康步的收尾：梯度 AllReduce（DP 组内）→ Optimizer Step。这就是「更新」。
      var arStart = round2(GRAD_SYNC_START + profile.syncWait);
      events.push(event(id + '-allreduce', '梯度 AllReduce', 'reduce', arStart, GRAD_SYNC_END, {
        track: 2, collective: 'DP AllReduce', rankId: 'Rank ' + rank,
        group: 'DP' + WORLD.dp + ' · rank ' + rankOf(stage, 0, profile.tp, profile.ep) + '/' + rankOf(stage, 1, profile.tp, profile.ep),
        payload: '1.86 GiB (bucket ×24)', bandwidth: '43.7 GB/s',
        syncWait: fmtMs(profile.syncWait), gradNorm: '1.2' + (rank % 10)
      }));
      events.push(event(id + '-optim', 'Optimizer Step', 'update', OPT_START, OPT_END, {
        track: 2, optimizer: 'AdamW · β=(0.9,0.95)', rankId: 'Rank ' + rank,
        gradClip: 'global_norm ≤ 1.0', lr: '1.42e-4', updated: rank === OOM_RANK ? '本卡显存峰值 61.8 / 64 GB' : '分片参数 1.86 GiB'
      }));
      if (profile.syncWait > 0.3) {
        events.push(event(id + '-arwait', '等待 DP 栅栏', 'wait', round2(GRAD_SYNC_START), arStart, {
          track: 2, detail: true, reason: 'TP/EP 分片完成时间差', syncPoint: 'DP AllReduce 入口'
        }));
      }
    } else if (cutoff) {
      if (cutoff.ring === 'origin') {
        events.push(event(id + '-fault', 'EP A2A TIMEOUT · ' + FAULT.layer, 'fault', cutoff.at, windowEnd, {
          track: 1, rankId: 'Rank ' + rank, layer: FAULT.layer, collective: FAULT.op,
          sendBuffer: '0 token', recvBuffer: '≈9832 token', status: '死锁 · ' + FAULT.watchdog + ' 后中止',
          reason: FAULT.reason, microbatch: 'm' + FAULT.micro
        }));
      } else {
        var text = RING_TEXT[cutoff.ring];
        events.push(event(id + '-wait', text[0], 'wait', cutoff.at, windowEnd, {
          track: 0, rankId: 'Rank ' + rank, reason: text[1],
          blockedBy: 'Rank ' + FAULT.rank + ' · ' + FAULT.op, status: '本步未完成'
        }));
      }
      // 「本该发生」的更新窗口：画成空心轮廓，说明它不是延后，而是压根没开始。
      events.push(event(id + '-allreduce-miss', '梯度 AllReduce（未发生）', 'ghost', GRAD_SYNC_START, GRAD_SYNC_END, {
        track: 2, detail: true, status: '未进入', reason: '集合通信组内有 rank 未到达，AllReduce 无法开始'
      }));
      events.push(event(id + '-optim-miss', 'Optimizer Step（未发生）', 'ghost', OPT_START, OPT_END, {
        track: 2, detail: true, status: '未进入', reason: '梯度未同步，本步参数未更新'
      }));
    }
    return events;
  }

  // 事故场景的时间窗：pipeline 正常收尾之后再留一段，让死锁那条读得出「一直没回来」。
  var INCIDENT_WINDOW_END = round2(STEP_END + 210);

  /* ══ 5. 行树 ═══════════════════════════════════════════════════════════ */

  function buildRows(incident) {
    var rows = [];
    var stepLabel = incident ? ('Step ' + INCIDENT_STEP + ' · 未完成') : ('Step ' + HEALTHY_STEP + ' · ' + fmtMs(STEP_END));
    var windowEnd = incident ? INCIDENT_WINDOW_END : STEP_END;

    rows.push(row('step', '训练步 · ' + WORLD.model, [
      event('step-total', stepLabel, incident ? 'fault' : 'step', 0, windowEnd, {
        step: String(incident ? INCIDENT_STEP : HEALTHY_STEP), schedule: WORLD.schedule,
        topology: 'DP' + WORLD.dp + ' × PP' + WORLD.pp + ' × TP' + WORLD.tp + ' × EP' + WORLD.ep,
        microbatches: WORLD.micro + ' micro-batch', worldSize: WORLD.worldSize + ' Rank',
        status: incident ? ('rank ' + FAULT.rank + ' EP all-to-all 死锁，参数未更新') : '正常闭环：前向 → 反向 → 梯度同步 → 参数更新'
      })
    ], { depth: 0, rowType: 'step' }));

    STAGES.forEach(function (info) {
      var s = info.stage;
      var firstRank = rankOf(s, 0, 0, 0), lastRank = rankOf(s, 1, 1, 1);
      var stageId = 'stage-' + s;
      var fwdStart = SCHED.compF[s][0].start;
      var fwdEnd = SCHED.compF[s][WORLD.micro - 1].end;
      var bwdStart = SCHED.compB[s][0].start;
      var bwdEnd = SCHED.compB[s][WORLD.micro - 1].end;
      var cut = incident ? faultCutoff(rankProfile(s, 0, 0, 0)) : null;
      var stageBwdEnd = cut ? Math.min(bwdEnd, cut.at) : bwdEnd;

      var stageEvents = [
        event(stageId + '-fwd', 'PP' + s + ' 前向 · ' + WORLD.micro + ' micro-batch', 'forward', fwdStart, fwdEnd, {
          stage: 'PP' + s, layers: info.layers, scope: info.scope, note: info.note,
          rankGroup: 'Rank ' + firstRank + '–' + lastRank, envelope: '最早 rank 启动 → 最晚 rank 完成'
        }),
        event(stageId + '-bwd', 'PP' + s + ' 反向', 'backward', bwdStart, Math.max(bwdStart + 0.5, stageBwdEnd), {
          stage: 'PP' + s, layers: info.layers, rankGroup: 'Rank ' + firstRank + '–' + lastRank,
          status: cut ? ('在 m' + FAULT.micro + ' 附近被截断') : '完成 ' + WORLD.micro + ' 个 micro-batch'
        })
      ];
      if (s === WORLD.pp - 1) {
        // 事故步的 loss 是 NaN —— 它本身就是错误现场之一，所以走 fault（红）；健康步是玫红。
        stageEvents.push(event(stageId + '-loss', incident ? 'Loss = NaN' : 'Loss', incident ? 'fault' : 'loss',
          SCHED.compF[s][0].end, SCHED.compF[s][0].end + 1.5, {
            stage: 'PP' + s, loss: incident ? 'NaN' : '2.184',
            note: incident ? 'router softmax FP8 溢出沿前向传播到 loss' : '交叉熵 + z-loss'
          }));
      }
      if (incident) {
        stageEvents.push(event(stageId + '-wait', s === FAULT.stage ? ('PP' + s + ' 被 rank ' + FAULT.rank + ' 阻塞') : ('PP' + s + ' 空等'),
          s === FAULT.stage ? 'fault' : 'wait', cut.at, windowEnd, {
            stage: 'PP' + s, blockedBy: 'Rank ' + FAULT.rank + ' · ' + FAULT.op,
            reason: RING_TEXT[cut.ring] ? RING_TEXT[cut.ring][1] : FAULT.reason, status: '本步未完成'
          }));
      } else {
        stageEvents.push(event(stageId + '-ar', 'PP' + s + ' 梯度 AllReduce', 'reduce', GRAD_SYNC_START, GRAD_SYNC_END, {
          stage: 'PP' + s, collective: 'DP AllReduce', payload: '1.86 GiB × 8 rank', bandwidth: '43.7 GB/s'
        }));
        stageEvents.push(event(stageId + '-opt', 'Optimizer Step', 'update', OPT_START, OPT_END, {
          stage: 'PP' + s, optimizer: 'AdamW', lr: '1.42e-4', gradClip: 'global_norm ≤ 1.0'
        }));
      }

      rows.push(row(stageId, 'PP' + s + ' · ' + info.scope, stageEvents, {
        depth: 1, parentRow: 'step', rowType: 'stage',
        meta: { stage: s, scope: info.scope, layers: info.layers, firstRank: firstRank, lastRank: lastRank }
      }));

      // 该 stage 的 8 张卡
      for (var dp = 0; dp < WORLD.dp; dp += 1) {
        for (var tp = 0; tp < WORLD.tp; tp += 1) {
          for (var ep = 0; ep < WORLD.ep; ep += 1) {
            var profile = rankProfile(s, dp, tp, ep);
            var events = buildRankEvents(profile, incident);
            var suffix = profile.rank === FAULT.rank && incident ? ' · 死锁'
              : profile.rank === OOM_RANK ? ' · 显存 61.8 GB' : '';
            rows.push(row('rank-' + profile.rank, 'Rank ' + profile.rank + ' · ' + profile.role + suffix, events, {
              depth: 2, parentRow: stageId, rowType: 'rank',
              meta: profile, detailEvents: events
            }));
          }
        }
      }

      // 激活驻留：前向存下、反向吃掉，PP2 这一条最长也最胖（问题一 OOM 的舞台）
      var holdEnd = incident && s === FAULT.stage ? windowEnd : bwdEnd;
      var holdGiB = s === FAULT.stage ? 12.4 : s === 0 ? 9.1 : s === 1 ? 10.6 : 8.3;
      rows.push(row('hold-' + s, 'PP' + s + ' · 激活驻留', [
        event('hold-' + s + '-band', '保存激活 ' + holdGiB.toFixed(1) + ' GiB', 'hold', fwdStart, holdEnd, {
          stage: 'PP' + s, memory: holdGiB.toFixed(1) + ' GiB', producer: 'PP' + s + ' 前向', consumer: 'PP' + s + ' 反向',
          checkpoint: '未开启 activation checkpoint',
          peakRank: s === FAULT.stage ? ('Rank ' + OOM_RANK + ' 峰值 61.8 / 64 GB') : '—',
          status: incident && s === FAULT.stage ? '反向未跑完 → 激活迟迟不释放' : '反向消费后释放'
        })
      ], { depth: 2, parentRow: stageId, rowType: 'hold', meta: { stage: s } }));
    });

    return rows;
  }

  /* ══ 6. 渲染 ═══════════════════════════════════════════════════════════ */

  var SHELL_HTML = ''
    + '<div class="trs-root">'
    + '  <header class="trs-toolbar">'
    + '    <span class="trs-title">训练步泳道 · 按 Rank</span>'
    + '    <span class="trs-meta" data-trs-meta></span>'
    + '    <button type="button" class="trs-focus-chip" data-trs-focus hidden'
    + '            title="退出定位聚焦，恢复全部泳道的配色">定位聚焦中 · 仅高亮 rank ' + FAULT.rank + ' ✕</button>'
    + '    <span class="trs-spacer"></span>'
    + '    <span class="trs-group" role="group" aria-label="场景">'
    + '      <button type="button" class="trs-seg" data-trs-scene="healthy">健康步 ' + HEALTHY_STEP + '</button>'
    + '      <button type="button" class="trs-seg is-on" data-trs-scene="incident">事故步 ' + INCIDENT_STEP + '</button>'
    + '    </span>'
    + '    <span class="trs-group" role="group" aria-label="时间范围" data-trs-ranges></span>'
    + '    <span class="trs-zoom">'
    + '      <button type="button" class="trs-icon" data-trs-zoom="out" aria-label="缩小">−</button>'
    + '      <input type="range" class="trs-slider" min="0" max="6" step="1" value="0" aria-label="时间轴缩放">'
    + '      <button type="button" class="trs-icon" data-trs-zoom="in" aria-label="放大">+</button>'
    + '      <output class="trs-zoom-readout">1×</output>'
    + '    </span>'
    + '  </header>'
    + '  <div class="trs-scroll" data-trs-scroll><div class="trs-surface" data-trs-surface><canvas data-trs-canvas></canvas></div></div>'
    + '  <footer class="trs-footer">'
    + '    <div class="trs-legend">'
    + '      <span title="micro-batch 前向；条内向右的箭头底纹表示前向流向"><i class="trs-flow trs-flow-f" style="--trs-c:' + COLORS.forward + '"></i>前向</span>'
    + '      <span title="micro-batch 反向；条内向左的箭头底纹表示反向流向"><i class="trs-flow trs-flow-b" style="--trs-c:' + COLORS.backward + '"></i>反向</span>'
    + '      <span title="DP 组内梯度 AllReduce：把 8 张卡的梯度对齐成同一份"><i style="--trs-c:' + COLORS.reduce + '"></i>梯度同步</span>'
    + '      <span title="AdamW 参数更新：一个训练步真正「学到东西」的那一下"><i style="--trs-c:' + COLORS.update + '"></i>参数更新</span>'
    + '      <span title="PP 激活/梯度交接、TP AllGather、EP All-to-All（展开 Rank 行后出现在通信流轨）"><i style="--trs-c:' + COLORS.comm + '"></i>通信</span>'
    + '      <span title="流水线气泡与集合通信栅栏上的空等"><i style="--trs-c:' + COLORS.wait + '"></i>空等</span>'
    + '      <span title="半透明青带 = 前向存下的激活一直驻留到反向消费才释放"><i class="trs-hold" style="--trs-c:' + COLORS.hold + '"></i>激活驻留</span>'
    + '      <span title="事故：rank 23 的 EP all-to-all 未返回"><i style="--trs-c:' + COLORS.fault + '"></i>故障阻塞</span>'
    + '    </div>'
    + '    <span class="trs-status" data-trs-status></span>'
    + '  </footer>'
    + '</div>';

  var lastInstance = null;   // 最近一次渲染出来的实例，供 focusFault / clearFocus 找上门

  function render(host, options) {
    if (!host) return null;
    if (host.__trsInstance) { lastInstance = host.__trsInstance; host.__trsInstance.draw(); return host.__trsInstance; }

    var opts = options || {};
    host.innerHTML = SHELL_HTML;

    var root = host.querySelector('.trs-root');
    var canvas = host.querySelector('[data-trs-canvas]');
    var scroll = host.querySelector('[data-trs-scroll]');
    var surface = host.querySelector('[data-trs-surface]');
    var metaEl = host.querySelector('[data-trs-meta]');
    var statusEl = host.querySelector('[data-trs-status]');
    var rangesEl = host.querySelector('[data-trs-ranges]');
    var slider = host.querySelector('.trs-slider');
    var zoomReadout = host.querySelector('.trs-zoom-readout');

    var tooltip = global.PtoSwimlaneTaskPattern.createTooltip();
    root.appendChild(tooltip);

    /* ── 状态 ── */
    var scene = opts.scene === 'healthy' ? 'healthy' : 'incident';
    var rowsCache = { healthy: null, incident: null };
    var rows = [];
    var ranges = {};
    var activeRange = 'step';
    var zoom = 1;
    var selectedId = null;
    var hoverId = null;
    var hitRects = [];
    var expanded = new Set(['step', 'stage-' + FAULT.stage]);
    /* 「定位聚焦」：聚光灯（js/training-spotlight.js 问题二 · 步④ 通信调度层）照到本图时打开。
       打开后整图去色，只留 rank 23 那条红色的 all-to-all 死锁条 —— 光洞把视线引到这块面板，
       面板自己再把视线引到那一条上，用户不用在 32 条泳道里自己找。
       focusRestore 记住进入前的场景/范围/展开态，退出时原样还回去。 */
    var faultFocus = false;
    var focusRestore = null;
    var focusChip = host.querySelector('[data-trs-focus]');

    function currentRows() {
      if (!rowsCache[scene]) rowsCache[scene] = buildRows(scene === 'incident');
      return rowsCache[scene];
    }

    function buildRanges() {
      var incident = scene === 'incident';
      var end = incident ? INCIDENT_WINDOW_END : STEP_END;
      var s2f = SCHED.compF[FAULT.stage][0].start;
      var out = {
        step: { start: 0, end: end, tick: 50, label: '0.00–' + end.toFixed(2) + ' ms · 整个训练步' },
        pp2: {
          start: Math.max(0, s2f - 6), end: incident ? INCIDENT_WINDOW_END : STEP_END, tick: 40,
          label: '聚焦 PP' + FAULT.stage + ' · Rank ' + rankOf(FAULT.stage, 0, 0, 0) + '–' + rankOf(FAULT.stage, 1, 1, 1)
        }
      };
      if (incident) {
        var a2a = faultCutoff(rankProfile(FAULT.stage, 1, 1, 1)).at;
        out.incident = { start: round2(a2a - 40), end: round2(a2a + 120), tick: 20, label: '事故窗口 · rank ' + FAULT.rank + ' all-to-all 未返回' };
      } else {
        out.update = { start: round2(GRAD_SYNC_START - 24), end: round2(OPT_END + 6), tick: 10, label: '更新窗口 · 梯度 AllReduce + Optimizer Step' };
      }
      return out;
    }

    var RANGE_LABELS = {
      step: '训练步全貌', pp2: 'PP' + FAULT.stage + ' 聚焦',
      incident: '事故窗口', update: '更新窗口'
    };

    function syncRangeButtons() {
      rangesEl.innerHTML = Object.keys(ranges).map(function (key) {
        return '<button type="button" class="trs-seg' + (key === activeRange ? ' is-on' : '') + '" data-trs-range="' + key + '">'
          + esc(RANGE_LABELS[key] || key) + '</button>';
      }).join('');
    }

    function applyScene() {
      rows = currentRows();
      ranges = buildRanges();
      if (!ranges[activeRange]) activeRange = 'step';
      syncRangeButtons();
      host.querySelectorAll('[data-trs-scene]').forEach(function (btn) {
        btn.classList.toggle('is-on', btn.dataset.trsScene === scene);
      });
      metaEl.textContent = WORLD.model + ' · ' + WORLD.worldSize + ' Rank · DP' + WORLD.dp + '×PP' + WORLD.pp
        + '×TP' + WORLD.tp + '×EP' + WORLD.ep + ' · ' + WORLD.schedule + ' · ' + WORLD.micro + ' micro-batch';
      selectedId = null;
      setStatus(scene === 'incident'
        ? '事故步 ' + INCIDENT_STEP + '：rank ' + FAULT.rank + ' 在 ' + FAULT.layer + ' 的 EP all-to-all 未返回，全网停在梯度同步之前 —— 点条看细节，点行头左侧 +/− 展开下一层'
        : '健康步 ' + HEALTHY_STEP + '：1F1B 前向/反向交错跑满 → DP 梯度 AllReduce → Optimizer Step 闭环');
    }

    function setStatus(text) { statusEl.textContent = text; }

    /* ── 定位聚焦 ─────────────────────────────────────────────────────────
       聚光灯把光洞开在这块面板上时调进来（见 js/training-spotlight.js 问题二 · 步④）。
       做三件事：切到事故场景 + 事故窗口、把故障 Rank 那一行滚到视野中间、整图去色只留红。
       退出时把进入前的场景/范围/展开态/缩放原样还回去 —— 用户在聚光灯之前摆好的视图不该被吞掉。 */
    function syncFocusChip() { if (focusChip) focusChip.hidden = !faultFocus; }

    function scrollFaultRowIntoView() {
      var list = visibleRows();
      var y = LAYOUT.headerH;
      for (var i = 0; i < list.length; i += 1) {
        var rowH = list[i].rowType === 'rank' && isExpanded(list[i]) ? LAYOUT.expandedRankHeight : LAYOUT.rowHeight;
        if (list[i].id === 'rank-' + FAULT.rank) {
          scroll.scrollTop = Math.max(0, y - Math.max(0, (scroll.clientHeight - rowH) / 2));
          return;
        }
        y += rowH;
      }
    }

    function setFaultFocus(on) {
      on = !!on;
      if (on === faultFocus) { if (on) { draw(); scrollFaultRowIntoView(); } return; }
      if (on) {
        focusRestore = { scene: scene, range: activeRange, expanded: new Set(expanded), zoom: zoom };
        faultFocus = true;
        scene = 'incident';
        expanded = new Set(['step', 'stage-' + FAULT.stage]);
        applyScene();
        if (ranges.incident) activeRange = 'incident';
        syncRangeButtons();
        syncFocusChip();
        scroll.scrollLeft = 0;
        setZoom(1);
        scrollFaultRowIntoView();
        draw();
        setStatus('定位聚焦：只留 rank ' + FAULT.rank + ' 在 ' + FAULT.layer + ' 的 ' + FAULT.op
          + ' 那一条红色，其余泳道已去色 —— 点右上「定位聚焦中 ✕」或任一段控可退出');
        return;
      }
      faultFocus = false;
      syncFocusChip();
      if (focusRestore) {
        scene = focusRestore.scene;
        expanded = focusRestore.expanded;
        applyScene();
        if (ranges[focusRestore.range]) activeRange = focusRestore.range;
        syncRangeButtons();
        setZoom(focusRestore.zoom);
        focusRestore = null;
      }
      draw();
    }

    // 用户自己动了工具条 = 他要自己看，聚焦让位（不还原视图，因为他正要去别处）
    function leaveFocusOnUserAction() {
      if (!faultFocus) return;
      faultFocus = false;
      focusRestore = null;
      syncFocusChip();
    }

    /* ── 行可见性 ── */
    function childRows(id) { return rows.filter(function (r) { return r.parentRow === id; }); }
    function isExpandable(r) { return childRows(r.id).length > 0 || (r.rowType === 'rank' && r.detailEvents && r.detailEvents.length > 0); }
    function isExpanded(r) { return expanded.has(r.id); }
    function parentChainOpen(r) {
      var pid = r.parentRow;
      while (pid) {
        if (!expanded.has(pid)) return false;
        var parent = rows.find(function (x) { return x.id === pid; });
        pid = parent ? parent.parentRow : null;
      }
      return true;
    }
    function visibleRows() { return rows.filter(parentChainOpen); }
    function eventsForRow(r) {
      var open = r.rowType === 'rank' && isExpanded(r);
      return (r.events || []).filter(function (item) { return open ? true : !item.metrics.detail; });
    }

    /* ── canvas 基础 ── */
    function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
    function monoFont() { return cssVar('--font-mono') || 'monospace'; }
    function colorFor(item) { return COLORS[item.kind] || COLORS.forward; }

    function roundedRect(ctx, x, y, w, h, r) {
      if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); return; }
      ctx.beginPath(); ctx.rect(x, y, w, h);
    }

    // 前向 / 反向条上的箭头底纹：不看颜色也能读出方向。
    function drawChevrons(ctx, x, y, w, h, direction) {
      if (w <= 0 || h <= 0) return;
      ctx.save();
      roundedRect(ctx, x, y, w, h, 3);
      ctx.clip();
      ctx.strokeStyle = 'rgba(255,255,255,.30)';
      ctx.lineWidth = 0.55;
      ctx.lineJoin = 'miter';
      for (var px = x - 6; px < x + w + 6; px += 6) {
        ctx.beginPath();
        if (direction === 'forward') {
          ctx.moveTo(px + 0.65, y + 0.65);
          ctx.lineTo(px + 6 - 0.65, y + h / 2);
          ctx.lineTo(px + 0.65, y + h - 0.65);
        } else {
          ctx.moveTo(px + 6 - 0.65, y + 0.65);
          ctx.lineTo(px + 0.65, y + h / 2);
          ctx.lineTo(px + 6 - 0.65, y + h - 0.65);
        }
        ctx.stroke();
      }
      ctx.restore();
    }

    function fitText(ctx, value, maxWidth) {
      var text = String(value || '');
      if (ctx.measureText(text).width <= maxWidth) return text;
      var fitted = text;
      while (fitted.length > 1 && ctx.measureText(fitted + '…').width > maxWidth) fitted = fitted.slice(0, -1);
      return fitted + '…';
    }

    function drawBarLabel(ctx, item, x, y, w, h, color) {
      if (w < 30) return;
      ctx.save();
      roundedRect(ctx, x + 1, y + 1, Math.max(0, w - 2), Math.max(0, h - 2), 2);
      ctx.clip();
      ctx.fillStyle = color || 'rgba(255,255,255,.94)';
      ctx.font = (w >= 76 ? '600 9.5px ' : '600 8.5px ') + monoFont();
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.shadowColor = 'rgba(0,0,0,.72)';
      ctx.shadowBlur = 2;
      ctx.fillText(fitText(ctx, item.label, w - 9), x + 5, y + h / 2 + 0.5);
      ctx.restore();
    }

    function drawRowExpandIcon(ctx, r, x, y, rowH) {
      if (!isExpandable(r)) return null;
      var size = 14;
      var iconY = y + Math.max(2, (rowH - size) / 2);
      ctx.save();
      roundedRect(ctx, x, iconY, size, size, 4);
      ctx.fillStyle = cssVar('--surface-1') || 'rgba(255,255,255,.9)';
      ctx.strokeStyle = cssVar('--border-strong') || 'rgba(255,255,255,.6)';
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = cssVar('--foreground') || '#111';
      ctx.font = '800 12px ' + monoFont();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(isExpanded(r) ? '−' : '+', x + size / 2, iconY + size / 2 + 0.25);
      ctx.restore();
      return { x: x, y: iconY, w: size, h: size };
    }

    function drawRowHeader(ctx, r, x, y, rowH, maxWidth, textColor) {
      var parts = String(r.label).split(' · ');
      var tag = parts[0];
      var rest = parts.slice(1).join(' · ');
      var tagH = 18;
      var tagY = y + (rowH - tagH) / 2;
      ctx.save();
      ctx.font = '650 10px ' + monoFont();
      var tagLimit = rest ? Math.max(46, maxWidth * 0.5) : maxWidth;
      var tagText = fitText(ctx, tag, Math.max(24, tagLimit - 14));
      var tagW = Math.min(tagLimit, Math.max(30, ctx.measureText(tagText).width + 14));
      roundedRect(ctx, x, tagY, tagW, tagH, 6);
      ctx.fillStyle = r.meta && r.meta.rank === FAULT.rank && scene === 'incident'
        ? rgba(COLORS.fault, 0.22) : (cssVar('--surface-2') || 'rgba(255,255,255,.06)');
      ctx.strokeStyle = r.meta && r.meta.rank === FAULT.rank && scene === 'incident'
        ? COLORS.fault : (cssVar('--border-strong') || 'rgba(255,255,255,.28)');
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(tagText, x + 7, tagY + tagH / 2 + 0.25);
      if (rest) {
        ctx.font = '600 10px ' + monoFont();
        ctx.fillStyle = cssVar('--foreground-secondary') || textColor;
        ctx.fillText(fitText(ctx, rest, Math.max(16, maxWidth - tagW - 8)), x + tagW + 8, y + rowH / 2 + 0.25);
      }
      ctx.restore();
      return { x: x, y: y, w: maxWidth, h: rowH };
    }

    function drawExpandedRankHeader(ctx, r, x, y, rowH, maxWidth, textColor) {
      var tagH = LAYOUT.eventHeight;
      ctx.save();
      ctx.font = '650 10px ' + monoFont();
      var tagText = fitText(ctx, r.label, Math.max(24, maxWidth - 86));
      var tagW = Math.min(Math.max(54, maxWidth - 72), Math.max(54, ctx.measureText(tagText).width + 14));
      roundedRect(ctx, x, y + LAYOUT.trackTop, tagW, tagH, 6);
      ctx.fillStyle = cssVar('--surface-2') || 'rgba(255,255,255,.06)';
      ctx.strokeStyle = cssVar('--border-strong') || 'rgba(255,255,255,.28)';
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(tagText, x + 7, y + LAYOUT.trackTop + tagH / 2 + 0.25);
      ctx.fillStyle = cssVar('--foreground-muted') || '#888';
      ctx.font = '600 9px ' + monoFont();
      ctx.textAlign = 'right';
      TRACK_LABELS.forEach(function (label, track) {
        ctx.fillText(label, x + maxWidth, y + LAYOUT.trackTop + LAYOUT.eventHeight / 2 + track * LAYOUT.trackPitch);
      });
      ctx.restore();
      return { x: x, y: y, w: maxWidth, h: rowH };
    }

    function drawConnector(ctx, r, y, rowH, color) {
      var depth = r.depth || 0;
      if (!depth) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.72;
      ctx.setLineDash([2, 3]);
      for (var level = 1; level <= depth; level += 1) {
        var gx = 12 + level * 15;
        ctx.beginPath(); ctx.moveTo(gx + 0.5, y + 1); ctx.lineTo(gx + 0.5, y + rowH - 1); ctx.stroke();
      }
      ctx.setLineDash([]);
      var bx = 12 + depth * 15;
      ctx.beginPath();
      ctx.moveTo(bx + 0.5, y + rowH / 2 + 0.5);
      ctx.lineTo(bx + 11.5, y + rowH / 2 + 0.5);
      ctx.stroke();
      ctx.restore();
    }

    /* ── 主绘制 ── */
    function draw() {
      if (!canvas.isConnected) return;
      var list = visibleRows();
      var range = ranges[activeRange];
      if (!range) return;
      var dpr = Math.min(global.devicePixelRatio || 1, 2);
      var gutter = LAYOUT.gutter;
      var headerH = LAYOUT.headerH;
      var layoutY = headerH;
      var layouts = list.map(function (r) {
        var rowH = r.rowType === 'rank' && isExpanded(r) ? LAYOUT.expandedRankHeight : LAYOUT.rowHeight;
        var item = { row: r, y: layoutY, rowH: rowH };
        layoutY += rowH;
        return item;
      });

      var viewportW = Math.max(560, scroll.clientWidth);
      var width = Math.round(gutter + Math.max(480, viewportW - gutter) * zoom);
      var height = layoutY + 12;
      surface.style.width = width + 'px';
      surface.style.height = height + 'px';
      canvas.style.width = viewportW + 'px';
      canvas.style.height = height + 'px';
      canvas.width = Math.round(viewportW * dpr);
      canvas.height = Math.round(height * dpr);

      var ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, viewportW, height);
      ctx.translate(-scroll.scrollLeft, 0);
      hitRects = [];

      var plotLeft = gutter;
      var plotRight = width - 16;
      var span = range.end - range.start || 1;
      var toX = function (ms) { return plotLeft + ((ms - range.start) / span) * (plotRight - plotLeft); };
      var text = cssVar('--foreground') || '#fff';
      var muted = cssVar('--foreground-muted') || '#888';
      var subtle = cssVar('--border-subtle') || 'rgba(255,255,255,.08)';
      var surfaceColor = cssVar('--surface-2') || '#1c1c1c';

      ctx.fillStyle = surfaceColor;
      ctx.fillRect(0, 0, width, headerH);

      // 时间刻度
      ctx.font = '500 10px ' + monoFont();
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
      var tickStep = range.tick;
      for (var tick = Math.ceil(range.start / tickStep) * tickStep; tick <= range.end; tick += tickStep) {
        var px = toX(tick);
        ctx.strokeStyle = subtle;
        ctx.beginPath(); ctx.moveTo(px + 0.5, headerH); ctx.lineTo(px + 0.5, height); ctx.stroke();
        ctx.fillStyle = muted;
        ctx.fillText(tick + ' ms', px, 20);
      }

      // 事故场景：在死锁起点画一条竖向红线，把「之前 / 之后」劈开
      if (scene === 'incident') {
        var faultAt = faultCutoff(rankProfile(FAULT.stage, 1, 1, 1)).at;
        if (faultAt >= range.start && faultAt <= range.end) {
          var fx = toX(faultAt);
          ctx.save();
          ctx.strokeStyle = rgba(COLORS.fault, 0.85);
          ctx.lineWidth = 1.2;
          ctx.setLineDash([4, 3]);
          ctx.beginPath(); ctx.moveTo(fx + 0.5, headerH - 12); ctx.lineTo(fx + 0.5, height); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = COLORS.fault;
          ctx.font = '700 9.5px ' + monoFont();
          ctx.textAlign = 'left';
          ctx.fillText('rank ' + FAULT.rank + ' all-to-all 未返回', fx + 5, headerH - 18);
          ctx.restore();
        }
      }

      // 行 + 事件条
      layouts.forEach(function (layout, index) {
        var r = layout.row, y = layout.y, rowH = layout.rowH;
        if (index % 2) { ctx.fillStyle = cssVar('--surface-disabled') || 'rgba(255,255,255,.02)'; ctx.fillRect(0, y, width, rowH); }
        ctx.strokeStyle = subtle;
        ctx.beginPath(); ctx.moveTo(0, y + rowH + 0.5); ctx.lineTo(width, y + rowH + 0.5); ctx.stroke();

        var openRank = r.rowType === 'rank' && isExpanded(r);
        if (openRank) {
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,.055)';
          for (var t = 1; t < TRACK_LABELS.length; t += 1) {
            var dy = y + LAYOUT.trackTop + t * LAYOUT.trackPitch - 1;
            ctx.beginPath(); ctx.moveTo(gutter, dy + 0.5); ctx.lineTo(width, dy + 0.5); ctx.stroke();
          }
          ctx.restore();
        }

        eventsForRow(r).forEach(function (item) {
          if (item.endMs < range.start || item.startMs > range.end) return;
          var s = Math.max(item.startMs, range.start);
          var e = Math.min(item.endMs, range.end);
          var barX = toX(s);
          var barW = Math.max(3, toX(e) - barX);
          var track = openRank ? Math.max(0, Math.min(TRACK_LABELS.length - 1, Number(item.metrics.track) || 0)) : 0;
          var barY = openRank
            ? y + LAYOUT.trackTop + track * LAYOUT.trackPitch
            : y + Math.round((rowH - LAYOUT.eventHeight) / 2);
          var selected = item.id === selectedId;
          /* 定位聚焦下：非 fault 的条压成中性灰 + 低透明；save/restore 包住整段绘制，
             各分支里的 return 直接跳出也不会把画布状态漏出去（外层统一 restore）。 */
          var dim = faultFocus && item.kind !== 'fault';
          var barColor = dim ? DIM_COLOR : colorFor(item);
          ctx.save();
          if (dim) ctx.globalAlpha = DIM_ALPHA;
          paintEvent(item, dim, barColor, barX, barY, barW, selected, text);
          ctx.restore();
        });

        // 一条事件的实际绘制。抽成函数是为了让上面那层 save/restore 能兜住所有 return 分支。
        function paintEvent(item, dim, barColor, barX, barY, barW, selected, text) {
          if (item.kind === 'hold') {
            var bandH = 15;
            ctx.save();
            roundedRect(ctx, barX, barY + 1, barW, bandH, 4);
            ctx.fillStyle = rgba(barColor, selected ? 0.32 : 0.18);
            ctx.strokeStyle = selected ? '#fff' : rgba(barColor, 0.78);
            ctx.lineWidth = selected ? 1.6 : 1.1;
            ctx.fill();
            ctx.stroke();
            ctx.save();
            roundedRect(ctx, barX, barY + 1, barW, bandH, 4);
            ctx.clip();
            ctx.fillStyle = text;
            ctx.font = '700 9.5px ' + monoFont();
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(item.label, barX + 8, barY + 1 + bandH / 2);
            ctx.restore();
            ctx.restore();
            hitRects.push({ x: barX, y: barY - 1, w: barW, h: bandH + 6, item: item, action: 'select' });
            return;
          }

          if (item.kind === 'ghost') {
            // 「本该发生但没发生」：空心虚线框，和实心条区分开
            ctx.save();
            roundedRect(ctx, barX, barY, barW, LAYOUT.eventHeight, 3);
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = rgba(dim ? DIM_COLOR : COLORS.update, selected ? 0.95 : 0.6);
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
            drawBarLabel(ctx, item, barX, barY, barW, LAYOUT.eventHeight, rgba(dim ? DIM_COLOR : COLORS.update, 0.92));
            hitRects.push({ x: barX, y: barY - 1, w: barW, h: LAYOUT.eventHeight + 4, item: item, action: 'select' });
            return;
          }

          global.PtoSwimlaneTaskPattern.drawTaskBar(ctx, {
            x: barX, y: barY, width: barW, height: LAYOUT.eventHeight,
            baseColor: barColor,
            task: { label: item.label, opName: item.label },
            isSelected: selected,
            isRelated: !dim && !!(hoverId && item.metrics.microbatch && hoverId === item.metrics.microbatch),
            fontFamily: monoFont()
          });
          if (item.kind === 'forward' || item.kind === 'backward') {
            drawChevrons(ctx, barX, barY, barW, LAYOUT.eventHeight, item.kind);
            drawBarLabel(ctx, item, barX, barY, barW, LAYOUT.eventHeight);
          }
          if (item.kind === 'fault') {
            // 聚焦时再给红条压一圈外发光：去色之后它是画面上唯一有彩度的东西，直接跳出来
            ctx.save();
            if (faultFocus) { ctx.shadowColor = rgba(COLORS.fault, 0.95); ctx.shadowBlur = 12; }
            roundedRect(ctx, barX, barY - 1, barW, LAYOUT.eventHeight + 2, 3);
            ctx.strokeStyle = COLORS.fault;
            ctx.lineWidth = faultFocus ? 1.8 : 1.4;
            ctx.stroke();
            ctx.restore();
          }
          hitRects.push({ x: barX, y: barY - 1, w: barW, h: LAYOUT.eventHeight + 4, item: item, action: 'select' });
        }
      });

      // 左侧行头（不随横向滚动）
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = surfaceColor;
      ctx.fillRect(0, 0, gutter, headerH);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = text;
      ctx.font = '600 11px ' + (cssVar('--font-sans') || 'sans-serif');
      ctx.fillText('泳道 / Rank', 12, 18);
      ctx.fillStyle = muted;
      ctx.font = '500 9.5px ' + monoFont();
      ctx.fillText(range.label, 12, 33);

      layouts.forEach(function (layout, index) {
        var r = layout.row, y = layout.y, rowH = layout.rowH;
        ctx.fillStyle = index % 2 ? (cssVar('--surface-1') || surfaceColor) : (cssVar('--background') || '#111');
        ctx.fillRect(0, y, gutter, rowH);
        ctx.strokeStyle = subtle;
        ctx.beginPath(); ctx.moveTo(0, y + rowH + 0.5); ctx.lineTo(gutter, y + rowH + 0.5); ctx.stroke();
        // 底色画完再压暗：暗的只是行头内容（连接线/展开键/标签），不是那格底
        var dimRow = faultFocus && r.id !== 'rank-' + FAULT.rank;
        ctx.save();
        if (dimRow) ctx.globalAlpha = 0.42;
        drawConnector(ctx, r, y, rowH, subtle);
        var contentX = 12 + (r.depth || 0) * 15 + (r.depth ? 13 : 0);
        var iconRect = drawRowExpandIcon(ctx, r, contentX, y, rowH);
        var labelX = contentX + (iconRect ? 20 : 0);
        var maxW = gutter - labelX - 10;
        var labelRect = (r.rowType === 'rank' && isExpanded(r))
          ? drawExpandedRankHeader(ctx, r, labelX, y, rowH, maxW, text)
          : drawRowHeader(ctx, r, labelX, y, rowH, maxW, text);
        ctx.restore();
        hitRects.push({ x: labelRect.x, y: labelRect.y, w: labelRect.w, h: labelRect.h, row: r, action: 'row-label', fixed: true });
        if (iconRect) hitRects.push({ x: iconRect.x, y: iconRect.y, w: iconRect.w, h: iconRect.h, row: r, action: 'toggle-row', fixed: true });
      });
      ctx.strokeStyle = cssVar('--border-default') || subtle;
      ctx.beginPath(); ctx.moveTo(gutter - 0.5, 0); ctx.lineTo(gutter - 0.5, height); ctx.stroke();
    }

    /* ── tooltip / 交互 ── */
    var METRIC_LABELS = {
      track: null, detail: null, phase: '阶段', microbatch: 'micro-batch', rankId: 'Rank', stage: 'PP stage',
      layers: '层段', scope: '层段', tpShard: 'TP 分片', epShard: 'EP 分片', localTensorShard: '本卡张量分片',
      localExperts: '本卡 expert', expertLoad: 'expert 负载', weightGrad: '权重梯度', inputGrad: '输入梯度',
      collective: '集合通信', group: '通信组', payload: '载荷', bandwidth: '带宽', transactionId: '事务号',
      direction: '方向', syncWait: '同步等待', gradNorm: 'grad_norm', optimizer: '优化器', gradClip: '梯度裁剪',
      lr: '学习率', updated: '更新量', reason: '原因', status: '状态', blockedBy: '被谁阻塞', sendBuffer: 'send buffer',
      recvBuffer: 'recv buffer', layer: '层', memory: '显存', producer: '生产者', consumer: '消费者',
      checkpoint: 'checkpoint', peakRank: '峰值卡', note: '备注', rankGroup: 'Rank 组', envelope: '包络',
      topology: '并行拓扑', microbatches: '切分', worldSize: '规模', schedule: '调度', step: 'step', loss: 'loss',
      syncPoint: '同步点'
    };

    function kindLabel(kind) {
      return { step: '训练步', forward: '前向', backward: '反向', wait: '空等', comm: '通信',
        reduce: '梯度同步', update: '参数更新', hold: '激活驻留', loss: 'Loss', fault: '故障阻塞',
        ghost: '未发生' }[kind] || kind;
    }

    function tooltipHtml(item) {
      var lines = [
        ['类型', kindLabel(item.kind)],
        ['时间窗', fmtMs(item.startMs) + ' → ' + fmtMs(item.endMs)],
        ['持续', fmtMs(item.durationMs)]
      ];
      Object.keys(item.metrics).forEach(function (key) {
        var label = METRIC_LABELS[key];
        if (!label) return;
        lines.push([label, item.metrics[key]]);
      });
      return '<div class="pto-swimlane-task-tooltip__title">' + esc(item.label) + '</div>'
        + lines.map(function (pair) {
          return '<div class="pto-swimlane-task-tooltip__row"><span class="pto-swimlane-task-tooltip__key">'
            + esc(pair[0]) + '</span><span class="pto-swimlane-task-tooltip__value">' + esc(pair[1]) + '</span></div>';
        }).join('');
    }

    function rowTooltipHtml(r) {
      var detail = r.rowType === 'step'
        ? '一个训练步的全貌：' + WORLD.pp + ' 个 PP stage 上下摞着看，展开任一 stage 看它的 8 张卡。'
        : r.rowType === 'stage'
          ? '训练步 → PP' + r.meta.stage + '（' + r.meta.scope + '）→ Rank ' + r.meta.firstRank + '–' + r.meta.lastRank + '；展开显示 DP2 × TP2 × EP2 的 8 个物理 Rank 与本 stage 的激活驻留。'
          : r.rowType === 'rank'
            ? 'PP' + r.meta.stage + ' → Rank ' + r.meta.rank + '（DP' + r.meta.dp + ' / TP' + r.meta.tp + ' / EP' + r.meta.ep + '）；展开后在同一行内分「计算流 / 通信流 / 更新流」三轨。'
            : 'PP' + r.meta.stage + ' 前向存下的激活一直驻留到反向消费才释放。';
      return '<div class="pto-swimlane-task-tooltip__title">' + esc(r.label) + '</div>'
        + '<div class="pto-swimlane-task-tooltip__row"><span class="pto-swimlane-task-tooltip__key">层级</span>'
        + '<span class="pto-swimlane-task-tooltip__value">' + esc(detail) + '</span></div>';
    }

    function hitAt(ev) {
      var rect = canvas.getBoundingClientRect();
      var localX = ev.clientX - rect.left;
      var py = ev.clientY - rect.top;
      for (var i = hitRects.length - 1; i >= 0; i -= 1) {
        var hit = hitRects[i];
        var px = hit.fixed ? localX : localX + scroll.scrollLeft;
        if (px >= hit.x && px <= hit.x + hit.w && py >= hit.y && py <= hit.y + hit.h) return hit;
      }
      return null;
    }

    canvas.addEventListener('mousemove', function (ev) {
      var hit = hitAt(ev);
      if (hit && hit.action === 'row-label') {
        canvas.style.cursor = 'help';
        global.PtoSwimlaneTaskPattern.showTooltip(tooltip, { label: hit.row.label }, ev, {
          bounds: root, getTooltipHtml: function () { return rowTooltipHtml(hit.row); }
        });
        return;
      }
      var item = hit && hit.item ? hit.item : null;
      var nextHover = item && item.metrics.microbatch ? item.metrics.microbatch : null;
      if (nextHover !== hoverId) { hoverId = nextHover; draw(); }
      if (!item) {
        global.PtoSwimlaneTaskPattern.hideTooltip(tooltip);
        canvas.style.cursor = hit ? 'pointer' : 'crosshair';
        return;
      }
      canvas.style.cursor = 'pointer';
      global.PtoSwimlaneTaskPattern.showTooltip(tooltip, item, ev, {
        bounds: root, getTooltipHtml: function () { return tooltipHtml(item); }
      });
    });

    canvas.addEventListener('mouseleave', function () {
      hoverId = null;
      global.PtoSwimlaneTaskPattern.hideTooltip(tooltip);
      draw();
    });

    canvas.addEventListener('click', function (ev) {
      var hit = hitAt(ev);
      if (!hit) return;
      if (hit.action === 'toggle-row') {
        if (expanded.has(hit.row.id)) expanded.delete(hit.row.id); else expanded.add(hit.row.id);
        draw();
        setStatus((expanded.has(hit.row.id) ? '展开' : '收起') + ' ' + hit.row.label);
        return;
      }
      if (hit.item) {
        selectedId = hit.item.id;
        draw();
        setStatus(hit.item.label + ' · ' + kindLabel(hit.item.kind) + ' · '
          + fmtMs(hit.item.startMs) + ' → ' + fmtMs(hit.item.endMs)
          + (hit.item.metrics.reason ? ' · ' + hit.item.metrics.reason : ''));
      }
    });

    /* ── 工具栏 ── */
    host.addEventListener('click', function (ev) {
      if (ev.target.closest('[data-trs-focus]')) { setFaultFocus(false); return; }
      var sceneBtn = ev.target.closest('[data-trs-scene]');
      if (sceneBtn) {
        leaveFocusOnUserAction();
        scene = sceneBtn.dataset.trsScene;
        expanded = new Set(['step', 'stage-' + FAULT.stage]);
        applyScene();
        draw();
        return;
      }
      var rangeBtn = ev.target.closest('[data-trs-range]');
      if (rangeBtn) {
        leaveFocusOnUserAction();
        activeRange = rangeBtn.dataset.trsRange;
        syncRangeButtons();
        scroll.scrollLeft = 0;
        draw();
        return;
      }
      var zoomBtn = ev.target.closest('[data-trs-zoom]');
      if (zoomBtn) {
        var idx = ZOOM_STEPS.indexOf(zoom);
        setZoom(ZOOM_STEPS[Math.max(0, Math.min(ZOOM_STEPS.length - 1, idx + (zoomBtn.dataset.trsZoom === 'in' ? 1 : -1)))]);
      }
    });

    function setZoom(next) {
      zoom = next;
      slider.value = String(ZOOM_STEPS.indexOf(zoom));
      zoomReadout.textContent = zoom + '×';
      draw();
    }
    slider.addEventListener('input', function () { setZoom(ZOOM_STEPS[Number(slider.value)] || 1); });

    var scrollFrame = 0;
    scroll.addEventListener('scroll', function () {
      if (scrollFrame) return;
      scrollFrame = requestAnimationFrame(function () { scrollFrame = 0; draw(); });
    });

    var resizeFrame = 0;
    function onResize() {
      if (resizeFrame) return;
      resizeFrame = requestAnimationFrame(function () { resizeFrame = 0; draw(); });
    }
    global.addEventListener('resize', onResize);
    // 主题切换后 css 变量变了，需要重画（v2 顶栏有深/浅切换）
    new MutationObserver(onResize).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    applyScene();
    draw();

    var instance = {
      draw: draw,
      setScene: function (s) { leaveFocusOnUserAction(); scene = s; applyScene(); draw(); },
      setFaultFocus: setFaultFocus
    };
    host.__trsInstance = instance;
    lastInstance = instance;
    return instance;
  }

  /* focusFault / clearFocus 是给聚光灯（js/training-spotlight.js）用的跨组件开关：
     它只知道"要照亮底部这块面板"，具体怎么把视线收到那一条上由本模块自己决定。
     没渲染过就什么都不做，聚光灯那边不必判空。 */
  function focusFault() { if (lastInstance) lastInstance.setFaultFocus(true); }
  function clearFocus() { if (lastInstance) lastInstance.setFaultFocus(false); }

  global.PtoTrainingRankSwimlane = {
    render: render, focusFault: focusFault, clearFocus: clearFocus, WORLD: WORLD, FAULT: FAULT
  };
})(window);
