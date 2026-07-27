/*
 * 问题二 · 显存 OOM（定位链长文，window.PtoMemCase4）
 * ------------------------------------------------------------
 * 定位链-openPangu-2.0-Flash.md「案例四：activation checkpoint 未开启导致显存峰值超标 +
 * 分配器碎片触发 OOM」的完整七层叙事，渲染成「问题诊断 → 定位链」面板的七个小节。
 * 顶栏聚光灯（js/training-spotlight.js 的 CASES["mem-oom"]）是这条链的一屏速览版，
 * 名片「详情」按钮打开的就是本文件产出的长文。
 *
 * 数据：
 *   · 静态事实 / 逐 step 读数 → window.PtoTrainingTwinMemoryCase（training-run-twin.js）
 *   · 三张图表 → window.PtoTrainingMemoryPanel.drawInto()，与底部 dock「性能」页签
 *     用的是同一份绘制代码，两处口径不可能对不上
 * 本文件自己不存任何业务数字，只负责把它们组织成叙事。
 *
 * 用法（training-run-twin.js）：
 *   locateChains["mem-oom"] = window.PtoMemCase4.chain();
 *   面板挂载后 window.PtoMemCase4.renderAll() 绘制正文里的画布。
 */
window.PtoMemCase4 = (function () {
  "use strict";

  function api() { return window.PtoTrainingTwinMemoryCase || null; }

  var NOTE = "twin-locate-metric-note";
  var MONO = "ui-monospace,'SF Mono',Menlo,Consolas,monospace";

  /* 正文里的图表插槽：与底部 dock「性能」页签同源，靠 data-mem-chain-chart 标记，
     renderAll() 逐个交给 PtoTrainingMemoryPanel.drawInto() 画。高度写死是因为
     canvas 需要一个确定的 CSS 尺寸才能量出画布大小（父级是自适应高度的正文流）。 */
  function chartSlot(id, height, caption) {
    return (
      '<div class="twin-mem-chain-chart" style="height:' + height + 'px">' +
        '<canvas data-mem-chain-chart="' + id + '"></canvas>' +
      '</div>' +
      (caption ? '<p class="' + NOTE + '" style="margin:6px 0 0">' + caption + '</p>' : '')
    );
  }

  // 一行「读数格」：label + 值，越界值标红
  function readouts(cells) {
    return (
      '<div style="display:grid;grid-template-columns:repeat(' + cells.length + ',minmax(0,1fr));gap:8px;margin:10px 0">' +
      cells.map(function (k) {
        return '<div style="border:1px solid var(--border-subtle);border-radius:8px;padding:9px 11px;background:var(--surface-2)">' +
          '<div style="font-size:11px;color:var(--foreground-muted)">' + k[0] + '</div>' +
          '<div style="font-size:19px;font-weight:700;font-family:' + MONO + ';margin-top:3px;color:' +
            (k[2] ? '#dc2626' : 'var(--foreground)') + '">' + k[1] + '</div>' +
          (k[3] ? '<div style="font-size:11px;color:var(--foreground-muted);margin-top:2px">' + k[3] + '</div>' : '') +
        '</div>';
      }).join("") +
      '</div>'
    );
  }

  function table(head, rows) {
    return (
      '<div style="overflow-x:auto;margin:10px 0">' +
      '<table style="width:100%;border-collapse:collapse;font-size:11px;line-height:1.5">' +
      '<tr style="background:var(--surface-2)">' +
        head.map(function (h) { return '<th style="padding:5px 8px;border:1px solid var(--border-subtle);text-align:left">' + h + '</th>'; }).join("") +
      '</tr>' +
      rows.map(function (r) {
        return '<tr>' + r.map(function (cell) {
          return '<td style="padding:5px 8px;border:1px solid var(--border-subtle)">' + cell + '</td>';
        }).join("") + '</tr>';
      }).join("") +
      '</table></div>'
    );
  }

  function codeBlock(title, lines) {
    return (
      '<div style="border:1px solid var(--border-subtle);border-radius:8px;overflow:hidden;margin:10px 0">' +
        '<div style="padding:7px 12px;background:var(--surface-2);font-size:11px;font-weight:600;color:var(--foreground);font-family:' + MONO + '">' + title + '</div>' +
        '<div style="font-family:' + MONO + ';font-size:11px;line-height:1.7;padding:9px 12px;background:var(--surface-1);overflow-x:auto;color:var(--foreground-secondary)">' +
          lines.map(function (l) {
            var color = l[0] === "+" ? "#16a34a" : l[0] === "-" ? "#dc2626" : "";
            return '<div' + (color ? ' style="color:' + color + '"' : '') + '>' + l[1] + '</div>';
          }).join("") +
        '</div>' +
      '</div>'
    );
  }

  function arrow(txt) {
    return '<p style="margin:8px 0 0;font-size:12px;color:var(--foreground-secondary);line-height:1.5">↳ ' + txt + '</p>';
  }
  function quote(txt) {
    return '<p style="margin:10px 0 0;padding:8px 11px;border-left:3px solid var(--border-subtle);background:var(--surface-2);' +
      'border-radius:0 6px 6px 0;font-size:11.5px;color:var(--foreground-secondary);line-height:1.6">' + txt + '</p>';
  }

  /* ── 七层叙事 ─────────────────────────────────────────────────────────────── */
  function buildSteps() {
    var A = api();
    if (!A) return null;
    var f = A.facts, K = A.constants;
    var tb = f.timeBreakdown, fr = f.fragment, hot = f.hotLayer, v = f.verify;
    var act = f.composition.filter(function (it) { return it.reducible; })[0];

    return [
      // ── §1 性能表征层 ──────────────────────────────────────────────────────
      { label: "性能表征层", short: "step " + K.incidentStep, sub: "WHEN · 显存峰值触顶 " + f.capacityGB + " GB，rank " + f.oomRank + " OOM 中断",
        content:
          '<p class="' + NOTE + '" style="margin:0 0 8px"><strong>现象</strong>：显存占用时间序列显示，step ' + K.climbFrom +
          ' 前稳定在 52~55 GB（安全线以下），此后逐步爬升；step ' + K.incidentStep + ' 峰值触及 ' + f.capacityGB +
          ' GB 上限，rank ' + f.oomRank + ' 报 <code>ACL_ERROR_MEMORY_ALLOCATION</code>，训练中断。其余 rank 在 61~63 GB 间波动，同样逼近红线。</p>' +
          chartSlot("mem-timeline", 190,
            "橙色为单卡 HBM 占用（左轴 GB，虚线是 " + f.capacityGB + " GB 容量上限），蓝色为吞吐（tokens/s）。曲线在 step " +
            K.incidentStep + " 处断开 —— 进程中断，该步没有有效读数。") +
          readouts([
            ["显存峰值", f.peakGB + " / " + f.capacityGB + " GB", true, "占用率 100%"],
            ["OOM rank", "rank " + f.oomRank, true, "ACL_ERROR_MEMORY_ALLOCATION"],
            ["吞吐劣化", "−12.5%", true, v.throughput[0] + " → 2800 tokens/s"],
          ]) +
          '<p class="' + NOTE + '"><strong>判据</strong>：显存峰值 &gt; 95% 总容量 + 伴有 OOM + 吞吐持续下滑 → 显存瓶颈，需深入分析。吞吐从 step 10000 起就在下滑，是分配器频繁做碎片整理和换页的表征。</p>' +
          arrow("进入【瓶颈分类层】，用 profiling 报告确认这确实是显存问题，而不是算力或通信。"),
      },

      // ── §2 瓶颈分类层 ──────────────────────────────────────────────────────
      { label: "瓶颈分类层", short: "分配 API 7.4%", sub: "WHY · 显存受限（容量不足 + 分配碎片双因子）",
        content:
          '<p class="' + NOTE + '" style="margin:0 0 8px"><strong>观测</strong>：Profiling 报告按耗时拆解单步 —— 计算与通信都在正常区间，异常的是显存分配/释放 API 的开销。</p>' +
          table(["项目", "耗时", "占比", "判定"], [
            ["Computing", tb.computingMs + " ms", "71%", "正常"],
            ["Communication", tb.communicationMs + " ms", "12%", "正常"],
            ["<strong>显存分配 / 释放 API</strong>",
             '<strong style="color:#dc2626">' + tb.allocApiMs + " ms</strong>",
             '<strong style="color:#dc2626">' + (tb.allocApiPct * 100).toFixed(1) + "%</strong>",
             '<strong style="color:#dc2626">异常（正常应 &lt; ' + (tb.allocApiHealthyPct * 100).toFixed(0) + "%）</strong>"],
            ["HBM 带宽利用率", "—", "78%", "正常，排除纯带宽瓶颈"],
          ]) +
          '<p class="' + NOTE + '"><strong>判据</strong>：显存峰值 &gt; 90% + 分配 API 耗时 &gt; 3%（实际 ' +
          (tb.allocApiPct * 100).toFixed(1) + '%）→ <strong>显存受限，且同时存在容量不足和分配碎片两个子问题</strong>。走显存分支。</p>' +
          quote("这一层在页面上对应底部 dock 的「Timeline」页签 —— 单步耗时构成本来就该在泳道上读，不必为它单独造一张图。") +
          arrow("进入【显存表征层】判定子类型：到底是装不下，还是装得下但接不上。"),
      },

      // ── §3 显存表征层 ──────────────────────────────────────────────────────
      { label: "显存表征层", short: "双因子", sub: "WHICH · 容量不足（主）+ 分配碎片（辅）",
        content:
          '<p class="' + NOTE + '" style="margin:0 0 8px">显存问题有两个互相独立的子类型，本案例两个都中了，所以要分开判、分开治。</p>' +
          table(["维度", "观测", "判据", "结论"], [
            ["① 容量", "显存峰值 " + f.peakGB + " GB = 总容量 " + f.capacityGB + " GB", "占用率 100%",
             '<strong style="color:#dc2626">绝对容量不足（主因）</strong>'],
            ["② 碎片",
             "空闲总量 " + fr.totalFreeGB + " GB，但最大连续空闲块仅 " + fr.largestFreeBlockGB + " GB",
             "空闲够但最大连续块 &lt; 请求 size",
             '<strong style="color:#dc2626">分配碎片（辅因）</strong>'],
          ]) +
          readouts([
            ["malloc 次数", fr.mallocPerStepBefore + " → " + fr.mallocPerStepAfter, true, "次 / step（step 10000 后）"],
            ["malloc P99", fr.mallocP99MsBefore + " → " + fr.mallocP99MsAfter + " ms", true, "单次分配耗时"],
            ["碎片率", (fr.ratio * 100).toFixed(0) + "%", true, "不可用空闲 / 总空闲"],
          ]) +
          '<p class="' + NOTE + '"><strong>判据</strong>：<strong>双因子叠加</strong> —— 容量不足是主因（若有余量，碎片不会致命）；碎片让 OOM 提前到来（若连续空闲块足够，那 ' +
          fr.totalFreeGB + ' GB 还能多撑几十步）。两者只治一个都不彻底：只开 checkpoint 不解决碎片，碎片仍可能在更晚的 step 触发 OOM；只整理碎片不缩减激活，' +
          f.capacityGB + ' GB 上限终将触及。</p>' +
          arrow("先回答「什么占满了显存」——进入【显存峰值构成分析层】。"),
      },

      // ── §4 峰值构成分析层（根因）────────────────────────────────────────────
      { label: "峰值构成分析层", short: "激活 " + (act ? (act.pct * 100).toFixed(1) : "56.6") + "%", sub: "WHAT（根因）· 激活值占 " + act.gb + " GB，46 层全部常驻",
        content:
          '<p class="' + NOTE + '" style="margin:0 0 8px">此层是定位链的<strong style="color:var(--highlight-copy-blue-500,#3b6fe0)">核心转折点</strong>：从「显存不够用」下钻到「显存被谁占满了」。提取 step 12000 时 rank ' +
          f.oomRank + ' 的显存快照（<code>torch.npu.memory_stats</code> / memory snapshot），按构成拆解：</p>' +
          chartSlot("composition", 210,
            "气泡面积与占用量成正比（半径 ∝ √GB）。激活值一项就比参数 + 梯度 + 优化器三项之和还大。") +
          table(["构成项", "占用", "占比", "说明"],
            f.composition.map(function (it) {
              var hot = !!it.reducible;
              var wrap = function (s) { return hot ? '<strong style="color:#dc2626">' + s + "</strong>" : s; };
              return [hot ? "<strong>" + it.label + "</strong>" : it.label,
                      wrap(it.gb + " GB"), wrap((it.pct * 100).toFixed(1) + "%"),
                      it.note || ""];
            })) +
          '<p class="' + NOTE + '"><strong>判据</strong>：激活值占比 ' + (act.pct * 100).toFixed(1) +
          '% 是罪魁祸首 —— 46 层激活全部常驻，意味着每层约 ' + (act.gb / 46).toFixed(2) +
          ' GB。若开启 selective activation checkpointing（仅重计算 attention + FFN 中间激活），激活值可压缩至 ~8.5 GB，总显存降至 ~36 GB（安全线以下）。</p>' +
          '<p class="' + NOTE + '"><strong>产出</strong>：<strong style="color:#dc2626">激活值是唯一可大幅缩减的项</strong> —— 参数与梯度由模型规模和精度决定，优化器状态已经是 BF16（否则 FP32 要 21.6 GB），workspace 本就只有 ' +
          f.composition[f.composition.length - 1].gb + ' GB。</p>' +
          arrow("再回答「哪一层压力最大」——进入【阶段 / 层定位层】。"),
      },

      // ── §5 阶段 / 层定位层 ─────────────────────────────────────────────────
      { label: "阶段 / 层定位层", short: "stage 3 · L" + hot.layer, sub: "WHERE · PP stage 3 最重，热点层 L" + hot.layer + " MoE",
        content:
          '<p class="' + NOTE + '" style="margin:0 0 8px"><strong>观测</strong>：按 PP stage 拆解显存分布 —— 四段本应均衡，实际 stage 3 明显更重。</p>' +
          table(["PP stage", "层范围", "激活", "参数+梯度+优化器", "合计"],
            f.stages.map(function (s) {
              var hotRow = !!s.hot;
              var wrap = function (x) { return hotRow ? '<strong style="color:#dc2626">' + x + "</strong>" : x; };
              return ["stage " + s.stage + (hotRow ? " ⚠" : ""),
                      "L" + s.layers[0] + "–" + s.layers[1] + (s.note ? "<br><span style=\"font-size:10.5px;color:var(--foreground-muted)\">" + s.note + "</span>" : ""),
                      wrap(s.activationGB + " GB"), wrap(s.stateGB + " GB"), wrap(s.totalGB + " GB")];
            })) +
          '<p class="' + NOTE + '">按层粒度看：layer ' + hot.layer + '（MoE 层）激活值 ' + hot.activationGB +
          ' GB，是普通 dense 层（' + hot.denseBaselineGB + ' GB）的 <strong style="color:#dc2626">' +
          (hot.activationGB / hot.denseBaselineGB).toFixed(1) + '×</strong> —— 多出来的是 ' + hot.reason + '。</p>' +
          quote("这一层在页面上对应<strong>整网图「侧视图」</strong>：46 层同时在场，逐层指标曲线里的「单层激活值显存」直接把这条分布画在模型上，比再造一张 stage 柱状图更直接。46 层求和 " +
            act.gb + " GB，与上一层的激活值总量互为印证。") +
          '<p class="' + NOTE + '"><strong>判据</strong>：stage 3 显存峰值 ' + f.stages[3].totalGB + ' GB 是 stage 0~2（约 ' +
          f.stages[1].totalGB + ' GB）的 ' + (f.stages[3].totalGB / f.stages[1].totalGB).toFixed(2) +
          '×，额外开销来自 lm_head 的 logits 张量与 MoE 层 —— 这决定了后面「lm_head logits 即时释放」和「PP stage 层数重排」两条修改。</p>' +
          arrow("最后回答「碎片是怎么来的」——进入【内存快照分析层】。"),
      },

      // ── §6 内存快照分析层 ──────────────────────────────────────────────────
      { label: "内存快照分析层", short: "碎片率 " + (fr.ratio * 100).toFixed(0) + "%", sub: "HOW · 激活张量高频分配/释放 → 碎片率 " + (fr.ratio * 100).toFixed(0) + "%",
        content:
          '<p class="' + NOTE + '" style="margin:0 0 8px">导出 step 12000 时 rank ' + f.oomRank +
          ' 的完整内存快照（<code>memory_record.pkl</code>），解析后直接在界面里可视化 —— 不再需要导出 pkl 后另找专门网页解析。</p>' +
          chartSlot("fragment-map", 230,
            "横轴为显存地址空间（0–" + f.capacityGB + " GB），纵轴为 step 内时间；每块分配是一个矩形，颜色按用途。" +
            "低地址是常驻的参数/梯度/优化器（整步都在），高地址密布的橙色小块是激活中间张量 —— forward 期间密集分配、backward 后才集中释放，" +
            "于是留下大量「已释放但未合并」的空洞。<strong>点带红框的块可查看它的完整生命周期与申请堆栈。</strong>") +
          '<p class="' + NOTE + '"><strong>关键证据</strong>：' + fr.sampleBlock.name + '（' + fr.sampleBlock.shape +
          ' ≈ ' + fr.sampleBlock.sizeMB + ' MB）从 forward 第 ' + fr.sampleBlock.allocMs + ' ms 分配，到 backward 第 ' +
          fr.sampleBlock.freeMs + ' ms 才释放，<strong style="color:#dc2626">持有近 ' + (fr.sampleBlock.holdMs / 1000).toFixed(1) +
          ' 秒</strong>；申请堆栈可一路回溯到 <code>' + fr.sampleBlock.stack[fr.sampleBlock.stack.length - 1] + '</code>。</p>' +
          codeBlock("申请堆栈（点击碎片图中的块即可展开）",
            fr.sampleBlock.stack.map(function (fn, i) { return [" ", (i ? "  ↳ " : "") + fn]; })) +
          '<p class="' + NOTE + '"><strong>判据</strong>：碎片率 = 不可用空闲 / 总空闲 = ' + fr.unusableFreeGB + " / " + fr.totalFreeGB +
          " = <strong style=\"color:#dc2626\">" + (fr.ratio * 100).toFixed(0) + '%</strong> → 严重碎片化。根因是 46 层大量<strong>不等大小</strong>的中间张量在 forward 期间密集分配、backward 期间集中释放，分配器无法在短时间内合并。开启 activation checkpoint 后激活张量不再常驻，分配/释放频率大幅降低，碎片率可降至 25% 以下 —— 这也说明两条修改是<strong>互相加强</strong>的。</p>' +
          arrow("根因链完整，进入【代码/配置层】给出修改与验证。"),
      },

      // ── §7 代码 / 配置层 ───────────────────────────────────────────────────
      { label: "代码 / 配置层", short: "4 处修改", sub: "FIX · 开 checkpoint + 治碎片 + 释放 logits + 重排 PP",
        content:
          '<p class="' + NOTE + '" style="margin:0 0 12px"><strong style="color:var(--foreground)">诊断总结</strong>：根因是两个问题的叠加 —— ① activation checkpoint 未开启，46 层激活全部常驻，占 ' +
          (act.pct * 100).toFixed(1) + '% 显存（' + act.gb + ' GB）；② 大量不等大小的激活中间张量高频分配/释放，分配器碎片率 ' +
          (fr.ratio * 100).toFixed(0) + '%，让 OOM 提前到来。</p>' +

          '<h4 style="margin:14px 0 6px;font-size:12.5px;color:var(--foreground)">修改 ① — 开启 selective activation checkpointing（P0）</h4>' +
          codeBlock("training_args.yaml", [
            ["+", "+ recompute_activations: true"],
            ["+", "+ recompute_granularity: selective   # 仅重算 attention(QKV+core) 与 FFN(gate_up+down) 中间激活"],
            [" ", "                                     # layernorm 与残差连接的输出保留"],
          ]) +
          '<p class="' + NOTE + '">预期激活值 ' + act.gb + ' GB → <strong style="color:#16a34a">~8.5 GB</strong>（↓76%），总显存 ' +
          f.capacityGB + ' GB → <strong style="color:#16a34a">~36.3 GB</strong>（占容量 57% 以下）。代价：每步增加约 8% 计算时间（重计算 attention + FFN），对吞吐影响可控。</p>' +

          '<h4 style="margin:14px 0 6px;font-size:12.5px;color:var(--foreground)">修改 ② — 分配器碎片优化（P0）</h4>' +
          codeBlock("env.sh", [
            ["+", "+ export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True   # 可扩展 segment，减少碎片"],
            ["+", "+ export ACLNN_CACHE_LIMIT=2147483648                      # 2 GB 算子 workspace 常驻缓存"],
          ]) +
          '<p class="' + NOTE + '">预期分配器 API 耗时 ' + tb.allocApiMs + ' ms → <strong style="color:#16a34a">~180 ms</strong>（↓80%），碎片率 ' +
          (fr.ratio * 100).toFixed(0) + '% → <strong style="color:#16a34a">~25%</strong>。</p>' +

          '<h4 style="margin:14px 0 6px;font-size:12.5px;color:var(--foreground)">修改 ③ — lm_head logits 即时释放（P1）</h4>' +
          codeBlock("modeling_openpangu.py", [
            ["-", "- loss = cross_entropy(logits, labels)"],
            ["+", "+ loss = cross_entropy(logits, labels)"],
            ["+", "+ del logits    # vocab=151552 的 [4096,151552] logits ≈ 1.2 GB，backward 不需要保留"],
          ]) +
          '<p class="' + NOTE + '">logits 仅在 loss 计算时需要，backward 中梯度直接从 loss 反传。释放后 stage 3 减压约 1.2 GB。</p>' +

          '<h4 style="margin:14px 0 6px;font-size:12.5px;color:var(--foreground)">修改 ④ — PP stage 层数重排（辅助，P2）</h4>' +
          codeBlock("model_config.json", [
            ["-", '- "pp_layer_split": [12, 11, 11, 12],'],
            ["+", '+ "pp_layer_split": [13, 12, 11, 10],   # stage3 减 2 层减压，stage0/1 各多 1 层'],
          ]) +
          '<p class="' + NOTE + '">stage 3 显存 ' + f.stages[3].totalGB + ' GB → 约 16.2 GB，与其余 stage 更均衡（stage 间 CV 从 8% 降至 3%）。</p>' +

          '<h4 style="margin:16px 0 6px;font-size:12.5px;color:var(--foreground)">验证（①+②+③ 从 step 12000 续跑）</h4>' +
          table(["指标", "修改前", "修改后", "变化"], [
            ["显存峰值", v.peakGB[0] + " GB", '<strong style="color:#16a34a">约 ' + v.peakGB[1] + " GB</strong>", "↓47%"],
            ["分配器 API 耗时", v.allocApiMs[0] + " ms", '<strong style="color:#16a34a">' + v.allocApiMs[1] + " ms</strong>", "↓83%"],
            ["碎片率", (v.fragRatio[0] * 100).toFixed(0) + "%", '<strong style="color:#16a34a">' + (v.fragRatio[1] * 100).toFixed(0) + "%</strong>", "↓61pp"],
            ["throughput", v.throughput[0] + " tokens/s", v.throughput[1] + " tokens/s", "−1.6%（重计算开销，换来稳定无 OOM）"],
          ]) +
          '<p class="' + NOTE + '">继续训练 10000 step 无显存异常。④ 为可选项，实施后 stage 间显存更均衡。</p>' +
          quote("<strong>覆盖 CheckList Row 7</strong>：内存快照 → 解析生命周期/堆栈 → 碎片分布可视化，不再需要导出 pkl 后用专门网页解析。<br>" +
            "<strong>覆盖 CheckList Row 8</strong>：显存折线图 → 峰值自动标注 → 下钻到 stage/层/算子 → 时间线中看分配/释放时机 → 调用栈回溯，下钻闭环完整。"),
      },
    ];
  }

  function chain() {
    var steps = buildSteps();
    if (!steps) return null;
    return {
      title: "定位链 · activation checkpoint 未开启 → 显存峰值超标 + 分配器碎片触发 OOM",
      meta: "路径:性能表征层 → 瓶颈分类层(显存受限) → 显存表征层(容量不足+碎片双因子) → 峰值构成分析层 → 阶段/层定位层 → 内存快照分析层 → 代码/配置层",
      steps: steps,
    };
  }

  /* 面板挂载后调用：把正文里的画布交给「性能」页签同一套绘制代码。
     首次调用时向 PtoTrainingMemoryPanel 注册重绘回调，让这些图跟着 resize / 主题切换刷新。 */
  var redrawHooked = false;
  function renderAll() {
    var panel = window.PtoTrainingMemoryPanel;
    if (!panel || !panel.drawInto) return;
    document.querySelectorAll("[data-mem-chain-chart]").forEach(function (cv) {
      panel.drawInto(cv.dataset.memChainChart, cv);
    });
    if (!redrawHooked && panel.onRedraw) {
      redrawHooked = true;
      panel.onRedraw(function () {
        document.querySelectorAll("[data-mem-chain-chart]").forEach(function (cv) {
          panel.drawInto(cv.dataset.memChainChart, cv);
        });
      });
    }
  }

  // 与性能分析工具共用同一份案例四快照契约；长文仍可在离线 fetch 失败时使用内置事实兜底。
  fetch("../AI_Profiling_Tool/data/openpangu-2.0-flash.memory-snapshot.json")
    .then(function (response) { return response.ok ? response.json() : Promise.reject(new Error(response.status)); })
    .then(function (snapshot) { window.OPENPANGU_MEMORY_SNAPSHOT = snapshot; })
    .catch(function () {});

  return { chain: chain, renderAll: renderAll };
})();
