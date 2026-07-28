(function (global) {
  "use strict";

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/"/g, "&quot;");
  }

  function kind(v) {
    var k = String(v || "").toLowerCase();
    if (k === "temp") return "workspace";
    if (k === "resident") return "parameters";
    return k || "workspace";
  }

  function label(k) {
    return ({
      activation: "激活",
      parameters: "参数",
      gradients: "梯度",
      optimizer: "优化器",
      workspace: "临时空间"
    })[k] || k;
  }

  function fact(name, value, danger) {
    return '<div class="pto-hbm-snapshot__fact' + (danger ? " is-danger" : "") + '"><span>' +
      name + "</span><strong>" + esc(value) + "</strong></div>";
  }

  function render(container, snapshot, options) {
    options = options || {};
    var s = snapshot.summary || {};
    var rows = snapshot.lifetimes || [];
    var selected = options.initialSelectedId || (rows[0] && rows[0].id);
    var maxT = Math.max.apply(Math, [1].concat(rows.map(function (x) { return Number(x.freeMs) || 0; })));
    var fragments = snapshot.fragmentationMap || [];
    var fragmentTotal = fragments.reduce(function (sum, x) { return sum + Number(x.sizeGB || 0); }, 0) || 1;
    var requestFragmentWidth = (Number(s.requestedGB) || 0) / fragmentTotal * 100;
    var requestFitWidth = Math.min(100, (Number(s.largestFreeBlockGB) || 0) / (Number(s.requestedGB) || 1) * 100);
    var shortageGB = Math.max(0, (Number(s.requestedGB) || 0) - (Number(s.largestFreeBlockGB) || 0));
    var fragmentWindow = snapshot.fragmentationWindow || {};
    var fragmentStart = Number(fragmentWindow.startGB || 0);
    var fragmentEnd = fragmentStart + fragmentTotal;
    var fragmentLeft = fragmentStart / Number(s.capacityGB || 1) * 100;
    var fragmentRight = fragmentEnd / Number(s.capacityGB || 1) * 100;
    var fragmentTargetLeft = 100 / 6;
    var fragmentTargetRight = 100 - fragmentTargetLeft;
    var fragmentRows = (snapshot.fragmentAllocations || []).map(function (x) {
      var copy = {};
      Object.keys(x).forEach(function (key) { copy[key] = x[key]; });
      return copy;
    });
    var fragmentCursor = fragmentStart;
    fragments.forEach(function (segment) {
      if (segment.kind === "blocker") {
        var allocation = fragmentRows.find(function (x) { return x.id === segment.allocationId; });
        if (allocation) {
          allocation.offsetGB = fragmentCursor;
          allocation.sizeGB = Number(segment.sizeGB || 0);
        }
      }
      fragmentCursor += Number(segment.sizeGB || 0);
    });
    var detailRows = rows.concat(fragmentRows);
    container.innerHTML =
      '<section class="pto-hbm-snapshot">' +
        '<div class="pto-hbm-snapshot__context"><div><b>问题2 · 显存 OOM</b><span>' +
          esc(snapshot.model || "训练任务") + '</span></div><div><strong>rank ' + esc(snapshot.rank) +
          '</strong><span>OOM 问题卡</span></div><div><strong>step ' + esc(snapshot.step) +
          '</strong><span>失败前快照 · step ' + esc(snapshot.incidentStep) +
          ' 中断</span></div><p>本视图内的整卡地址、局部放大和生命周期均来自 rank ' +
          esc(snapshot.rank) + ' 的同一份快照，不是集群汇总或其他 Rank 的示意图。</p></div>' +
        '<div class="pto-hbm-snapshot__evidence">' +
          fact("HBM 峰值", (s.peakGB || 0) + "/" + (s.capacityGB || 0) + " GB", true) +
          fact("空闲总量", (s.freeGB || 0) + " GB") +
          fact("最大连续空闲块", (s.largestFreeBlockGB || 0) + " GB") +
          fact("本次请求", (s.requestedGB || 0) + " GB") +
        "</div>" +
        '<div class="pto-hbm-snapshot__verdict"><b>为什么 OOM？</b>空闲总量足够，但最大连续块 ' +
          esc(s.largestFreeBlockGB) + " GB 小于请求 " + esc(s.requestedGB) +
          " GB；碎片率 " + esc(s.fragmentationRatio) + "%。</div>" +
        '<div class="pto-hbm-snapshot__body"><div class="pto-hbm-snapshot__plots">' +
          '<section class="pto-hbm-snapshot__section"><div class="pto-hbm-snapshot__title">rank ' +
            esc(snapshot.rank) + ' · 整卡 64 GB 地址空间与关键分配 ' +
            '<span class="pto-hbm-snapshot__hint">step ' + esc(snapshot.step) +
            ' 快照 · 斜纹底色=其他已占用 · 彩色色块=关键分配 · 纯灰=真实空闲段</span></div>' +
            '<div class="pto-hbm-snapshot__legend">' +
              ["activation", "parameters", "gradients", "optimizer", "workspace"].map(function (k) {
                return '<span><i style="background:var(--hbm-' + k + ')"></i>' + label(k) + "</span>";
              }).join("") +
              '<span><i class="is-other"></i>其他已占用（未展开）</span><span><i class="is-gap"></i>空闲碎片（合计 ' +
              esc(s.freeGB) + " GB）</span></div>" +
            '<div class="pto-hbm-snapshot__axis pto-hbm-snapshot__address-axis"><span>0 GB</span><span>' +
              esc(s.capacityGB / 2) + " GB</span><span>" + esc(s.capacityGB) + " GB</span></div>" +
            '<div class="pto-hbm-snapshot__address-wrap"><div class="pto-hbm-snapshot__address">' +
              rows.map(function (x) {
                return '<button class="pto-hbm-snapshot__address-block" data-id="' + esc(x.id) +
                  '" data-kind="' + kind(x.kind) + '" style="left:' + (x.offsetGB / s.capacityGB * 100) +
                  "%;width:" + (x.sizeGB / s.capacityGB * 100) + '%" title="' + esc(x.name) +
                  " · " + esc(x.sizeGB) + ' GB"></button>';
              }).join("") +
              '<span class="pto-hbm-snapshot__zoom-source" style="left:' + fragmentLeft + '%;width:' +
                (fragmentRight - fragmentLeft) + '%" title="' + esc(fragmentWindow.label || "碎片放大区域") +
                " · " + fragmentStart.toFixed(2) + "–" + fragmentEnd.toFixed(2) + ' GB">' +
                fragments.map(function (x) {
                  var miniAllocation = fragmentRows.find(function (row) { return row.id === x.allocationId; });
                  return '<i class="' + (x.kind === "free" ? "is-free" : "is-occupied") +
                    '" data-kind="' + kind(miniAllocation && miniAllocation.kind) +
                    '" style="width:' + (Number(x.sizeGB || 0) / fragmentTotal * 100) + '%"></i>';
                }).join("") + "</span>" +
            "</div></div>" +
            '<div class="pto-hbm-snapshot__zoom-bridge" aria-label="总览框选区域展开为下方局部地址图"><span>rank ' +
              esc(snapshot.rank) + ' · ' + fragmentStart.toFixed(2) + "–" + fragmentEnd.toFixed(2) +
              ' GB 空闲区域放大</span><svg viewBox="0 0 100 90" preserveAspectRatio="none">' +
              '<defs><linearGradient id="hbmZoomFlow" x1="0" y1="0" x2="0" y2="1">' +
              '<stop offset="0" stop-color="currentColor" stop-opacity=".18"></stop>' +
              '<stop offset=".52" stop-color="currentColor" stop-opacity=".10"></stop>' +
              '<stop offset="1" stop-color="currentColor" stop-opacity=".25"></stop></linearGradient></defs>' +
              '<path d="M ' + fragmentLeft + ' 0 C ' + fragmentLeft + ' 32, ' +
              fragmentTargetLeft + ' 58, ' + fragmentTargetLeft + ' 90 H ' +
              fragmentTargetRight + ' C ' + fragmentTargetRight + ' 58, ' +
              fragmentRight + ' 32, ' + fragmentRight +
              ' 0 Z" fill="url(#hbmZoomFlow)"></path></svg></div>' +
            '<div class="pto-hbm-snapshot__fragment-map" aria-label="空闲碎片地址分布">' +
              fragments.map(function (x, index) {
                var width = Number(x.sizeGB || 0) / fragmentTotal * 100;
                if (x.kind === "free") {
                  var largest = Number(x.sizeGB) === Number(s.largestFreeBlockGB);
                  return '<span class="is-free' + (largest ? " is-largest" : "") + '" style="width:' + width +
                    '%" title="空闲 ' + esc(x.sizeGB) + ' GB"><b>' + esc(x.sizeGB) + "G</b></span>";
                }
                var allocation = fragmentRows.find(function (row) { return row.id === x.allocationId; });
                return '<button class="is-blocker pto-hbm-snapshot__fragment-allocation" data-id="' +
                  esc(x.allocationId) + '" data-kind="' + kind(allocation && allocation.kind) +
                  '" style="width:' + width + '%" title="' +
                  esc(allocation ? allocation.name : "活跃分配") + " · " + esc(x.sizeGB) +
                  ' GB · 点击查看详情"></button>';
              }).join("") +
            '</div><div class="pto-hbm-snapshot__request-attempt"><div class="pto-hbm-snapshot__request-scale"><i style="width:' +
              requestFragmentWidth.toFixed(3) + '%"><em style="width:' + requestFitWidth.toFixed(3) +
              '%"></em><strong style="width:' + (100 - requestFitWidth).toFixed(3) +
              '%"></strong></i></div><div class="pto-hbm-snapshot__request-copy"><span>待分配 ' +
              esc(s.requestedGB) + ' GB（与最大空洞左对齐）</span><b>超出 ' +
              shortageGB.toFixed(1) + ' GB</b></div></div>' +
            '<div class="pto-hbm-snapshot__fragment-caption"><b>最大空洞只有 ' +
              esc(s.largestFreeBlockGB) + " GB</b><span>总空闲 " + esc(s.freeGB) +
              " GB 分散在 " + fragments.filter(function (x) { return x.kind === "free"; }).length +
              " 个不连续地址段，不能拼接成 0.5 GB。</span></div>" +
          '</section></div><aside class="pto-hbm-snapshot__detail"></aside></div>' +
      "</section>";

    var root = container.firstElementChild;
    var detail = root.querySelector(".pto-hbm-snapshot__detail");

    function select(id) {
      selected = id;
      root.querySelectorAll("[data-id]").forEach(function (el) {
        el.classList.toggle("is-selected", el.dataset.id === id);
      });
      var x = detailRows.find(function (v) { return String(v.id) === String(id); }) || detailRows[0];
      if (!x) return;
      var duration = x.freeMs - x.allocMs;
      detail.innerHTML = "<h3>" + esc(x.name) + '</h3><div class="pto-hbm-snapshot__detail-kind">' +
        label(kind(x.kind)) + " · " + esc(x.source || "未知来源") + (x.line ? ":" + x.line : "") +
        '</div><div class="pto-hbm-snapshot__detail-life"><div><span>生命周期</span><b>' +
        esc(duration) + ' ms</b></div><div class="pto-hbm-snapshot__detail-life-track"><i data-kind="' +
        kind(x.kind) + '" style="left:' + (x.allocMs / maxT * 100) + "%;width:" +
        (duration / maxT * 100) + '%"></i></div><div class="pto-hbm-snapshot__detail-life-axis"><span>0 ms</span><span>' +
        Math.round(maxT / 2) + " ms</span><span>" + maxT + " ms</span></div>" +
        '</div><div class="pto-hbm-snapshot__kv"><div><span>地址</span><b>' + esc(x.offsetGB) + "–" +
        esc((x.offsetGB + x.sizeGB).toFixed(3)) + " GB</b></div><div><span>大小</span><b>" +
        esc(x.sizeGB) + " GB</b></div><div><span>申请</span><b>" + esc(x.allocMs) +
        " ms</b></div><div><span>释放</span><b>" + esc(x.freeMs) +
        " ms</b></div><div><span>持有</span><b>" + esc(duration) +
        ' ms</b></div></div><div class="pto-hbm-snapshot__stack"><b>调用栈</b><br>' +
        esc(x.stack || "暂无调用栈") +
        '</div><div class="pto-hbm-snapshot__actions"><button class="btn btn-ghost btn-sm" data-action="timeline">Timeline</button>' +
        '<button class="btn btn-ghost btn-sm" data-action="source">源码</button></div>';
    }

    root.addEventListener("click", function (e) {
      var block = e.target.closest("[data-id]");
      if (block) select(block.dataset.id);
      var action = e.target.closest("[data-action]");
      if (!action) return;
      var x = detailRows.find(function (v) { return String(v.id) === String(selected); });
      if (action.dataset.action === "timeline") {
        if (options.onOpenTimeline) options.onOpenTimeline(x);
      } else if (options.onOpenSource) {
        options.onOpenSource(x);
      }
    });
    select(selected);
    return {
      select: select,
      resize: function () {},
      destroy: function () { container.innerHTML = ""; }
    };
  }

  global.PtoHbmMemorySnapshot = { render: render };
})(window);
