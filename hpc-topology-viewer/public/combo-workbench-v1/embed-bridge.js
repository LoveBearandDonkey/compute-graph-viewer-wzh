/* ══════════════════════════════════════════════════════════════════════
   swimlane.html ←→ 组合工作台 的内嵌桥。

   纪律：**不碰上游那个 IIFE 里的任何变量**。它把 setActiveRange / draw / 主题
   都关在闭包里，外面本来就够不着——但那些状态各自都有一颗现成的按钮，
   所以这里一律走 el.click()，让上游自己的事件处理器去改自己的状态。
   好处是上游怎么重构内部实现都不会把这座桥拆掉，只要按钮还在。

   支持两条入口，语义一致：
     · URL 参数    ?theme=light|dark  ?chrome=0  ?range=global|pp2|l34
       主题与顶栏必须赶在首帧之前落下，所以本文件是 <head> 里的**同步**脚本。
     · postMessage {type:'pto:state', range, theme, hl:{rank}}
       与 /parallel-topology/demo.html 同一套约定（那边也认 pto:state / 回
       pto:state-ack），宿主一份代码就能同时驱动两张卡；hl.rank 是后加的一条，
       宿主拿它来在这一格浮一枚"Rank N 在其他视图选中"的小徽标（见下方 ③b）。
   宿主拿 {type:'pto:state-ack', range} 对齐卡片头的段控。
   ══════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  var params = new URLSearchParams(location.search);

  /* ── ① 首帧之前：主题与顶栏 ───────────────────────────────────────── */
  var initTheme = params.get('theme');
  if (initTheme === 'light' || initTheme === 'dark') {
    document.documentElement.dataset.theme = initTheme;
  }
  if (params.get('chrome') === '0') {
    document.documentElement.dataset.chrome = '0';
  }

  /* ── ② DOM 就绪之后：时间范围、主题按钮文案、消息监听 ──────────────── */
  function whenReady(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else {
      fn();
    }
  }

  function rangeButton(id) {
    if (!id) return null;
    var group = document.getElementById('rangeToggle');
    if (!group) return null;
    var all = group.querySelectorAll('[data-range]');
    for (var i = 0; i < all.length; i++) {
      if (all[i].dataset.range === id) return all[i];
    }
    return null;
  }

  function activeRange() {
    var group = document.getElementById('rangeToggle');
    if (!group) return null;
    var on = group.querySelector('[data-range][aria-pressed="true"]');
    return on ? on.dataset.range : null;
  }

  /* 段控是禁用态时（Rank 前向/反向要先展开某个 Rank）静默忽略——
     宿主传了个当下点不动的范围，不该让这一格白屏或抛错。 */
  function applyRange(id) {
    var btn = rangeButton(id);
    if (!btn || btn.disabled || btn.getAttribute('aria-pressed') === 'true') return false;
    btn.click();
    return true;
  }

  /* 主题只有一颗切换按钮（没有「设为浅色」这种幂等入口），所以先比对再点， */
  /* 点完把它的文案掰回来——URL 改过主题时它显示的还是初始那一版。 */
  function applyTheme(value) {
    if (value !== 'light' && value !== 'dark') return false;
    var toggle = document.getElementById('themeToggle');
    if (!toggle) return false;
    if (document.documentElement.dataset.theme !== value) toggle.click();
    syncThemeLabel();
    return true;
  }

  function syncThemeLabel() {
    var toggle = document.getElementById('themeToggle');
    if (toggle) toggle.textContent = document.documentElement.dataset.theme === 'light' ? '深色' : '浅色';
  }

  function reportTo(target) {
    if (!target || target === window) return;
    target.postMessage({
      type: 'pto:state-ack',
      source: 'microbatch-lifecycle-swimlane',
      range: activeRange(),
      theme: document.documentElement.dataset.theme || 'dark'
    }, '*');
  }

  /* ── ③b 反向联动确认：整网图/主视图选中了一张卡 → 这一格浮一枚会自己淡出的
     小徽标「Rank N 在其他视图选中」。宿主用的是同一条 pto:state.hl 通道（跟
     驱动 demo.html 那两格高亮的 pushHL 是同一个函数、同一个消息形状），这里
     只多认一个 hl.rank 字段。

     故意不去找 Rank N 对应哪一行、把它展开或滚到可见——那要伸手进上游那个
     IIFE 的 expandedRowIds/focusedRankRowId 状态机（"纪律"里禁止碰的那些
     变量），产品原话也只是「有点反应」，没要求真的把那一行拉出来；泳道自己
     只模拟 32 个物理 Rank 里的一部分代表行，宿主也没法保证选中的 Rank 一定
     在这份 mock 里有对应行。浮标只加在 DOM 上、只读 CSS 变量取色，不碰
     上游状态，跟纪律不冲突。 */
  var rankBadgeEl = null, rankBadgeTimer = 0;
  function hideRankBadge() {
    if (rankBadgeTimer) { clearTimeout(rankBadgeTimer); rankBadgeTimer = 0; }
    if (rankBadgeEl) rankBadgeEl.style.opacity = '0';
  }
  function showRankBadge(n) {
    if (rankBadgeTimer) { clearTimeout(rankBadgeTimer); rankBadgeTimer = 0; }
    if (n === null || n === undefined) { hideRankBadge(); return; }
    if (!rankBadgeEl) {
      rankBadgeEl = document.createElement('div');
      rankBadgeEl.style.cssText = 'position:fixed;top:10px;right:12px;z-index:9999;' +
        'padding:4px 10px;border-radius:999px;font:600 12px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;' +
        'background:var(--background-elevated,#1a1f2b);color:var(--foreground,#fff);' +
        'border:1px solid var(--border-strong,rgba(255,255,255,.2));' +
        'box-shadow:0 2px 8px rgba(0,0,0,.25);pointer-events:none;opacity:0;transition:opacity .15s ease;';
      document.body.appendChild(rankBadgeEl);
    }
    rankBadgeEl.textContent = '◀ Rank ' + n + ' 在其他视图选中';
    requestAnimationFrame(function () { if (rankBadgeEl) rankBadgeEl.style.opacity = '1'; });
    rankBadgeTimer = setTimeout(hideRankBadge, 3200);
  }

  whenReady(function () {
    syncThemeLabel();
    applyRange(params.get('range'));

    window.addEventListener('message', function (ev) {
      var msg = ev && ev.data;
      if (!msg || msg.type !== 'pto:state') return;
      if (msg.range) applyRange(msg.range);
      if (msg.theme) applyTheme(msg.theme);
      if (msg.hl && Object.prototype.hasOwnProperty.call(msg.hl, 'rank')) showRankBadge(msg.hl.rank);
      reportTo(ev.source);
    });

    /* 页内自己换范围时也回报一次，宿主卡片头的段控才不会跟页面对不上 */
    var group = document.getElementById('rangeToggle');
    if (group) {
      group.addEventListener('click', function () {
        setTimeout(function () { reportTo(window.parent); }, 0);
      });
    }

    reportTo(window.parent);
  });
})();
