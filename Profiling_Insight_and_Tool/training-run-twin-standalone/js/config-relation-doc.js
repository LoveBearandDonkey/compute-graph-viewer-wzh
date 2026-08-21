/* ══ 文档视图（顶栏第三档「文档」）═════════════════════════════════════════
   正文是**静态 HTML**，写在 config-relation-observer.html 的 .cro-region--doc
   里，不由本文件拼字符串 —— 它是一篇要逐字打磨的散文，塞进 JS 模板串之后每改
   一个字都要在转义里找位置，得不偿失（这一点和 YAML 视图正相反：那份 yaml 的
   每一行都随当前配置变，只能生成）。

   本文件只做四件正文里写不了的事：
     1. 生成左侧两级目录 —— 一级取 .cro-doc__section[id]，二级取该节内的
        .cro-doc__term[id]（「配置项逐条详解」那一章有 14 条，二级目录是为了
        能直接跳到某个配置项，不用先跳到章首再翻）
     2. 点目录滚到对应位置
     3. 滚动时高亮当前位置（章与词条各高亮一处）
     4. 画联动图的贝塞尔连线（坐标只能实测，写不进静态 HTML）

   视图切换（哪一档显示哪些区）由 js/config-relation-yaml.js 的 setup() 统管，
   它是 #croViewTabs 的唯一监听方；本文件不碰页签。
   ══════════════════════════════════════════════════════════════════════════ */
(function (global) {
  "use strict";

  const doc = global.document;
  const NS = "http://www.w3.org/2000/svg";

  /* ══ 联动图连线 ═════════════════════════════════════════════════════════
     算法整段照搬事件详情「计算血缘」页签的 paintIncidentLineageEdges（见
     js/config-relation-observer.js）：量两端盒子的 getBoundingClientRect，
     换算到 shell 自身坐标系，三次贝塞尔 M→C，控制点各自水平外推 (Δx)/2，
     关系名摆在两端中点上方 5px。那边是「算子在五层之间怎么变换」，这边是
     「配置项之间怎么牵动」，同一种图，本该同一套画法。

     没有照抄的只有两点：那边的边有 is-active / is-muted 两态（跟着选中联动），
     这边是静态文档，边不需要状态；那边的 label 追加在所有 path 之后以压住
     连线，这边同理但只有四条边，直接分两趟 append。 */
  const FLOWS = ["up", "down", "must", "often"];

  /* 箭头 marker 只建一份：marker 的 id 是**文档级**的，跨 <svg> 引用完全合法，
     所以 14 张图共 4 枚，而不是每张图各建 4 枚。
     orient="auto" 让箭头跟着曲线末端的切线转 —— 四条边都是左→右，出来就都是
     朝右的箭头，不用按方向分别建两套。refX 取到接近视窗右缘（7.4/8），箭尖正好
     落在路径终点，也就是目标盒的左缘上。 */
  function ensureArrowDefs(host) {
    if (doc.getElementById("cro-doc-arrow-up")) return;
    const svg = doc.createElementNS(NS, "svg");
    svg.setAttribute("class", "cro-doc__map-defs");
    svg.setAttribute("aria-hidden", "true");
    const defs = doc.createElementNS(NS, "defs");
    FLOWS.forEach((flow) => {
      const marker = doc.createElementNS(NS, "marker");
      marker.setAttribute("id", `cro-doc-arrow-${flow}`);
      marker.setAttribute("viewBox", "0 0 8 8");
      marker.setAttribute("refX", "7.4");
      marker.setAttribute("refY", "4");
      marker.setAttribute("markerWidth", "8");
      marker.setAttribute("markerHeight", "8");
      // userSpaceOnUse 而不是默认的 strokeWidth：线宽只有 1.25，按线宽缩放出来的
      // 箭头会小到看不清，而这枚箭头正是要读的那个信息。
      marker.setAttribute("markerUnits", "userSpaceOnUse");
      marker.setAttribute("orient", "auto");
      const tip = doc.createElementNS(NS, "path");
      tip.setAttribute("class", "cro-doc__map-arrow");
      tip.setAttribute("data-flow", flow);
      tip.setAttribute("d", "M0 0.6 L8 4 L0 7.4 Z");
      marker.appendChild(tip);
      defs.appendChild(marker);
    });
    svg.appendChild(defs);
    host.appendChild(svg);
  }

  function paintMap(map) {
    const node = map.querySelector(".cro-doc__map-node");
    if (!node) return;
    /* client 而不是 offset：.cro-doc__map 有 1px 描边，而 SVG 的 inset:0 是从
       描边**内侧**（padding box）起算的。用 offsetWidth 会让 viewBox 比实际绘图
       区大 2px，整幅连线跟着偏半格 —— 血缘那边的 shell 没有描边，所以原版直接
       用了 offset。下面的原点也要相应加上 clientLeft/clientTop（= 描边宽度）。 */
    const width = map.clientWidth;
    const height = map.clientHeight;
    if (!width || !height) return;   // 本档没显示，尺寸全是 0

    let svg = map.querySelector(".cro-doc__map-edges");
    if (!svg) {
      svg = doc.createElementNS(NS, "svg");
      svg.setAttribute("class", "cro-doc__map-edges");
      svg.setAttribute("aria-hidden", "true");
      map.insertBefore(svg, map.firstChild);
    }
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    const shell = map.getBoundingClientRect();
    // 有祖先做过 transform 缩放时 rect 会和布局尺寸对不上，这两个比值把它折回来
    const scaleX = shell.width / (map.offsetWidth || 1) || 1;
    const scaleY = shell.height / (map.offsetHeight || 1) || 1;
    // 原点落在描边内侧，与 SVG 的 inset:0 对齐
    const originX = shell.left + map.clientLeft * scaleX;
    const originY = shell.top + map.clientTop * scaleY;
    const n = node.getBoundingClientRect();
    const nodeMidY = (n.top + n.height / 2 - originY) / scaleY;

    const labels = [];
    map.querySelectorAll(".cro-doc__map-box").forEach((box) => {
      const flow = box.dataset.flow || "";
      const title = box.querySelector(".cro-doc__map-box-title");
      const b = box.getBoundingClientRect();
      const boxMidY = (b.top + b.height / 2 - originY) / scaleY;
      /* 左列的两个盒子是「因」（盒子 → 中心），右列的两个是「果」（中心 → 盒子）。
         按盒子中心在中心节点的哪一侧判，而不是按 data-flow 硬编码 —— 万一以后
         调了格位，方向自己就跟对了。 */
      const isCause = b.left + b.width / 2 < n.left + n.width / 2;
      /* 两端各留一点空隙：起点离源盒 2px、终点离目标盒 3px（箭尖就落在这 3px 上），
         否则线会贴死在描边上、箭头也压在边框里看不出是个箭头。
         四条边都是左→右，所以两端的偏移方向对因/果两类是一样的。 */
      const x1 = ((isCause ? b.right : n.right) - originX) / scaleX + 2;
      const y1 = isCause ? boxMidY : nodeMidY;
      const x2 = ((isCause ? n.left : b.left) - originX) / scaleX - 3;
      const y2 = isCause ? nodeMidY : boxMidY;
      const bend = Math.max(12, (x2 - x1) * 0.5);

      const path = doc.createElementNS(NS, "path");
      path.setAttribute("class", "cro-doc__map-edge");
      path.setAttribute("data-flow", flow);
      path.setAttribute("d", `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
      if (FLOWS.includes(flow)) path.setAttribute("marker-end", `url(#cro-doc-arrow-${flow})`);
      svg.appendChild(path);

      if (!title) return;
      const label = doc.createElementNS(NS, "text");
      label.setAttribute("class", "cro-doc__map-edge-label");
      label.setAttribute("data-flow", flow);
      label.setAttribute("x", String((x1 + x2) / 2));
      label.setAttribute("y", String((y1 + y2) / 2 - 5));
      label.setAttribute("text-anchor", "middle");
      label.textContent = title.textContent.trim();
      labels.push(label);
    });
    // 字压线：四条边先铺完，标签整体后置
    labels.forEach((label) => svg.appendChild(label));
  }

  function setup() {
    const body = doc.getElementById("croDocBody");
    const list = doc.getElementById("croDocTocList");
    const toc = list && list.closest(".cro-doc__toc");
    const board = doc.getElementById("croBoard");
    if (!body || !list) return;

    const sections = Array.from(body.querySelectorAll(".cro-doc__section[id]"));
    if (!sections.length) return;

    /* ── 建目录 ──
       文案取自正文里的 .cro-doc__h / .cro-doc__term-name，不另写一份：改标题
       只改正文那一处。一级的序号由 CSS counter 给（见 config-relation-doc.css），
       这里不写死数字；二级不编号 —— 配置项之间没有先后关系，编号只会误导。

       anchors 是**扁平**的一张表（章与词条按文档顺序混排），滚动高亮只需要在
       这一张表上找「最后一个已越过阅读线的锚点」，不必分两层各找一次。 */
    const anchors = [];

    const mkLink = (cls, targetId, text) => {
      const link = doc.createElement("button");
      link.type = "button";
      link.className = cls;
      link.dataset.target = targetId;
      link.textContent = text;
      return link;
    };

    sections.forEach((section) => {
      const item = doc.createElement("li");
      const title = section.querySelector(".cro-doc__h");
      const link = mkLink("cro-doc__toc-link", section.id,
        title ? title.textContent.trim() : section.id);
      item.appendChild(link);
      anchors.push({ el: section, link, chapter: link });

      const terms = Array.from(section.querySelectorAll(".cro-doc__term[id]"));
      if (terms.length) {
        const sub = doc.createElement("ul");
        sub.className = "cro-doc__toc-sub";
        terms.forEach((term) => {
          const name = term.querySelector(".cro-doc__term-name");
          const subItem = doc.createElement("li");
          const subLink = mkLink("cro-doc__toc-sublink", term.id,
            name ? name.textContent.trim() : term.id);
          subItem.appendChild(subLink);
          sub.appendChild(subItem);
          // chapter 记的是它所属的那一章：词条高亮时，章也要跟着标出来
          anchors.push({ el: term, link: subLink, chapter: link });
        });
        item.appendChild(sub);
      }

      list.appendChild(item);
    });

    const links = anchors.map((a) => a.link);
    let activeEl = null;

    const setActive = (entry, scrollToc) => {
      if (!entry || entry.el === activeEl) return;
      activeEl = entry.el;
      links.forEach((link) => link.classList.remove("is-active"));
      list.querySelectorAll(".cro-doc__toc-link").forEach((link) => {
        link.classList.remove("is-within");
      });
      entry.link.classList.add("is-active");
      // 停在某个词条上时，它所属的那一章标成「正在这一章里」——比章也涂成
      // is-active 弱一档，两处高亮才分得出「你在哪一章」和「你在哪一条」。
      if (entry.chapter !== entry.link) entry.chapter.classList.add("is-within");

      /* 目录本身可能已经滚出去了（23 项，窄屏放不下）。只在滚动驱动时补位，
         点击驱动时不动 —— 用户刚点的那一项就在他手指底下，再挪一次是打扰。 */
      if (scrollToc && toc && toc.scrollHeight > toc.clientHeight) {
        const top = entry.link.offsetTop;
        const bottom = top + entry.link.offsetHeight;
        if (top < toc.scrollTop) toc.scrollTop = top - 8;
        else if (bottom > toc.scrollTop + toc.clientHeight) {
          toc.scrollTop = bottom - toc.clientHeight + 8;
        }
      }
    };

    /* ── 当前位置 ──
       不用 IntersectionObserver：文档区在非 doc 档是 display:none，那时所有
       entry 都是 not-intersecting，切回来会先闪一下「没有选中项」。直接按
       滚动位置算反而稳 —— 取最后一个顶边已经越过阅读线的锚点。
       阅读线放在容器上沿往下 1/4 处：正好是眼睛落点，比用 0（顶边）更早换档、
       读起来跟手。 */
    const sync = () => {
      if (!body.offsetParent && body.offsetHeight === 0) return;   // 本档没显示，别算
      const line = body.scrollTop + body.clientHeight * 0.25;
      let current = anchors[0];
      anchors.forEach((entry) => {
        if (entry.el.offsetTop <= line) current = entry;
      });
      // 已经滚到底：末项可能永远够不到阅读线（它比 3/4 屏矮），直接判它
      if (body.scrollTop + body.clientHeight >= body.scrollHeight - 4) {
        current = anchors[anchors.length - 1];
      }
      setActive(current, true);
    };

    list.addEventListener("click", (event) => {
      const link = event.target.closest("[data-target]");
      if (!link) return;
      const target = doc.getElementById(link.dataset.target);
      if (!target) return;
      /* scrollIntoView 会连带滚外层的 .cro-board，这里只想滚正文那一栏，
         所以自己算偏移量。offsetTop 是相对 .cro-doc__body 的（它是定位祖先，
         见 CSS 里它自身的 position:relative）。 */
      body.scrollTo({ top: Math.max(0, target.offsetTop - 8), behavior: "smooth" });
      setActive(anchors.find((a) => a.el === target), false);
    });

    body.addEventListener("scroll", () => {
      global.requestAnimationFrame(sync);
    }, { passive: true });

    /* ── 联动图 ── */
    const maps = Array.from(body.querySelectorAll(".cro-doc__map"));
    if (maps.length) ensureArrowDefs(body);
    const paintMaps = () => maps.forEach(paintMap);

    /* 逐图观察自身尺寸而不是只听 window.resize：格子里的条目会随字体载入、
       中英文换行阈值变化而改变高度，那时窗口尺寸没动，但连线的两端已经挪了。
       画 SVG 不改变 map 自身尺寸（连线层是绝对定位的），不会自激。 */
    if (global.ResizeObserver && maps.length) {
      const ro = new global.ResizeObserver((entries) => {
        entries.forEach((entry) => paintMap(entry.target));
      });
      maps.forEach((map) => ro.observe(map));
    } else {
      global.addEventListener("resize", () => {
        global.requestAnimationFrame(paintMaps);
      });
    }

    /* 切到文档档的那一刻要对一次：本档此前 display:none，尺寸全是 0，
       sync 与 paintMaps 都会被挡回去。.cro-board 的 class 一翻就重算。 */
    if (board) {
      new MutationObserver(() => {
        if (!board.classList.contains("is-doc")) return;
        global.requestAnimationFrame(() => { sync(); paintMaps(); });
      }).observe(board, { attributes: true, attributeFilter: ["class"] });
    }

    setActive(anchors[0], false);
    paintMaps();
  }

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", setup);
  else setup();
})(window);
