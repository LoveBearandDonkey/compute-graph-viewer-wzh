/**
 * 逻辑魔方 pattern 的几何回归检查（Playwright）。
 *
 * 为什么要有这个脚本：连线「没连到卡上」已经出现过两次，成因每次都不同
 *   ① 走线用样条（CatmullRom）→ 控制点之间外扩成弧，线从卡旁边绕过去；
 *   ② 飞行动画按帧固定系数缓动 → 4000 卡时帧率掉到 10fps 上下，cur 迟迟到不了
 *      target，卡与线一起停在半路（看上去就是线没连到卡）；
 *   ③ 管用 CurvePath(LineCurve3…) 建 → TubeGeometry 按弧长等距采样，采样点落不到
 *      折线拐点上，线从卡旁边抄近路（端点仍然精确，只测端点查不出来）。
 * 这类问题肉眼很难在每次改动后复查，所以固化成断言：
 *   · 每条通信线的**每个折线顶点**都必须落在对应 rank 的卡心（不只端点——曲线若按弧长
 *     参数化，中间拐点会被抄近路跳过，只测端点看不出来，这正是第三次复发的成因）；
 *   · 重排动画必须在 1.5s 内收敛（cur 精确吸附到 target）。
 *
 * 用法：
 *   1) 装一次 Playwright：`npm i -D playwright`（容器里若已预装 Chromium，
 *      设 PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1，并用 CHROMIUM=/opt/pw-browsers/chromium 指路）；
 *   2) 起静态服务器：`python3 -m http.server 8178 --directory public`；
 *   3) `node scripts/verify-rubik-cube.mjs [url]`。退出码非 0 表示有回归。
 */
let chromium;
try { ({ chromium } = await import('playwright')); }
catch (e) {
  console.error('需要 Playwright：npm i -D playwright（或把它装在能被 node 解析到的位置）');
  process.exit(2);
}

const URL = process.argv[2] || 'http://localhost:8178/rubik-pattern.html';
const CONFIGS = [
  { tp: 8, pp: 5, dp: 100, ep: 2 },     // 盘古 Pro MoE 真实策略（4000 卡）
  { tp: 2, pp: 4, dp: 16, ep: 8 },      // 128 卡小规格
];
const TOL = 0.12;                        // 管半径 ~0.06，留一倍余量
const SETTLE_MS = 1500;

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || '/opt/pw-browsers/chromium' });
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(URL, { waitUntil: 'networkidle' });
await page.waitForTimeout(2500);
await page.evaluate(() => window.rubik.setPlaying(false));

const fails = [];
for (const cfg of CONFIGS) {
  await page.evaluate((c) => window.rubik.setConfig(c), cfg);
  await page.waitForTimeout(SETTLE_MS);
  for (const mode of [0, 1, 2, 3, 4]) {
    await page.evaluate((m) => { window.rubik.setMode(m); window.rubik.select(37); }, mode);
    await page.waitForTimeout(SETTLE_MS);
    const r = await page.evaluate(() => {
      const h = window.rubik, m = h.model, v = { x: 0, y: 0, z: 0 };
      const P = (rank) => { m.posOf(rank, h.state.mode, v); return [v.x, v.y, v.z]; };
      let worst = 0, lines = 0, worstDim = '', worstAt = '';
      h.scene.traverse((o) => {
        const e = o.userData && o.userData.edge;
        if (!e || !e.ranks) return;
        lines++;
        const pos = o.geometry.attributes.position;
        const RAD = 6;                                  // radialSegments 5 → 每环 6 个顶点
        const rings = pos.count / RAD - 1;              // TubeGeometry: (tubular+1) 环
        const perSeg = rings / (e.ranks.length - 1);    // 每段几环（commLine 里取 3）
        // 逐个折线顶点检查：不只端点——曲线若按弧长参数化，中间的拐点会被"抄近路"跳过，
        // 只测端点是看不出来的（这正是第三次「线没连到卡上」的成因）。
        for (let k = 0; k < e.ranks.length; k++) {
          const ring = Math.round(k * perSeg);
          const base = ring * RAD;
          if (base + RAD > pos.count) continue;
          const c = [0, 0, 0];
          for (let i = 0; i < RAD; i++) { c[0] += pos.getX(base + i); c[1] += pos.getY(base + i); c[2] += pos.getZ(base + i); }
          c[0] /= RAD; c[1] /= RAD; c[2] /= RAD;
          const q = P(e.ranks[k]);
          const d = Math.hypot(c[0] - q[0], c[1] - q[1], c[2] - q[2]);
          if (d > worst) { worst = d; worstDim = e.dim; worstAt = k + '/' + (e.ranks.length - 1); }
        }
      });
      return { lines, worst: +worst.toFixed(4), worstDim, worstAt };
    });
    const ok = r.lines > 0 && r.worst <= TOL;
    console.log(`${ok ? 'ok  ' : 'FAIL'} cfg ${cfg.tp}·${cfg.pp}·${cfg.dp}·${cfg.ep} mode ${mode}: ${r.lines} 条线，折线顶点最大偏离卡心 ${r.worst}${r.worstDim ? ` (${r.worstDim} 第 ${r.worstAt} 个顶点)` : ''}`);
    if (!ok) fails.push(`cfg ${JSON.stringify(cfg)} mode ${mode} worst=${r.worst} lines=${r.lines}`);
  }
}

if (errors.length) fails.push('page errors: ' + errors.join(' | '));
await browser.close();
if (fails.length) { console.error('\n回归：\n- ' + fails.join('\n- ')); process.exit(1); }
console.log('\n全部通过：连线端点都落在卡心，重排动画在 %dms 内收敛。', SETTLE_MS);
