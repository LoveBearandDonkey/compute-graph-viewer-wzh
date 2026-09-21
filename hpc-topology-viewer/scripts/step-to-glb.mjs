#!/usr/bin/env node
/**
 * step-to-glb.mjs — convert industrial STEP/STP CAD to GLB, fully in Node
 * (no FreeCAD / Blender). Parses the STEP solids with occt-import-js
 * (OpenCASCADE compiled to WASM), meshes them, writes a glTF, and lets
 * gltf-pipeline emit the final .glb.
 *
 * Usage:
 *   node scripts/step-to-glb.mjs <input.step> <part-id> [--draco]
 *
 * Example:
 *   node scripts/step-to-glb.mjs ~/Downloads/manifold.step cdu-liquid-manifold
 *
 * Output: src/scene/models/<part-id>.glb  → auto-loaded by model-registry.
 *
 * Deps are fetched on demand via `npx`; nothing is added to package.json.
 * Tip: STEP solids are dense — run a Blender "Decimate" if the part exceeds
 * ~50k triangles, or accept the auto-mesh (fine for these schematic views).
 */
import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, '../src/scene/models');
const require = createRequire(import.meta.url);

const argv = process.argv.slice(2);
const useDraco = argv.includes('--draco');
const [inputArg, nameArg] = argv.filter((a) => a !== '--draco');

if (!inputArg || !nameArg) {
  console.error('用法: node scripts/step-to-glb.mjs <input.step> <part-id> [--draco]');
  process.exit(1);
}
const input = resolve(process.cwd(), inputArg);
if (!existsSync(input)) { console.error(`✗ 找不到输入文件: ${input}`); process.exit(1); }
const ext = extname(input).toLowerCase();
if (ext !== '.step' && ext !== '.stp') { console.error(`✗ 仅支持 .step/.stp，收到 ${ext}`); process.exit(1); }

mkdirSync(OUT_DIR, { recursive: true });

// ── ensure occt-import-js is available (install into a scratch dir if missing) ──
function loadOcct() {
  try { return require('occt-import-js'); } catch { /* fall through */ }
  console.log('· 首次运行：安装 occt-import-js（OpenCASCADE WASM）…');
  execSync('npm install --no-save occt-import-js', { stdio: 'inherit', cwd: resolve(__dirname, '..') });
  return require('occt-import-js');
}

function align4(n) { return (n + 3) & ~3; }

async function main() {
  const occtimportjs = loadOcct();
  const occt = await occtimportjs();
  const fileBuf = new Uint8Array(readFileSync(input));
  const res = occt.ReadStepFile(fileBuf, null);
  if (!res || !res.success || !res.meshes?.length) {
    console.error('✗ STEP 解析失败或没有可网格化的实体。');
    process.exit(2);
  }
  console.log(`· 解析到 ${res.meshes.length} 个实体，开始构建 glTF…`);

  // assemble a single binary buffer; one primitive (+material) per mesh
  const chunks = [];
  let byteLen = 0;
  const bufferViews = [];
  const accessors = [];
  const materials = [];
  const meshes = [];
  const nodes = [];

  const pushView = (typedArray, target) => {
    const bytes = Buffer.from(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength);
    const padded = align4(byteLen) - byteLen;
    if (padded) { chunks.push(Buffer.alloc(padded)); byteLen += padded; }
    const offset = byteLen;
    chunks.push(bytes); byteLen += bytes.length;
    bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length, ...(target ? { target } : {}) });
    return bufferViews.length - 1;
  };

  res.meshes.forEach((m, i) => {
    const pos = Float32Array.from(m.attributes.position.array);
    const idxArr = m.index.array;
    const idx = Uint32Array.from(idxArr);
    // bounds for POSITION accessor
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let p = 0; p < pos.length; p += 3) for (let k = 0; k < 3; k++) {
      const v = pos[p + k]; if (v < min[k]) min[k] = v; if (v > max[k]) max[k] = v;
    }
    const posView = pushView(pos, 34962);
    accessors.push({ bufferView: posView, componentType: 5126, count: pos.length / 3, type: 'VEC3', min, max });
    const posAcc = accessors.length - 1;

    let normAcc;
    if (m.attributes.normal?.array?.length) {
      const norm = Float32Array.from(m.attributes.normal.array);
      const nView = pushView(norm, 34962);
      accessors.push({ bufferView: nView, componentType: 5126, count: norm.length / 3, type: 'VEC3' });
      normAcc = accessors.length - 1;
    }
    const idxView = pushView(idx, 34963);
    accessors.push({ bufferView: idxView, componentType: 5125, count: idx.length, type: 'SCALAR' });
    const idxAcc = accessors.length - 1;

    const c = m.color || [0.78, 0.80, 0.84];
    // deliberately drop source product names → keep models brand-free (anti-scrape)
    materials.push({ pbrMetallicRoughness: { baseColorFactor: [c[0], c[1], c[2], 1], metallicFactor: 0.4, roughnessFactor: 0.6 }, name: `mat_${i}` });
    const attributes = { POSITION: posAcc };
    if (normAcc !== undefined) attributes.NORMAL = normAcc;
    meshes.push({ primitives: [{ attributes, indices: idxAcc, material: i }], name: `mesh_${i}` });
    nodes.push({ mesh: i });
  });

  const bin = Buffer.concat(chunks);
  const gltf = {
    asset: { version: '2.0', generator: 'step-to-glb.mjs (occt-import-js)' },
    scene: 0,
    scenes: [{ nodes: nodes.map((_, i) => i) }],
    nodes,
    meshes,
    materials,
    accessors,
    bufferViews,
    buffers: [{ byteLength: bin.length, uri: `data:application/octet-stream;base64,${bin.toString('base64')}` }],
  };

  const tmp = resolve(OUT_DIR, `${nameArg}.tmp.gltf`);
  const out = resolve(OUT_DIR, `${nameArg}.glb`);
  writeFileSync(tmp, JSON.stringify(gltf));
  console.log('· 压缩 / 打包为 GLB…');
  execSync(`npx --yes gltf-pipeline -i "${tmp}" -o "${out}"${useDraco ? ' -d' : ''}`, { stdio: 'inherit' });
  rmSync(tmp, { force: true });

  console.log(`\n✓ 完成: ${out}`);
  console.log(`已按 part id「${nameArg}」命名，刷新页面即自动加载。`);
  console.log(`⚠ 朝向若不对，在 parts-catalog.ts 给该零件加 modelRotationDeg: [x,y,z]。\n`);
}

main().catch((e) => { console.error('\n✗ 转换失败:', e?.message || e); process.exit(1); });
