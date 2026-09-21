#!/usr/bin/env node
/**
 * convert-to-glb.mjs — convert open-source hardware models to optimized GLB and
 * drop them where the model-registry auto-detects them.
 *
 * Usage:
 *   node scripts/convert-to-glb.mjs <input-file> <part-id> [--draco]
 *
 * IMPORTANT: <part-id> MUST equal a CatalogPart.id (see src/scene/parts-catalog.ts)
 *   e.g.  an OAM accelerator model → part id "npu-accelerator-module"
 *
 * Examples:
 *   node scripts/convert-to-glb.mjs ~/Downloads/oam.glb  npu-accelerator-module
 *   node scripts/convert-to-glb.mjs ~/Downloads/dimm.obj mem-ddr5-rdimm
 *
 * Output: src/scene/models/<part-id>.glb  → auto-loaded, no code edits needed.
 *
 * Supported inputs:
 *   .glb / .gltf   → optimize                       (via npx gltf-pipeline)
 *   .obj           → convert + optimize             (via npx obj2gltf + gltf-pipeline)
 *   .stl           → parse (binary/ASCII) + optimize
 *   .step / .stp   → use step-to-glb.mjs instead (OpenCASCADE)
 *   .iges / .x_t   → re-save as STEP/GLB in a CAD tool first
 *
 * Compression: by default NO Draco (so runtime needs no external decoder — keeps
 *   the offline / noindex deployment self-contained). Pass --draco to shrink the
 *   file (then the browser fetches the Draco decoder at load time).
 *
 * No package.json deps are added — tools are fetched on demand via `npx`.
 */
import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, rmSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, '../src/scene/models');

const args = process.argv.slice(2).filter((a) => a !== '--draco');
const useDraco = process.argv.includes('--draco');
const [inputArg, nameArg] = args;

if (!inputArg || !nameArg) {
  console.error('用法: node scripts/convert-to-glb.mjs <输入文件> <part-id> [--draco]');
  console.error('示例: node scripts/convert-to-glb.mjs ~/Downloads/oam.glb npu-accelerator-module');
  process.exit(1);
}

const input = resolve(process.cwd(), inputArg);
if (!existsSync(input)) {
  console.error(`✗ 找不到输入文件: ${input}`);
  process.exit(1);
}

mkdirSync(OUT_DIR, { recursive: true });
const out = resolve(OUT_DIR, `${nameArg}.glb`);
const ext = extname(input).toLowerCase();
const dracoFlag = useDraco ? ' -d' : '';

function run(cmd) {
  console.log(`$ ${cmd}`);
  execSync(cmd, { stdio: 'inherit' });
}

function align4(n) { return (n + 3) & ~3; }

/** Parse STL (binary or ASCII) → flat positions + normals, then write a glTF
 *  with an embedded buffer so gltf-pipeline can emit the .glb. */
function stlToGltf(stlPath, gltfPath) {
  const buf = readFileSync(stlPath);
  const ascii = buf.slice(0, 5).toString('ascii').toLowerCase() === 'solid'
    && !(buf.length > 84 && buf.readUInt32LE(80) * 50 + 84 === buf.length);
  const positions = [];
  const normals = [];
  if (ascii) {
    const txt = buf.toString('ascii');
    const fre = /facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)([\s\S]*?)endfacet/g;
    const vre = /vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)/g;
    let f;
    while ((f = fre.exec(txt))) {
      const nx = +f[1], ny = +f[2], nz = +f[3];
      let v; const body = f[4]; vre.lastIndex = 0;
      while ((v = vre.exec(body))) { positions.push(+v[1], +v[2], +v[3]); normals.push(nx, ny, nz); }
    }
  } else {
    const count = buf.readUInt32LE(80);
    let o = 84;
    for (let i = 0; i < count; i++) {
      const nx = buf.readFloatLE(o), ny = buf.readFloatLE(o + 4), nz = buf.readFloatLE(o + 8); o += 12;
      for (let v = 0; v < 3; v++) { positions.push(buf.readFloatLE(o), buf.readFloatLE(o + 4), buf.readFloatLE(o + 8)); o += 12; normals.push(nx, ny, nz); }
      o += 2;
    }
  }
  const pos = Float32Array.from(positions);
  const nrm = Float32Array.from(normals);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) for (let k = 0; k < 3; k++) { const x = pos[i + k]; if (x < min[k]) min[k] = x; if (x > max[k]) max[k] = x; }
  const posBytes = Buffer.from(pos.buffer);
  const padN = align4(posBytes.length) - posBytes.length;
  const nrmBytes = Buffer.from(nrm.buffer);
  const bin = Buffer.concat([posBytes, Buffer.alloc(padN), nrmBytes]);
  const nOff = posBytes.length + padN;
  const n = pos.length / 3;
  const gltf = {
    asset: { version: '2.0', generator: 'convert-to-glb.mjs (stl)' },
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.8, 0.82, 0.86, 1], metallicFactor: 0.4, roughnessFactor: 0.6 } }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: n, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5126, count: n, type: 'VEC3' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: posBytes.length, target: 34962 },
      { buffer: 0, byteOffset: nOff, byteLength: nrmBytes.length, target: 34962 },
    ],
    buffers: [{ byteLength: bin.length, uri: `data:application/octet-stream;base64,${bin.toString('base64')}` }],
  };
  writeFileSync(gltfPath, JSON.stringify(gltf));
}

try {
  if (ext === '.glb' || ext === '.gltf') {
    run(`npx --yes gltf-pipeline -i "${input}" -o "${out}"${dracoFlag}`);
  } else if (ext === '.obj') {
    const tmp = resolve(OUT_DIR, `${nameArg}.tmp.glb`);
    run(`npx --yes obj2gltf -i "${input}" -o "${tmp}" -b`);
    run(`npx --yes gltf-pipeline -i "${tmp}" -o "${out}"${dracoFlag}`);
    rmSync(tmp, { force: true });
  } else if (ext === '.stl') {
    const tmp = resolve(OUT_DIR, `${nameArg}.tmp.gltf`);
    stlToGltf(input, tmp);
    run(`npx --yes gltf-pipeline -i "${tmp}" -o "${out}"${dracoFlag}`);
    rmSync(tmp, { force: true });
  } else if (ext === '.step' || ext === '.stp' || ext === '.iges' || ext === '.igs') {
    console.error(`\n✗ ${ext} 无法在纯 Node 直接转换。`);
    if (ext === '.step' || ext === '.stp') {
      console.error('  STEP → 用全自动脚本: node scripts/step-to-glb.mjs ' + inputArg + ' ' + nameArg + '\n');
    } else {
      console.error('  方案 1 — Blender：导入 → File → Export → glTF 2.0 (.glb)，再跑本脚本压缩');
      console.error('  方案 2 — IGES → 先在 FreeCAD 另存 STEP，再 step-to-glb.mjs\n');
    }
    process.exit(2);
  } else {
    console.error(`✗ 不支持的格式: ${ext}`);
    process.exit(1);
  }

  console.log(`\n✓ 完成: ${out}`);
  console.log(`已按 part id「${nameArg}」命名，model-registry 会自动加载，刷新页面即可生效。`);
  console.log(`⚠ 若仍是程序化几何：确认 ${nameArg} 与 parts-catalog.ts 里的 id 完全一致。\n`);
} catch (err) {
  console.error('\n✗ 转换失败:', err.message);
  process.exit(1);
}
