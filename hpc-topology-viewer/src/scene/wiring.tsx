// ─── 连线样式（bus-wiring 风格）──────────────────────────────────────────────
// 参考 https://github.com/Cinnnnnnndy/bus-wiring 的连线样式：圆角管体（TubeGeometry
// 走 roundedCurve 二次贝塞尔折角）+ ShaderMaterial 实色 + 沿线流动的白色「彗星」高亮
// + 两端 connector 接点。颜色 / 粗细 / 透明度规则沿用本项目原有取值（lineWidth→管径）。
//
// 一个 <Wire> 取代原先的 drei <Line> / <lineSegments> / FlowLine / LinkTube：
//   • 连续折线（continuous）→ 单条圆角管 + 两端接点。
//   • segments 模式（点成对）→ 多段直管合并为一条几何（一次 draw call），可带逐段顶点色。
//   • 段数超过 maxTubes 时退回 drei <Line>（保护超大规模 full-mesh 的性能）。
import { useMemo, useEffect, createContext, useContext } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

export type Pt = [number, number, number];

// Uniform visual scale for communication wires. Keep caller lineWidth ratios, but
// render the whole family a bit slimmer so dense 3D topology fields stay readable.
const WIRE_WIDTH_SCALE = 0.25;
const WIRE_OPACITY_SCALE = 0.5;
// 世界半径 / 单位 lineWidth：调到 width 1≈0.012、width 4≈0.048（与 bus-wiring 0.038~0.05 同档）。
const R_PER_W = 0.012;
// 每个场景的半径倍率（FullPod 大场按 field 放大；默认 1）。
export const WireScale = createContext(1);

// ── 彗星着色器（移植自 bus-wiring，扩展：逐顶点色 + 虚线）──
const WIRE_VERT = /* glsl */ `
  varying float vArc;
  #ifdef USE_VCOLOR
    attribute vec3 vcolor;
    varying vec3 vCol;
  #endif
  void main() {
    vArc = uv.x;
    #ifdef USE_VCOLOR
      vCol = vcolor;
    #endif
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const WIRE_FRAG = /* glsl */ `
  uniform vec3  uColor;
  uniform float uOpacity;
  uniform float uActive;
  uniform float uOffset;
  uniform float uDensity;
  uniform float uDashed;
  varying float vArc;
  #ifdef USE_VCOLOR
    varying vec3 vCol;
  #endif
  void main() {
    #ifdef USE_VCOLOR
      vec3 base = vCol;
    #else
      vec3 base = uColor;
    #endif
    // 流动彗星：一段沿线移动的白色亮带
    float p    = fract(vArc * uDensity - uOffset);
    float band = smoothstep(0.0, 0.04, p) * (1.0 - smoothstep(0.04, 0.18, p));
    vec3  col  = mix(base, vec3(1.0), band * uActive * 0.6);
    // 虚线（idle 链路）：沿线开槽
    float dash = uDashed > 0.5 ? smoothstep(0.46, 0.5, fract(vArc * 9.0)) : 1.0;
    float a    = uOpacity * dash;
    if (a < 0.012) discard;
    gl_FragColor = vec4(col, a);
  }
`;

/** 把折点串转成带圆角的 CurvePath（折角处插二次贝塞尔）——移植自 bus-wiring/routing.ts。 */
export function roundedCurve(pts: THREE.Vector3[], radius: number): THREE.CurvePath<THREE.Vector3> {
  const cp = new THREE.CurvePath<THREE.Vector3>();
  if (pts.length < 2) return cp;
  if (pts.length === 2) { cp.add(new THREE.LineCurve3(pts[0].clone(), pts[1].clone())); return cp; }
  let from = pts[0].clone();
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i];
    const dIn = p.clone().sub(pts[i - 1]); const lIn = dIn.length() || 1; dIn.normalize();
    const dOut = pts[i + 1].clone().sub(p); const lOut = dOut.length() || 1; dOut.normalize();
    const r = Math.min(radius, lIn * 0.5, lOut * 0.5);
    const cStart = p.clone().addScaledVector(dIn, -r);
    const cEnd = p.clone().addScaledVector(dOut, r);
    cp.add(new THREE.LineCurve3(from.clone(), cStart));
    cp.add(new THREE.QuadraticBezierCurve3(cStart, p.clone(), cEnd));
    from = cEnd;
  }
  cp.add(new THREE.LineCurve3(from.clone(), pts[pts.length - 1].clone()));
  return cp;
}

const V = (p: Pt | THREE.Vector3) => (p instanceof THREE.Vector3 ? p : new THREE.Vector3(p[0], p[1], p[2]));

/** 把多段管体的 position / uv (/vcolor) 拍平合并成一条非索引几何（一次 draw call）。 */
function mergeFlat(geos: THREE.BufferGeometry[], colors: Pt[] | null): THREE.BufferGeometry {
  const flats = geos.map((g) => (g.index ? g.toNonIndexed() : g));
  let total = 0;
  for (const f of flats) total += f.getAttribute('position').count;
  const position = new Float32Array(total * 3);
  const uv = new Float32Array(total * 2);
  const vcol = colors ? new Float32Array(total * 3) : null;
  let o3 = 0, o2 = 0;
  flats.forEach((f, i) => {
    const pa = f.getAttribute('position'), ua = f.getAttribute('uv');
    position.set(pa.array as ArrayLike<number>, o3);
    if (ua) uv.set(ua.array as ArrayLike<number>, o2);
    if (vcol && colors) { const [r, g, b] = colors[i] ?? [1, 1, 1]; for (let k = 0; k < pa.count; k++) { vcol[o3 + k * 3] = r; vcol[o3 + k * 3 + 1] = g; vcol[o3 + k * 3 + 2] = b; } }
    o3 += pa.count * 3; o2 += pa.count * 2;
  });
  // 释放临时（toNonIndexed 产生的）几何
  flats.forEach((f, i) => { if (f !== geos[i]) f.dispose(); });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(position, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  if (vcol) g.setAttribute('vcolor', new THREE.BufferAttribute(vcol, 3));
  return g;
}

export interface WireProps {
  points: Array<Pt | THREE.Vector3>;
  /** points 为成对端点（每 2 个点一段独立直线）。 */
  segments?: boolean;
  color?: string;
  /** segments 模式逐「源点」颜色（每段 2 个，取首个）——用于负载热力图。 */
  vertexColors?: Pt[];
  /** 沿用原 lineWidth（→ 管径，按 WireScale 缩放）。 */
  lineWidth?: number;
  /** 显式世界半径（覆盖 lineWidth 映射）。 */
  radius?: number;
  opacity?: number;
  /** 流动彗星高亮（active 链路 / 通信中）。 */
  active?: boolean;
  /** idle 链路：管体开槽呈虚线。 */
  dashed?: boolean;
  /** 两端 connector 接点（色壳 + 白芯）。连续折线默认 true，segments 默认 false。 */
  endpoints?: boolean;
  /** 彗星流速。 */
  speed?: number;
  /** 折角圆角半径（连续折线）。 */
  cornerRadius?: number;
  radialSegments?: number;
  /** 段数上限：超过则退回 drei <Line>（性能保护）。 */
  maxTubes?: number;
  renderOrder?: number;
}

/** 端点接点：色壳球 + 白芯（方向无关，适配任意走向的连线）。 */
function Connector({ at, r, color, opacity }: { at: THREE.Vector3; r: number; color: string; opacity: number }) {
  return (
    <group position={at}>
      <mesh renderOrder={3}><sphereGeometry args={[r * 2.4, 14, 14]} /><meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} toneMapped={false} /></mesh>
      <mesh renderOrder={4}><sphereGeometry args={[r * 1.05, 10, 10]} /><meshBasicMaterial color="#ffffff" transparent opacity={Math.min(0.95, opacity + 0.1)} depthWrite={false} toneMapped={false} /></mesh>
    </group>
  );
}

export function Wire({
  points, segments = false, color = '#9aa4b2', vertexColors, lineWidth = 1, radius,
  opacity = 0.6, active = false, dashed = false, endpoints, speed = 0.6, cornerRadius,
  radialSegments = 6, maxTubes = 1400, renderOrder,
}: WireProps) {
  const scale = useContext(WireScale);
  const renderedLineWidth = lineWidth * WIRE_WIDTH_SCALE;
  const renderedOpacity = opacity * WIRE_OPACITY_SCALE;
  const r = radius ?? Math.max(0.003, renderedLineWidth * R_PER_W * scale);
  const useVColor = !!vertexColors;

  const verts = useMemo(() => points.map(V), [points]);

  // 段数过大（超大 full-mesh）→ 退回 drei <Line>，保护性能。
  const segCount = segments ? Math.floor(verts.length / 2) : 0;
  const fallback = segments && segCount > maxTubes;

  const built = useMemo(() => {
    if (fallback || verts.length < 2) return null;
    if (segments) {
      const geos: THREE.BufferGeometry[] = [];
      const cols: Pt[] | null = useVColor ? [] : null;
      for (let i = 0; i + 1 < verts.length; i += 2) {
        const a = verts[i], b = verts[i + 1];
        if (a.distanceTo(b) < 1e-4) continue;
        geos.push(new THREE.TubeGeometry(new THREE.LineCurve3(a, b), 1, r, radialSegments, false));
        if (cols) cols.push(vertexColors![i] ?? [1, 1, 1]);
      }
      if (!geos.length) return null;
      const merged = mergeFlat(geos, cols);
      geos.forEach((g) => g.dispose());
      return { geo: merged, density: 1.4 };
    }
    const cp = roundedCurve(verts, cornerRadius ?? r * 6);
    const len = Math.max(cp.getLength(), 0.01);
    const divisions = Math.max(verts.length * 14, 32);
    return { geo: new THREE.TubeGeometry(cp, divisions, r, radialSegments, false), density: Math.min(Math.max(len * 0.45, 1), 24) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [verts, segments, r, radialSegments, useVColor, cornerRadius, fallback]);

  const mat = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: WIRE_VERT, fragmentShader: WIRE_FRAG,
    defines: useVColor ? { USE_VCOLOR: '' } : {},
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: renderedOpacity },
      uActive: { value: active ? 1 : 0 },
      uOffset: { value: 0 },
      uDensity: { value: built?.density ?? 1.4 },
      uDashed: { value: dashed ? 1 : 0 },
    },
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [useVColor]);

  useEffect(() => {
    mat.uniforms.uColor.value.set(color);
    mat.uniforms.uOpacity.value = renderedOpacity;
    mat.uniforms.uActive.value = active ? 1 : 0;
    mat.uniforms.uDashed.value = dashed ? 1 : 0;
    if (built) mat.uniforms.uDensity.value = built.density;
  }, [color, renderedOpacity, active, dashed, built, mat]);

  useEffect(() => () => mat.dispose(), [mat]);
  useEffect(() => () => built?.geo.dispose(), [built]);

  useFrame((_, dt) => { if (active) mat.uniforms.uOffset.value += speed * dt; });

  if (fallback) {
    return (
      <Line points={verts} segments color={color} vertexColors={vertexColors as [number, number, number][] | undefined}
        lineWidth={renderedLineWidth} transparent opacity={renderedOpacity} renderOrder={renderOrder} />
    );
  }
  if (!built) return null;

  const showCaps = (endpoints ?? !segments) && !useVColor && verts.length >= 2;
  const capOpacity = renderedOpacity;

  return (
    <group>
      <mesh geometry={built.geo} material={mat} renderOrder={renderOrder} />
      {showCaps && <Connector at={verts[0]} r={r} color={color} opacity={capOpacity} />}
      {showCaps && <Connector at={verts[verts.length - 1]} r={r} color={color} opacity={capOpacity} />}
    </group>
  );
}
