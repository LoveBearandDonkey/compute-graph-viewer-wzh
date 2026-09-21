// ─────────────────────────────────────────────────────────────────────────────
// PartModel — render an installed open-source GLB in place of procedural geometry.
//
// Usage (drop-in, zero behaviour change until a .glb exists):
//
//   <ModelOr partId="npu-accelerator-module" size={[w, h, d]}
//            color={LC.npuBody} edgeColor={LC.rackEdge}>
//     {... existing procedural meshes (the fallback) ...}
//   </ModelOr>
//
// If models/<partId>.glb is present it is loaded, oriented (parts-catalog
// modelRotationDeg), auto-centered, and uniformly scaled to FIT the on-screen
// slot `size`. To keep installed models consistent with the procedural scene's
// flat-block + edge look, pass `color` (re-skins every mesh to that body colour)
// and/or `edgeColor` (adds a sharp-edge wireframe). If no model is installed —
// or it fails to load — the children render unchanged.
// ─────────────────────────────────────────────────────────────────────────────
import { Suspense, useMemo, type ReactNode } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { hasPartModel, resolvePartModelUrl } from './model-registry';
import { partRotationDeg, partFit, type FitMode } from './parts-catalog';
import { resolveSceneMaterial, useSceneVisualProfile } from './visual-profile';

function GlbModel({ url, partId, w, h, d, fit, color, edgeColor, edgeThreshold = 32 }: {
  url: string; partId: string; w: number; h: number; d: number; fit: FitMode;
  color?: string; edgeColor?: string; edgeThreshold?: number;
}) {
  const { scene } = useGLTF(url);
  const profile = useSceneVisualProfile();
  const object = useMemo(() => {
    // Clone so the same cached GLTF can be placed in many slots independently.
    const root = scene.clone(true);
    // 1) orient (applied before measuring so the fit uses the final pose)
    const [rx, ry, rz] = partRotationDeg(partId);
    root.rotation.set(
      THREE.MathUtils.degToRad(rx),
      THREE.MathUtils.degToRad(ry),
      THREE.MathUtils.degToRad(rz),
    );
    root.updateWorldMatrix(true, true);
    // 2) measure → center → fit-scale to the slot
    const box = new THREE.Box3().setFromObject(root);
    const dim = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(dim);
    box.getCenter(center);
    const sx = dim.x > 1e-6 ? w / dim.x : 1;
    const sy = dim.y > 1e-6 ? h / dim.y : 1;
    const sz = dim.z > 1e-6 ? d / dim.z : 1;
    // 3) re-skin to the scene palette + add sharp-edge wireframe so the installed
    //    model matches the procedural flat-block + edge style (legend colours).
    if (color || edgeColor) {
      const mat = resolveSceneMaterial(profile, 0.3, 0.6, 0);
      root.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!(mesh as unknown as { isMesh?: boolean }).isMesh || !mesh.geometry) return;
        if (color) mesh.material = new THREE.MeshStandardMaterial({ color, metalness: mat.metalness, roughness: mat.roughness });
        if (edgeColor) {
          const eg = new THREE.EdgesGeometry(mesh.geometry, edgeThreshold);
          mesh.add(new THREE.LineSegments(eg, new THREE.LineBasicMaterial({
            color: edgeColor,
            transparent: profile === 'opRankTime',
            opacity: profile === 'opRankTime' ? 0.62 : 1,
          })));
        }
      });
    }
    const wrap = new THREE.Group();
    if (fit === 'stretch') wrap.scale.set(sx, sy, sz);
    else wrap.scale.setScalar(Math.min(sx, sy, sz));
    // center the model at the group origin (matches centered procedural slabs)
    root.position.sub(center);
    wrap.add(root);
    return wrap;
  }, [scene, partId, w, h, d, fit, color, edgeColor, edgeThreshold, profile]);

  return <primitive object={object} />;
}

/** Render an installed GLB for `partId`, else fall back to `children`. The fit
 *  mode defaults to the part's catalog setting (override with the `fit` prop).
 *  `color` / `edgeColor` re-skin the model to the scene palette (see header). */
export function ModelOr({ partId, size, fit, color, edgeColor, edgeThreshold, children }: {
  partId: string; size: [number, number, number]; fit?: FitMode;
  color?: string; edgeColor?: string; edgeThreshold?: number; children: ReactNode;
}) {
  if (!hasPartModel(partId)) return <>{children}</>;
  const url = resolvePartModelUrl(partId)!;
  const [w, h, d] = size;
  // Suspense fallback = the procedural children, so there is never a blank frame.
  return (
    <Suspense fallback={<>{children}</>}>
      <GlbModel url={url} partId={partId} w={w} h={h} d={d} fit={fit ?? partFit(partId)}
        color={color} edgeColor={edgeColor} edgeThreshold={edgeThreshold} />
    </Suspense>
  );
}
