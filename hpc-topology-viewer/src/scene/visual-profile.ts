import { createContext, useContext } from 'react';

export type SceneVisualProfile = 'default' | 'opRankTime';

export const SceneVisualProfileContext = createContext<SceneVisualProfile>('default');

export function useSceneVisualProfile(): SceneVisualProfile {
  return useContext(SceneVisualProfileContext);
}

export function isOpRankTimeProfile(profile: SceneVisualProfile): boolean {
  return profile === 'opRankTime';
}

export function resolveSceneMaterial(profile: SceneVisualProfile, metalness: number, roughness: number, emissiveIntensity = 0) {
  if (!isOpRankTimeProfile(profile)) return { metalness, roughness, emissiveIntensity };
  return {
    metalness: Math.min(metalness, 0.035),
    roughness: Math.max(roughness, 0.72),
    emissiveIntensity: emissiveIntensity * 0.42,
  };
}

export function sceneSurface(dark: boolean, profile: SceneVisualProfile) {
  if (isOpRankTimeProfile(profile)) {
    return {
      background: dark ? '#0e1116' : '#ffffff',
      fog: dark ? '#0e1116' : '#ffffff',
      fogNear: dark ? 100 : 130,
      fogFar: dark ? 460 : 520,
      ambient: dark ? 1.62 : 1.58,
      key: dark ? 0.92 : 1.02,
      fill: dark ? 0.46 : 0.74,
      point: dark ? 0.24 : 0.22,
      pointColor: dark ? '#d8e1f0' : '#ffffff',
    };
  }
  return {
    background: dark ? '#101010' : '#f5f5f5',
    fog: dark ? '#101010' : '#f5f5f5',
    fogNear: 26,
    fogFar: 60,
    ambient: dark ? 1.35 : 1.05,
    key: dark ? 0.95 : 1.2,
    fill: dark ? 0.55 : 0.75,
    point: dark ? 0.7 : 1.0,
    pointColor: dark ? '#7e93cf' : '#e8f0ff',
  };
}
