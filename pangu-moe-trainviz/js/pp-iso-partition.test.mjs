import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(
  new URL('../op-rank-time-openpangu-flash-events.html', import.meta.url),
  'utf8'
);

const geometryBlock = page.match(
  /function cssDeckPpStageGeometry\(\)[\s\S]*?\n}\nfunction cssDeckPpWireframeHtml/
)?.[0];
const generator = page.match(
  /function cssDeckPpWireframeHtml\(\)[\s\S]*?\n}\nfunction cssDeckPpLabelsHtml/
)?.[0];
const selectionGuard = page.match(
  /function cssDeckPpPartitionsSelected\(\)[\s\S]*?\n}\nfunction syncCssDeckPpPartitions/
)?.[0];
const visibilitySync = page.match(
  /function syncCssDeckPpPartitions\(\)[\s\S]*?\n}\nfunction renderCssDeck/
)?.[0];

assert.ok(geometryBlock, 'PP stage geometry should exist');
const geometrySource = geometryBlock.slice(0, geometryBlock.lastIndexOf('\nfunction cssDeckPpWireframeHtml'));
const buildGeometry = new Function(
  'DEMO_STAGE_RANGES',
  'CSS_DECK_GAP',
  `${geometrySource}; return cssDeckPpStageGeometry();`
);
const geometry = buildGeometry([[0, 11], [12, 22], [23, 34], [35, 45]], 54);
assert.equal(geometry.length, 4, 'PP=4 should produce four stage volumes');
geometry.forEach((stage, index) => {
  assert.equal(stage.stage, index);
  assert.equal(stage.corners.length, 8, `PP${index} should have eight cube corners`);
  assert.equal(new Set(stage.corners.map((point) => `${point.x},${point.y},${point.z}`)).size, 8, `PP${index} corners should be unique`);
  assert.equal(stage.depth, (stage.hi - stage.lo + 1) * 54 - 28, `PP${index} depth should preserve the visible gap`);
});
for (let index = 0; index < geometry.length - 1; index += 1) {
  assert.equal(geometry[index].zMin - geometry[index + 1].zMax, 28, 'adjacent PP volumes should keep a 28px model-space gap');
}

const faceEdges = [[0, 1], [1, 2], [2, 3], [3, 0]];
const logicalEdges = [
  ...faceEdges,
  ...faceEdges.map(([a, b]) => [a + 4, b + 4]),
  [0, 4], [1, 5], [2, 6], [3, 7],
];
assert.equal(new Set(logicalEdges.map(([a, b]) => [a, b].sort((x, y) => x - y).join('-'))).size, 12, 'each PP volume should expose twelve unique logical edges');

assert.ok(generator, 'PP SVG wireframe generator should exist');
assert.match(generator, /cssDeckPpStageGeometry\(\)\.map\(\(\{stage,lo,hi\}\)=>/, 'wireframes should derive from shared PP stage geometry');
assert.equal((generator.match(/<path data-part=/g) || []).length, 3, 'one stage should group far, depth, and near edges into three low-cost paths');
assert.match(generator, /data-model-layer-slices/, 'virtualized model layers should share the low-cost SVG projection plane');
assert.match(page, /<svg id="cssDeckPpWireframe"[^>]*><\/svg>\s*<div id="cssDeckPpLabels"/, 'the SVG wireframe and labels should be siblings after the CSS 3D scene');
assert.match(page, /cssDeckScene\.innerHTML=input\+Array\.from/, 'the model scene should render without PP geometry inside its preserve-3d context');
assert.doesNotMatch(page, /opv-cssdeck-pp-box__face|opv-cssdeck-pp-partitions/, 'large CSS 3D partition faces should be removed');

assert.ok(selectionGuard, 'PP partition selection guard should exist');
assert.match(selectionGuard, /activeLens==='communication'/, 'only Communication mode may reveal PP partitions');
assert.match(selectionGuard, /parallelEventFilter==='PP'/, 'the PP communication filter should reveal partitions');
assert.match(selectionGuard, /parallelEventFilter==='all'&&objectFocus==='comm:pp'/, 'the linked object filter may reveal partitions while the event filter is ALL');

assert.ok(visibilitySync, 'PP partition visibility sync should exist');
assert.match(visibilitySync, /cssDeckMode&&cssDeckRendered&&cssDeckView==='iso'&&cssDeckPpPartitionsSelected\(\)/, 'only the rendered axis view may show the overlay');
assert.match(visibilitySync, /cssDeck\.dataset\.ppPartitions=visible\?'on':'off'/);
assert.match(visibilitySync, /if\(visible\)\{\s*scheduleCssDeckPpWireframe\(\);/, 'revealing PP should schedule a fresh projection');
assert.match(visibilitySync, /if\(!wasVisible\)requestAnimationFrame\(\(\)=>\{\s*if\(cssDeck\.dataset\.ppPartitions==='on'\)fitCssDeck\(\);/, 'the first PP reveal should fit all four stage volumes into the viewport');

assert.match(page, /\.opv-cssdeck-pp-wireframe\{[^}]*z-index:26;[^}]*contain:strict;pointer-events:none/, 'the wireframe should be a contained non-interactive 2D overlay');
assert.match(page, /\.opv-cssdeck-pp-wireframe \[data-part="far"\]\{stroke-width:\.9;stroke-opacity:\.42\}/, 'far cube edges should remain quiet');
assert.match(page, /\.opv-cssdeck-pp-wireframe \[data-part="depth"\]\{stroke-width:1\.35;stroke-opacity:\.66\}/, 'depth edges should connect the volume without dominating it');
assert.match(page, /\.opv-cssdeck-pp-wireframe \[data-part="near"\]\{stroke-width:2\.25;stroke-opacity:\.90\}/, 'near cube edges should carry a restrained stage boundary');
assert.match(page, /\.opv-cssdeck-pp-wireframe \[data-layer-slices\]\{[^}]*stroke-width:\.7;stroke-opacity:\.24\}/, 'virtualized layer slices should stay quieter than PP boundaries');
assert.match(page, /vector-effect:non-scaling-stroke/, 'wireframe borders should stay readable while zooming');
assert.match(page, /function cssDeckPpProjectPoint\(point,matrix,metrics\)/, '3D cube corners should project into the SVG overlay');
assert.match(page, /new DOMPoint\(point\.x,point\.y,point\.z,1\)\.matrixTransform\(matrix\)/, 'projection should consume the live CSS scene matrix');
assert.match(page, /function cssDeckSceneTransformValue\(/, 'the model and SVG overlay should share one transform snapshot');
assert.match(page, /const matrix=new MatrixCtor\(cssDeckSceneTransformValue\(\)\)/, 'projection should not sample a stale compositor transform');
assert.match(page, /sceneLeft:width\/2\+cssDeckScreenOffset\.x/, 'projection should use the same deterministic scene offset as the model');
assert.match(page, /function cssDeckPpLayerIsSample\(layer,range=/, 'PP mode should define a stable layer sampling policy');
assert.match(page, /data-stage-sample="\$\{stageSample\}"/, 'every full model card should expose whether it belongs to the PP sample set');
assert.match(page, /\.filter\(layer=>!cssDeckPpLayerIsSample\(layer,range\)\)/, 'non-sampled layers should be projected as lightweight SVG slices');
assert.match(page, /faceAZ>=faceBZ\?0:4/, 'near and far faces should be recomputed from the active camera');
assert.match(page, /function scheduleCssDeckPpWireframe\(\)/, 'projection updates should be rAF-coalesced');
assert.match(page, /cssDeckScene\.style\.top=[^;]+;\s*scheduleCssDeckPpWireframe\(\);/, 'camera pan should update the wireframe');
assert.match(page, /cssDeckScene\.style\.transform=cssDeckSceneTransformValue\(\);\s*scheduleCssDeckPpWireframe\(\);/, 'orbit and zoom should update model and wireframe from the same transform');
assert.match(page, /function cssDeckPpLabelSafeTop\(height\)/, 'stage labels should derive their safe band from the visible top controls');
assert.match(page, /Math\.max\(labelSafeTop,Math\.min\(height-labelHeight-12,anchorY\)\)/, 'stage labels should remain below the live top toolbar safe band');
assert.match(page, /const fitPpWireframe=cssDeckView==='iso'&&cssDeckPpPartitionsSelected\(\)/, 'axis fit should detect the PP wireframe state');
assert.match(page, /fitPpWireframe\s*\? Math\.min\(rect\.width\/1850,rect\.height\/2050\)/, 'PP fit should reserve enough space for the complete depth-projected boxes');

assert.match(page, /parallelEventFilter=tag;\s*clearParallelEventSelection\(\);\s*if\(tag!=='PP'\)setCssGraphView\('front'\);\s*syncCssDeckPpPartitions\(\);/);
assert.doesNotMatch(page, /setCssGraphView\(tag==='PP'\?'right':'front'\)/, 'PP selection should retain the current camera view');
assert.match(page, /\.opv-stage\{position:relative;isolation:isolate;overflow:hidden/, 'the stage should isolate 3D content from UI siblings');
assert.match(page, /\.opv-cssdeck\{[^}]*z-index:18;isolation:isolate;contain:layout paint;/, 'the CSS 3D compositor should remain paint-contained');
assert.match(page, /\.opv-stage-overlay-plane\{[^}]*z-index:500;isolation:isolate;contain:layout paint;/, 'all 2D overlays and controls should live in a paint-contained plane above the 3D compositor');
assert.match(page, /\.opv-cssdeck\[data-css-view="iso"\] \.opv-cssdeck-card,[\s\S]*?transform-style:flat;backface-visibility:hidden/, 'axis-view cards should flatten their internal graph to reduce unstable 3D compositor layers');
assert.match(page, /\.opv-cssdeck\[data-css-view="iso"\]\[data-pp-partitions="on"\] \.opv-cssdeck-card\[data-stage-sample="false"\]\{display:none\}/, 'PP mode should remove non-sampled full cards from compositor tiling');
assert.match(page, /data-stage-sample="false"\]\.is-selected,[\s\S]*?\.is-card-linked\{display:block\}/, 'selected or linked non-sampled cards should remain revealable on demand');
assert.match(page, /\.opv-stage-tools\{[^}]*z-index:420;/, 'the right toolbar should own the top layer inside the stable overlay plane');
assert.match(page, /\.opv-parallel-event-legend\{[^}]*z-index:380;/, 'the Communication toolbar should render above the model and PP wireframe');
assert.match(
  page,
  /<div class="opv-stage-overlay-plane"[\s\S]*?<div class="opv-stage-tools"[\s\S]*?<\/div>\s*<\/div>\s*<\/div>\s*<div class="pto-ide-frame__floating-playback-mount"/,
  'the right toolbar should live inside the dedicated stage overlay plane, before the stage closes'
);

console.log('PP axis partition tests passed');
