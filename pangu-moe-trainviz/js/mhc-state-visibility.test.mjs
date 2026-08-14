import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(
  new URL('../op-rank-time-openpangu-flash-events.html', import.meta.url),
  'utf8'
);

const functionSource = page.match(
  /function syncCssDeckMhcStateBundle\([\s\S]*?\n}\nfunction syncCssDeckTransferGapCells/
)?.[0];

assert.ok(functionSource, 'mHC state bundle visibility function should exist');
assert.match(
  functionSource,
  /const stateLaneVisible=cssDeckMode&&cssDeckView==='right';/,
  'the mHC layer-boundary lane should follow the right-side projection'
);
assert.doesNotMatch(
  functionSource,
  /activeLens|numericsMode/,
  'lens selection must not hide the persistent mHC layer-boundary lane'
);
assert.match(
  functionSource,
  /group\.removeAttribute\('display'\);\s*group\.style\.display='block';/,
  'a later valid layout pass should clear any stale SVG display attribute'
);
assert.match(
  functionSource,
  /const inputOutputY=inputStateRect\?inputStateRect\.bottom-layerRect\.top:NaN;/,
  'the mHC state lane should use the Parallel Embedding output-port height'
);
assert.match(
  functionSource,
  /const anchorY=role==='input'\?bottom:/,
  'the input terminal should attach to the bottom output port of Parallel Embedding'
);
assert.match(
  functionSource,
  /\?`M \$\{anchorX\.toFixed\(2\)\} \$\{y\.toFixed\(2\)\} L \$\{x\.toFixed\(2\)\} \$\{y\.toFixed\(2\)\}`/,
  'an aligned input terminal should be rendered as a horizontal segment'
);

console.log('mHC state visibility tests passed');
