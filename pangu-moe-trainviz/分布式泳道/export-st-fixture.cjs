// Reproducible extraction; reads the upstream Chrome trace, never executes it.
// Usage: node export-st-fixture.cjs /path/to/timeline-ST.json.gz
const fs = require('node:fs');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const input = fs.readFileSync(process.argv[2]);
const trace = JSON.parse(zlib.gunzipSync(input));
const fixture = {
  source: 'https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/timeline-ST.json.gz',
  configSource: 'https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/meta-ST.yaml',
  sourceSha256: crypto.createHash('sha256').update(input).digest('hex'),
  step: 23, topology: { dp: 2, pp: 4, tp: 1, dsp: 1, vpp: 1, world: 8 },
  sourceTimeUnit: 'us', structureMapping: null,
  events: trace.map((event, index) => ({ ...event, sourceIndex: index }))
    .filter(event => event.ph === 'X' && event.args.step === 23)
};
if (fixture.events.length !== 212) throw new Error('Unexpected ST step 23 event count');
process.stdout.write('/* Extracted ST step 23; see export-st-fixture.cjs and spec for provenance. */\n' +
  '(function(root) { const data = ' + JSON.stringify(fixture, null, 2) + ';\n' +
  'if (typeof module !== "undefined") module.exports = data; else root.STStep23 = data;\n})(globalThis);\n');
