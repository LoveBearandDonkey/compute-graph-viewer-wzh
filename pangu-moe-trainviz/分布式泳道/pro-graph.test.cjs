const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const context={structuredClone,URLSearchParams,location:{search:'?trainingProfile=pro'},console};context.window=context;
const runtime=require('./generate-training-data.cjs').generate().runtimeGraph;
const shared=path.resolve(__dirname,'../../vendor/pto-design-system/patterns/model-architecture-training-sidecar');
const model=path.resolve(__dirname,'../architecture/deepseek_v4_flash_csa');
for(const file of ['source-manifest.js','implementation-programs.js','mapping-registry.js'])vm.runInNewContext(fs.readFileSync(path.join(model,file),'utf8'),context);
for(const file of ['pattern.dv4-architecture-data.js','pattern.dv4.js'])vm.runInNewContext(fs.readFileSync(path.join(shared,file),'utf8'),context);
vm.runInNewContext(fs.readFileSync(path.join(model,'json-architecture.js'),'utf8'),context);
for(const collapsed of [new Set(),new Set(['dv4/layer/2']),new Set(['dv4/layer/2/mhc_attn'])]) {
  const graph=context.PtoCsaImplementationPattern.buildWholeModelGraph(new Set(),collapsed,runtime);
  const all=[...graph.nodes,...graph.clusters],ids=new Set(all.map(n=>n.id));
  assert.ok(ids.has('dv4/model/mtp')&&ids.has('dv4/model/loss'));
  assert.ok(ids.has(runtime.id)&&runtime.nodes.every(node=>ids.has(node.id)),'runtime auxiliary graph is rendered beside, not inside, source architecture');
  assert.equal(graph.metadata.runtimeGraphId,runtime.id);
  assert.ok(!all.some(n=>n.id.startsWith('csa_')),'no Flash internals');
  assert.ok(all.every(n=>!n.sourceRefs?.length),'no borrowed Flash source claims');
  for(const e of graph.edges)assert.ok(ids.has(e.source)&&ids.has(e.target),e.id);
  const mtp=graph.nodes.find(n=>n.id==='dv4/model/mtp'),loss=graph.nodes.find(n=>n.id==='dv4/model/loss');
  assert.ok(mtp.kind==='module'&&loss.y<graph.height);
}
const api=context.PtoCsaImplementationPattern;
const l1=api.buildWholeModelGraph(new Set(),api.proLevelCollapsed(false));
const l2=api.buildWholeModelGraph(new Set(),api.proLevelCollapsed(true));
assert.ok(l2.nodes.length>l1.nodes.length,'L2 actually reveals Pro role details');
assert.ok(!l1.nodes.some(n=>n.id.endsWith('/router'))&&l2.nodes.some(n=>n.id.endsWith('/router')));
assert.ok(l2.nodes.filter(n=>/\/(csa|hca)$/.test(n.id)).every(n=>!n.collapsed),'no inert attention expand buttons');
console.log('PASS Pro graph: distinct L1/L2, shared renderer, valid edges, no Flash internals/source claims.');
