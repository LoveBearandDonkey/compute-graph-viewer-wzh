const assert=require('node:assert/strict'),fs=require('node:fs');
const {generate,reference}=require('./generate-training-data.cjs');
const {validate}=require('./timeline-contract.js'),schema=require('./training-timeline.schema.json');
const c=require('./data/training-simulation.config.json'),d=generate();
validate(d,schema);
assert.equal(JSON.stringify(d),JSON.stringify(generate()),'deterministic');
assert.equal(JSON.stringify(d),fs.readFileSync(__dirname+'/data/deepseek-v4-pro.timeline.json','utf8').trim(),'artifact up to date');
assert.equal(d.training.microbatchesPerDP,32/2/2);assert.equal(d.training.world,16);
assert.equal(d.schemaVersion,'training-timeline.v2');
assert.ok(d.events.length<1500,'demo stays small');
assert.deepEqual(d.coverage.detailedLayers,[2]);
assert.deepEqual([...new Set(d.events.filter(e=>e.layer!==null).map(e=>e.layer))],[2]);
assert.ok(d.events.some(e=>e.granularity==='aggregate'&&e.layerRange),'coarse stage context retained');
assert.equal(d.model.compressRatios.length,62);assert.equal(d.model.compressRatios[60],4);assert.equal(d.model.compressRatios[61],0);
assert.equal(reference().pairs.length,32);assert.ok(reference().backwardForwardRatio>2&&reference().backwardForwardRatio<2.2);
assert.deepEqual(reference().optimizerPhases.map(phase=>phase.category),['grads-reduce-scatter','layernorm-grads-all-reduce','optimizer-clip-main-grad','optimizer','params-all-gather']);
assert.ok(reference().optimizerPhases.every(phase=>phase.count===8&&phase.sourceIndices.length===8),'ST optimizer lifecycle evidence retained');
assert.equal(d.parameterGroups.length,8);assert.equal(d.optimizerStateGroups.length,8);assert.equal(d.collectiveGroups.length,12);
assert.deepEqual(d.runtimeGraph.nodes.map(node=>node.id),['runtime/gradient-reduce','runtime/gradient-clip','runtime/optimizer-step','runtime/parameter-allgather']);
const mbKeys=new Map();
for(const e of d.events.filter(e=>e.kind==='forward'||e.kind==='backward')) {
  const k=`${e.rank}/${e.microbatchId}/${e.kind}`;if(!mbKeys.has(k))mbKeys.set(k,new Set());if(e.layer!==null)mbKeys.get(k).add(e.layer);
}
assert.equal(mbKeys.size,16*8*2);
for(const [key,layers]of mbKeys) {
  const rank=Number(key.split('/')[0]),stage=d.ranks.find(r=>r.rank===rank).stage;
  const [first,last]=c.stageLayerRanges[stage];assert.equal(layers.size,c.detailLayer>=first&&c.detailLayer<=last?1:0);
}
for(const r of d.ranks) {
  const compute=d.events.filter(e=>e.rank===r.rank&&['forward','backward','optimizer'].includes(e.kind)).sort((a,b)=>a.start-b.start);
  for(let i=1;i<compute.length;i++)assert.ok(compute[i].start>=compute[i-1].end-1e-8,`compute overlap R${r.rank}`);
}
const order=s=>d.summaries.filter(e=>e.dp===0&&e.stage===s&&e.ep===0).sort((a,b)=>a.start-b.start).map(e=>e.label).join(' ');
assert.equal(order(0),'F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7');
assert.equal(order(3),'F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7');
assert.equal(d.events.filter(e=>e.label.startsWith('输出头')&&e.kind==='forward').length,2*2*8);
assert.ok(d.events.filter(e=>e.runtimeNodeIds.length).every(e=>e.microbatchId===null));
assert.ok(d.events.some(e=>e.referenceEventCategory==='grads-reduce-scatter'&&e.collectiveGroupId));
assert.ok(d.events.some(e=>e.referenceEventCategory==='optimizer-clip-main-grad'&&e.runtimeNodeIds.includes('runtime/gradient-clip')));
assert.ok(d.events.some(e=>e.referenceEventCategory==='params-all-gather'&&e.runtimeNodeIds.includes('runtime/parameter-allgather')));
assert.ok(d.events.filter(e=>e.referenceEventCategory==='optimizer').every(e=>e.parameterGroupIds.length===2&&e.optimizerStateGroupIds.length===2&&e.shards.length===2));
const allBackwardEnd=Math.max(...d.events.filter(e=>e.kind==='backward').map(e=>e.end));
assert.ok(d.events.filter(e=>e.kind==='optimizer').every(e=>e.start>allBackwardEnd));
const invalid=structuredClone(d);invalid.events[0].dependsOn=[d.events.at(-1).id];assert.throws(()=>validate(invalid,schema),/causal/);
const missing=structuredClone(d);delete missing.events[0].microbatchId;assert.throws(()=>validate(missing,schema),/missing/);
const unknown=structuredClone(d);unknown.events[0].participants=[999];assert.throws(()=>validate(unknown,schema),/participants/);
assert.throws(()=>generate({...c,globalBatchSize:33}),/Unsupported/);
const changed=generate({...c,globalBatchSize:16});assert.equal(changed.training.microbatchesPerDP,4);validate(changed,schema);
console.log('PASS producer: lightweight single-layer demo, coarse context, schema + causal graph, 1F1B, no compute overlap, batch-derived counts, deterministic artifact.');
