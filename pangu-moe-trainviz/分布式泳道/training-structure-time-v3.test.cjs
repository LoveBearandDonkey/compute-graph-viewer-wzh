// Semantic regression tests against the actual page controller; no browser packages needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const fixture = require('./generate-training-data.cjs').generate();
const schema = require('./training-timeline.schema.json');
const contract = require('./timeline-contract.js');
(async () => {
const elements = new Map(), messages = [], windowHandlers = {}, scheduled = [], paintedText=[], tooltips=[], dependencyCurves=[], drawnTasks=[];
const ctx = new Proxy({
  measureText: value => ({ width: String(value).length * 6 }),
  fillText:(text,x,y)=>paintedText.push({text,x,y}),
  bezierCurveTo:(...points)=>dependencyCurves.push(points)
}, { get: (o, key) => o[key] ?? (() => {}) });
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, value: '', hidden: true, style: {}, dataset: {}, handlers: {}, innerHTML: '',
    clientWidth: 760, clientHeight: 900, scrollLeft: 0, scrollTop: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener(type, fn) { this.handlers[type] = fn; },
    setAttribute(key, value) { this[key] = value; }, getAttribute(key) { return this[key]; },
    querySelectorAll() { return []; }, querySelector() { return element('option'); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 440, height: 200 }; },
    contains() { return false; }, getContext() { return ctx; },
    contentWindow: { postMessage(...args) { messages.push(args); } }
  });
  return elements.get(id);
}
const sandbox = { console, TimelineContract: contract, fetch: async url => ({ok:true,json:async()=>url.includes('schema')?schema:fixture}), location: { origin: 'http://localhost' }, innerWidth: 1280, innerHeight: 900,
  document: { getElementById: element, documentElement: { dataset: { theme: 'dark' } }, addEventListener() {} },
  DeepSeekV4ArchitectureData: { describeLayer: () => ({ attentionType: 'csa', routerType: 'moe' }), attentionTypeForLayer: () => 'csa' },
  PtoSwimlaneTaskPattern: { createTaskColormap: () => ({ colorForTask: () => '#4369ef' }), drawTaskBar(context,options) {drawnTasks.push({alpha:context.globalAlpha,...options});}, initHoverTooltip: options => {tooltips.push(options);return { tooltip: {} };}, hideTooltip() {}, showTooltip() {} },
  getComputedStyle: () => ({ getPropertyValue: () => '#ffffff' }),
  requestAnimationFrame: fn => { scheduled.push(fn); return scheduled.length; },
  ResizeObserver: class { observe() {} }, addEventListener: (key, fn) => { windowHandlers[key] = fn; }
};
sandbox.window = sandbox;
vm.runInNewContext(fs.readFileSync(__dirname + '/training-structure-time-v3.js', 'utf8'), sandbox);
await sandbox.TrainingStructureReady;
const api = sandbox.TrainingStructureDemo;
const change = (id, value) => { element(id).value = value; element(id).onchange({ target: element(id) }); };
const flush = () => { while (scheduled.length) scheduled.shift()(); };
const clickHit = predicate => {
  flush(); const hit = api.getHits().find(predicate); assert.ok(hit, 'visible hit found');
  element('timeline').handlers.click({ clientX: hit.x + Math.min(2, hit.w / 2), clientY: hit.y + 2 });
};
const selectNode = nodeId => windowHandlers.message({ source: element('model-graph').contentWindow, origin: sandbox.location.origin, data: { type: 'csa-operator-select', nodeId } });
assert.equal(api.getState().dataset,fixture.id);assert.equal(api.getState().dp,'all');assert.equal(api.getState().mb,'all');
windowHandlers.message({source:element('model-graph').contentWindow,origin:sandbox.location.origin,data:{type:'csa-operator-ready'}});
assert.ok(messages.some(([message])=>message.type==='csa-runtime-graph'&&message.runtimeGraph.id===fixture.runtimeGraph.id),'validated runtime graph is sent to the model pane');
assert.match(element('deployment').dataset.filterTip,/默认状态/);assert.match(element('microbatch').dataset.filterTip,/默认状态/);
const filterTip=tooltips.find(t=>String(t.targets).includes('#deployment'));
assert.match(filterTip.getTooltipHtml(element('deployment')),/部署范围为默认状态/);
assert.ok(api.getRows().some(r=>r.id==='dp1'));
flush();
const rankHit=api.getHits().find(h=>h.row?.id==='dp0/pp0/r0');
assert.equal(rankHit.expandRect.y+rankHit.expandRect.h/2,rankHit.labelCenterY,'expand aligned with Rank title');
assert.ok(paintedText.some(t=>t.text==='Rank 0'&&t.y===rankHit.labelCenterY),'Rank title untruncated');
const canvasTip=tooltips.find(t=>t.root===element('timeline'));
assert.match(canvasTip.getTooltipHtml(rankHit),/逻辑训练进程/);
assert.match(canvasTip.getTooltipHtml({header:true}),/DP → PP/);
const modeTip=tooltips.find(t=>t.root===element('time-mode'));
assert.match(modeTip.getTooltipHtml({mode:'union'}),/不跨 Rank 或 MB/);
assert.match(modeTip.getTooltipHtml({mode:'events'}),/逐条展示/);
assert.equal(element('selection-context').hidden,true,'no default status strip');
assert.ok(!canvasTip.getTooltipHtml(rankHit).includes('<'),'brief header tooltip is one sentence');
assert.ok(!modeTip.getTooltipHtml({mode:'union'}).includes('<'),'brief mode tooltip has no title or divider');
assert.equal(api.displayEvents(api.getRows().find(r=>r.id==='dp0')).length,0,'expanded DP header does not overlay ranks');
clickHit(h=>h.row?.id==='dp0/pp0/r0');
const holdRow=api.getRows().find(r=>r.id==='dp0/pp0/r0/hold');
assert.ok(holdRow);
assert.equal(api.displayEvents(api.getRows().find(r=>r.id==='dp0/pp0/r0')).length,0,'expanded Rank header does not duplicate category events');
assert.ok(api.displayEvents(api.getRows().find(r=>r.id==='dp0/pp0/r0/comm')).every(e=>e.kind==='comm'),'one communication category row');
flush();assert.ok(api.getHits().some(h=>h.event?.kind==='optimizer'&&h.w>=5),'optimizer remains visible at full-step scale');
assert.ok(api.getHits().some(h=>h.event?.kind==='comm'&&h.w>=5),'communication remains visible at full-step scale');
assert.ok(!api.displayEvents(api.getRows().find(r=>r.id==='dp0/pp0/r0')).some(e=>e.kind==='hold'),'activation only in its detail row');
const holdLayout=api.layoutEvents(holdRow,.2);
assert.ok(new Set(holdLayout.map(p=>p.track)).size>1,'concurrent activation intervals split into tracks');
const forwardLayout=api.layoutEvents(api.getRows().find(r=>r.id==='dp0/pp0/r0/forward'),.2);
assert.equal(new Set(forwardLayout.map(p=>p.track)).size,1,'serial compute stays on one track even when zoomed out');
for(const row of api.getRows()) {
  const placed=api.layoutEvents(row,.2);
  for(let i=0;i<placed.length;i++)for(let j=0;j<i;j++) {
    const a=placed[i],b=placed[j];
    if(a.track!==b.track)continue;
    const ax=a.event.start*.2,bx=b.event.start*.2;
    const ae=a.event.end>a.event.start?a.event.end*.2:ax+3,be=b.event.end>b.event.start?b.event.end*.2:bx+3;
    assert.ok(ae<=bx+1e-8||be<=ax+1e-8,'no rendered overlap on a subtrack');
  }
}
clickHit(h=>h.row?.id==='dp0/pp0/r0');
change('deployment','0');
assert.match(element('deployment').dataset.filterTip,/已限定为 DP0/,'non-default select exposes its active state');
assert.equal(api.events.length,fixture.events.length);assert.ok(api.events.length<1500);assert.equal(api.dataset.training.microbatchesPerDP,8);
assert.equal(api.getRows()[0].id,'dp0');assert.equal(api.getRows().filter(r=>/^dp0\/pp\d$/.test(r.id)).length,4);
assert.ok(!api.getRows().some(r=>r.id==='mb'));
assert.ok(api.getRows().flatMap(api.displayEvents).some(e=>e.isSummary));
const initialRange=JSON.stringify(api.range()),initialCount=api.selectedEvents().length;
change('microbatch','MB01');assert.ok(api.selectedEvents().length>0&&api.selectedEvents().length<initialCount);
assert.equal(JSON.stringify(api.range()),initialRange);
assert.ok(api.getRows().flatMap(api.displayEvents).some(e=>e.microbatchId==='MB00'));
element('clear').onclick();assert.equal(api.getState().mb,'MB01');
element('clear-mb').onclick();assert.equal(api.getState().mb,'all');
clickHit(h=>h.event?.isSummary&&h.event.kind==='forward');assert.equal(api.getState().mb,'all');
const summaryState=api.getState(),summaryEvent=api.getHits().find(h=>h.event?.id===summaryState.eventKey)?.event;
assert.ok(summaryState.eventSourceIds.length>1,'summary selection retains all constituent events');
assert.match(api.structureSummary(summaryEvent,summaryEvent.sourceEventIds),/调度摘要；包含 \d+ 条阶段与算子明细/);
assert.ok(!api.structureSummary(summaryEvent,summaryEvent.sourceEventIds).includes('dv4/'),'详情不倾倒内部结构 ID');
const directLinks=api.eventConnections(summaryState.eventSourceIds);
assert.ok(directLinks.upstream.size+directLinks.downstream.size>0,'summary exposes direct dependencies');
drawnTasks.length=0;dependencyCurves.length=0;flush();
assert.ok(dependencyCurves.length>0,'selected event draws visible dependency connectors');
assert.ok(drawnTasks.some(task=>task.alpha===1&&task.isRelated),'direct dependencies stay fully visible');
assert.ok(drawnTasks.some(task=>task.alpha===.42),'events outside the selected dependency neighborhood remain visibly dimmed with the design-system disabled opacity');
const [from,to]=api.connectionEndpoints({x:10,y:20,w:30,h:12},{x:80,y:50,w:20,h:16});
assert.deepEqual(JSON.parse(JSON.stringify([from,to])),[{x:40,y:26},{x:80,y:58}],'connection uses the vertical centers of event end/start edges');
const [reverseFrom,reverseTo]=api.connectionEndpoints({x:80,y:50,w:20,h:16},{x:10,y:20,w:30,h:12});
assert.deepEqual(JSON.parse(JSON.stringify([reverseFrom,reverseTo])),[{x:100,y:58},{x:10,y:26}],'connection always leaves source OUT on the right and enters target IN on the left');
assert.ok(messages.some(([message])=>message.type==='csa-lineage-focus'&&message.nodeIds.length),'timeline selection visually links back to model nodes');
const selectionBeforeBlank=JSON.stringify(api.getState().selection);
element('timeline').handlers.click({clientX:700,clientY:10});
assert.equal(api.getState().eventId,null,'timeline blank clears the event selection');
assert.equal(api.getState().eventSourceIds.length,0,'timeline blank clears dependency highlighting');
assert.equal(JSON.stringify(api.getState().selection),selectionBeforeBlank,'timeline blank preserves independent structure filters');
assert.ok(!fs.readFileSync(__dirname+'/training-structure-time-v3.js','utf8').includes("tipRow('数据性质'"),'详情移除多余的数据性质行');
clickHit(h=>h.event?.isSummary&&h.event.kind==='forward');
element('trace').onclick();assert.notEqual(api.getState().mb,'all');
element('clear-mb').onclick();change('deployment','all');assert.equal(api.selectedEvents().length,fixture.events.length);
change('deployment','1');
const sync=api.events.find(e=>e.label==='Dense Gradient Reduce-Scatter'&&e.stage===0);
assert.ok(api.getRows().find(r=>r.id==='dp1/pp0/r8').list.some(e=>e.id===sync.id),'cross DP participant projected');
selectNode('runtime/optimizer-step');
assert.ok(api.selectedEvents().length>0&&api.selectedEvents().every(e=>e.runtimeNodeIds.includes('runtime/optimizer-step')),'runtime node selects its optimizer events');
change('deployment','0');selectNode('dv4/layer/2/mhc_attn/rmsnorm');
assert.equal(api.selectedEvents().length,8*2*2);
assert.ok(api.displayEvents(api.getRows().find(r=>r.id==='dp0/pp1/r2')).length>0,'structure selection keeps non-matching Rank context visible');
for(const nodeId of [
  'dv4/model/input_tokens','dv4/model/token_embedding','dv4/model/previous_decoder_layers','dv4/layer/2',
  'dv4/layer/2/input_streams','dv4/layer/2/mhc_attn/residual_mix_b','dv4/layer/2/mhc_attn/hybrid_attention',
  'dv4/layer/2/attention_output_streams','dv4/layer/2/mhc_ffn/residual_mix_b','dv4/layer/2/mhc_ffn/deepseek_moe',
  'dv4/layer/2/output_streams','dv4/model/remaining_decoder_layers','dv4/model/final_rmsnorm','dv4/model/lm_head',
  'dv4/model/output_tokens','dv4/model/mtp','dv4/model/loss'
]) { selectNode(nodeId);assert.ok(api.selectedEvents().length>0,`visible Pro node ${nodeId} has timeline events`); }
selectNode('dv4/layer/2/mhc_attn/rmsnorm');
const selection=JSON.stringify(api.getState().selection),sourceIds=new Set(api.selectedEvents().map(e=>e.id));
element('time-mode').onclick({target:{closest:()=>({dataset:{mode:'union'}})}});
for(const row of api.getRows())for(const e of api.displayEvents(row)) {
  const selected=e.sourceEventIds.filter(id=>sourceIds.has(id));
  assert.ok(!selected.length||selected.length===e.sourceEventIds.length,'union cannot include unrelated events');
}
clickHit(h=>h.event?.isUnion&&h.event.sourceEventIds.every(id=>sourceIds.has(id)));
assert.equal(api.getState().selection,null,'timeline event selection replaces the structure filter with visual graph lineage');assert.equal(api.getState().mb,'all');
element('timeline').handlers.click({clientX:700,clientY:10});
assert.equal(api.getState().eventId,null);
assert.equal(api.getState().selection,null,'blank deselection keeps the event-driven graph lineage separate from structure filters');
windowHandlers.message({source:element('model-graph').contentWindow,origin:sandbox.location.origin,data:{type:'csa-operator-toggle',source:'expansion',nodeId:'x'}});
assert.equal(api.getState().selection,null,'graph expansion does not recreate a cleared structure filter');
const optimizer=api.events.find(e=>e.referenceEventCategory==='optimizer');
api.focusGraph(optimizer);
assert.ok(messages.some(([message])=>message.type==='csa-lineage-focus'&&message.nodeIds.includes('runtime/optimizer-step')),'optimizer event focuses the runtime node rather than a fake model operator');
assert.match(api.structureSummary(optimizer,[optimizer]),/Optimizer Step/);
change('view','stage');change('stage','1');
const pp=api.events.find(e=>e.communicationType==='pp'&&e.stage===0);
assert.ok(api.getRows().find(r=>r.id==='dp0/pp1/r2').list.some(e=>e.id===pp.id));
assert.ok(api.getRows().every(r=>!r.id.includes('/pp')||r.id.includes('/pp1')));
console.log('PASS UI controller: JSON load, full-step defaults, 8 MB, L2-only operator selection, summaries, dependency links, blank deselection, independent focus, union, DP/PP projections.');
})().catch(e=>{console.error(e);process.exitCode=1;});
