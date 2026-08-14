import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {buildSimulated1F1BRuntime} from './analysis-data.js';

const pageUrl=new URL('../op-rank-time-openpangu-flash-events.html',import.meta.url);
const dataUrl=new URL('../data/profiling-task-drilldown-mock.json',import.meta.url);
const swimPatternCssUrl=new URL('../vendor/swimlane-task/pattern.css',import.meta.url);
const html=fs.readFileSync(pageUrl,'utf8');
const mock=JSON.parse(fs.readFileSync(dataUrl,'utf8'));
const swimPatternCss=fs.readFileSync(swimPatternCssUrl,'utf8');

const taskEnd=span=>Number(span.startUs)+Number(span.durUs);
const selectorKey=selector=>`${selector.stage}:${selector.microbatch}:${selector.phase}`;

test('limited profiling mock targets exactly the first three compute schedule slots',()=>{
  assert.equal(mock.schema,'pto.mock-profiling-drilldown.v1');
  assert.equal(mock.source,'mock-profile-json');
  assert.equal(mock.fidelity,'limited-sampled-task-drilldown');
  assert.equal(mock.tasks.length,3);

  const rt=buildSimulated1F1BRuntime({
    dp:2,pp:4,tp:2,ep:2,microbatches:8,
    stageRanges:[[0,11],[12,22],[23,34],[35,45]]
  });
  const representatives=Array.from({length:4},(_,stage)=>
    rt.ranks.find(rank=>rank.dp===0&&rank.stage===stage&&rank.tp===0&&rank.ep===0)
  );
  const earliest=representatives.flatMap((rank,stage)=>(rank?.tasks||[])
    .filter(task=>task.kind==='F'||task.kind==='B')
    .map(task=>({stage,microbatch:task.microbatch,phase:task.kind,startUs:task.startUs,durUs:task.durUs})))
    .sort((a,b)=>a.startUs-b.startUs)
    .slice(0,3);

  assert.deepEqual(mock.tasks.map(task=>selectorKey(task.selector)),earliest.map(selectorKey));
  mock.tasks.forEach((task,index)=>{
    assert.ok(Math.abs(task.startUs-earliest[index].startUs)<0.001);
    assert.ok(Math.abs(task.durUs-earliest[index].durUs)<0.001);
  });
});

test('each sampled task contains bounded operator, kernel and collective evidence',()=>{
  const ids=new Set();
  let kernelCount=0,collectiveCount=0;
  for(const task of mock.tasks){
    assert.ok(!ids.has(task.id));ids.add(task.id);
    assert.ok(task.durUs>0);
    assert.ok(task.operators.length>=7);
    const parentEnd=taskEnd(task);
    for(const operator of task.operators){
      assert.ok(!ids.has(operator.id));ids.add(operator.id);
      assert.equal(operator.kind,'operator');
      assert.ok(Number.isInteger(operator.layer));
      assert.ok(operator.nodeId);
      assert.ok(operator.startUs>=task.startUs);
      assert.ok(taskEnd(operator)<=parentEnd+0.001);
      assert.ok(operator.correlationId);
      for(const kernel of operator.kernels||[]){
        kernelCount++;
        assert.ok(!ids.has(kernel.id));ids.add(kernel.id);
        assert.equal(kernel.kind,'kernel');
        assert.ok(Number.isInteger(kernel.taskId));
        assert.ok(Number.isInteger(kernel.streamId));
        assert.ok(kernel.startUs>=operator.startUs);
        assert.ok(taskEnd(kernel)<=taskEnd(operator)+0.001);
      }
      for(const collective of operator.collectives||[]){
        collectiveCount++;
        assert.ok(!ids.has(collective.id));ids.add(collective.id);
        assert.equal(collective.kind,'collective');
        assert.ok(collective.eventKind);
        assert.ok(collective.primitive);
        assert.ok(Array.isArray(collective.participants)&&collective.participants.length>=2);
        assert.ok(collective.startUs>=operator.startUs);
        assert.ok(taskEnd(collective)<=taskEnd(operator)+0.001);
      }
    }
  }
  assert.ok(kernelCount>=20);
  assert.ok(collectiveCount>=4);
});

test('page loads, normalizes and exposes the imported mock profiling contract',()=>{
  assert.match(html,/const PROFILE_DRILLDOWN_URL='data\/profiling-task-drilldown-mock\.json'/);
  assert.match(html,/function normalizeProfilingTaskDefinition\(definition\)/);
  assert.match(html,/const PROFILING_TASK_BY_SELECTOR=new Map/);
  assert.match(html,/window\.OPENPANGU_PROFILE_DRILLDOWN=/);
  assert.match(html,/bodyCanvas\.dataset\.profileDrilldown=selectedProfilingTask\?\.id\|\|'collapsed'/);
});

test('eligible task click expands before normal seek/event handling and child click selects detail',()=>{
  const clickStart=html.indexOf("bodyCanvas.addEventListener('click'");
  const clickEnd=html.indexOf("bodyCanvas.addEventListener('pointermove'",clickStart);
  const clickHandler=html.slice(clickStart,clickEnd);
  assert.ok(clickStart>=0&&clickEnd>clickStart);
  assert.ok(clickHandler.indexOf('if(hit?.profileSpan)')<clickHandler.indexOf('const spec=catalogEventForTask'));
  assert.ok(clickHandler.indexOf('toggleProfilingTaskDrilldown(profilingTask)')<clickHandler.indexOf('const spec=catalogEventForTask'));
  assert.match(clickHandler,/selectProfilingSpan\(hit\.profileTask,hit\.profileSpan\)/);
});

test('blank Swimlane click clears event focus without collapsing the expanded task',()=>{
  const clickStart=html.indexOf("bodyCanvas.addEventListener('click'");
  const clickEnd=html.indexOf("bodyCanvas.addEventListener('pointermove'",clickStart);
  const clickHandler=html.slice(clickStart,clickEnd);
  const clearStart=html.indexOf('function clearSwimlaneEventSelection');
  const clearEnd=html.indexOf('function toggleProfilingTaskDrilldown',clearStart);
  const clearHandler=html.slice(clearStart,clearEnd);

  assert.match(clickHandler,/if\(!hit\)\{\s*clearSwimlaneEventSelection\(\);\s*seek\(event\);\s*return;/);
  assert.ok(clickHandler.indexOf('if(!hit)')<clickHandler.indexOf('if(hit?.profileSpan)'));
  assert.match(clearHandler,/selectedProfilingSpan=null/);
  assert.match(clearHandler,/clearParallelEventSelection\(\{renderInspector:false\}\)/);
  assert.match(clearHandler,/interactionStore\.clearSelection\(\{source:'analysis'\}\)/);
  assert.match(clearHandler,/\['profiling-span','parallel-event','model'\]\.includes\(selectedDetailKind\)/);
  assert.match(clearHandler,/setSelectedArch\(null,\{openInspector:false\}\)/);
  assert.doesNotMatch(clearHandler,/selectedProfilingTask=null/);
});

test('expanded row renders local model-op, device-kernel and collective lanes with Inspector linkage',()=>{
  assert.match(html,/function drawProfilingTaskDrilldown\(ctx,\{task,row,top,W\}\)/);
  assert.match(html,/const localX=value=>left\+clampValue/);
  assert.match(html,/\{label:'模型算子',spans:task\.operators\}/);
  assert.match(html,/\{label:'设备 Kernel',spans:task\.kernels\}/);
  assert.match(html,/\{label:'集合通信',spans:task\.collectives\}/);
  assert.match(html,/drawProfilingTaskCue\(ctx,x,y,w,h,selectedProfilingTask\?\.id===profileTask\.id,profilingSpanFocus\(\)/);
  assert.match(html,/else if\(selectedDetailKind==='profiling-span'&&selectedProfilingTask&&selectedProfilingSpan\)renderProfilingSpanInspector\(\)/);
  assert.match(html,/setSelectedArch\(mesh,\{openInspector:false\}\)/);
});

test('swimlane drilldown copy reads like a product surface instead of prototype notes',()=>{
  const inspectorStart=html.indexOf('function renderProfilingSpanInspector()');
  const inspectorEnd=html.indexOf('function renderSelectedDetailInspector()',inspectorStart);
  const drilldownStart=html.indexOf('function drawProfilingTaskDrilldown');
  const drilldownEnd=html.indexOf('function profilingSpanTooltip',drilldownStart);
  const tooltipStart=html.indexOf('function profilingSpanTooltip');
  const tooltipEnd=html.indexOf('function swimRowTooltip',tooltipStart);
  const swimTooltipStart=html.indexOf('function swimRowTooltip');
  const swimTooltipEnd=html.indexOf('function catalogEventForTask',swimTooltipStart);
  const visibleCopy=html.slice(inspectorStart,inspectorEnd)+html.slice(drilldownStart,drilldownEnd)+html.slice(tooltipStart,tooltipEnd)+html.slice(swimTooltipStart,swimTooltipEnd);

  assert.match(visibleCopy,/内置示例 Trace/);
  assert.match(visibleCopy,/模型算子/);
  assert.match(visibleCopy,/设备 Kernel/);
  assert.match(visibleCopy,/集合通信/);
  assert.match(visibleCopy,/阶段内时间/);
  assert.doesNotMatch(visibleCopy,/mock profile JSON|mock JSON|limited-sampled|点击后会|点击子事件|no sampled event/);
});

test('light-theme Swimlane tooltip uses the requested supported panel background',()=>{
  assert.match(swimPatternCss,/background: var\(--pto-swimlane-tooltip-bg, var\(--surface-3\)\);/);
  assert.match(html,/\.\/vendor\/swimlane-task\/pattern\.css\?v=20260716-tooltip-bg/);
  assert.match(html,/\.\/vendor\/swimlane-task\/pattern\.js\?v=20260716-tooltip-bg/);
  assert.match(html,/:root\[data-theme='light'\] #swimlaneView\{--pto-swimlane-tooltip-bg:#F8F8F8\}/);
});

test('profiling child selection mutes unrelated timeline events and model nodes',()=>{
  assert.match(html,/const profilingSpanFocus=\(\)=>selectedDetailKind==='profiling-span'/);
  assert.match(html,/function profilingTaskMatchesFocus\(task,row\)/);
  assert.match(html,/if\(profilingSpanFocus\(\)&&!profilingTaskMatchesFocus\(task,row\)\)return mutedSwimTaskColor\(\)/);
  assert.match(html,/const swimSelectionFocus=\(\)=>parallelEventSwimFocus\(\)\|\|profilingSpanFocus\(\)/);
  assert.match(html,/const related=profilingSpanRelated\(span\)/);
  assert.match(html,/function syncProfilingSpanSelection\(\)/);
  assert.match(html,/is-profiling-span-focused/);
  assert.match(html,/is-profile-linked/);
  assert.match(html,/path\.dataset\.source===selectedId\|\|path\.dataset\.target===selectedId/);
  assert.match(html,/parallelEventOverlay\?\.classList\.add\('is-profiling-span-focused'\)/);
  assert.match(html,/interactionStore\.setSelection\(\{type:'profiling-span'/);
});
