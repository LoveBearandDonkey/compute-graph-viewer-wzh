// Offline producer. stdout is the reproducible JSON artifact; no browser code is imported.
const crypto = require('node:crypto');
const model = require('./data/deepseek-v4-pro.config.json');
const defaults = require('./data/training-simulation.config.json');
const st = require('./st-step23.js');
const median = values => { const a = [...values].sort((a,b)=>a-b); return (a[Math.floor((a.length-1)/2)] + a[Math.floor(a.length/2)])/2; };
function reference() {
  const forward = new Map(st.events.filter(e=>e.name==='forward-compute').map(e=>[`${e.pid}/${e.tid}/${e.args.mb_id}`,e]));
  const pairs = st.events.filter(e=>e.name==='backward-compute').map(e=>({ backward:e, forward:forward.get(`${e.pid}/${e.tid}/${e.args.mb_id}`) }));
  const optimizerCategories=['grads-reduce-scatter','layernorm-grads-all-reduce','optimizer-clip-main-grad','optimizer','params-all-gather'];
  const optimizerPhases=optimizerCategories.map(category=>{
    const records=st.events.filter(e=>e.name===category);
    return {category,count:records.length,sourceIndices:records.map(e=>e.sourceIndex),medianDurationUs:median(records.map(e=>e.dur)),
      fields:['pid','tid','ts','dur','args.step','args.model_chunk','args.mb_id','args.seq_id']};
  });
  return { source:st.source, sourceSha256:st.sourceSha256, step:23,
    backwardForwardRatio:median(pairs.map(p=>p.backward.dur/p.forward.dur)),
    method:'Median of 32 matched (DP, Rank, MB) backward/forward duration ratios. Absolute latency and stage imbalance are not transferred.',
    pairs:pairs.map(p=>({forwardIndex:p.forward.sourceIndex, backwardIndex:p.backward.sourceIndex, forwardUs:p.forward.dur, backwardUs:p.backward.dur})),
    optimizerPhases };
}
function generate(config = defaults) {
  const c = structuredClone(config), ref = reference();
  const M = c.globalBatchSize / (c.dp*c.microBatchSize);
  if (!Number.isInteger(M) || M < c.pp || c.tp!==1 || c.ep!==2 || c.microBatchSize%c.ep || c.schedule!=='non-interleaved-1f1b' || c.recompute!==false) throw Error('Unsupported simulation config');
  if (!Number.isInteger(c.detailLayer) || c.detailLayer<0 || c.detailLayer>=model.numHiddenLayers) throw Error('Invalid detail layer');
  if (JSON.stringify(c.stageLayerRanges.flatMap(([a,b])=>Array.from({length:b-a+1},(_,i)=>a+i))) !== JSON.stringify(Array.from({length:model.numHiddenLayers},(_,i)=>i))) throw Error('PP partition must cover every main layer exactly once');
  const data = { schemaVersion:'training-timeline.v2', id:c.id, label:model.label, step:c.step, fidelity:'simulated', timeUnit:'ms', model,
    training:{...c,microbatchesPerDP:M,world:c.dp*c.pp*c.tp*c.ep}, calibration:ref,
    coverage:{detailedLayers:[c.detailLayer],coarseStageContext:true,kind:'representative-layer-demo',kernelTrace:false,hardwareFeasibility:'not-evaluated'},
    stages:c.stageLayerRanges.map(([first,last],id)=>({id,first,last})), ranks:[], parameterGroups:[], optimizerStateGroups:[], collectiveGroups:[], runtimeGraph:null,
    events:[], summaries:[], stepBounds:{start:0,end:0},
    provenance:[{id:'model-config',kind:'official-architecture',url:model.source},{id:'st-ratio',kind:'measured-reference',url:ref.source,sha256:ref.sourceSha256},
      {id:'st-optimizer-phases',kind:'measured-reference-shape',url:ref.source,sha256:ref.sourceSha256,categories:ref.optimizerPhases.map(p=>p.category)},
      {id:'simulation-config',kind:'assumption',path:'data/training-simulation.config.json'}] };
  const rank = (dp,pp,ep) => (dp*c.pp+pp)*c.ep+ep;
  for(let dp=0;dp<c.dp;dp++)for(let pp=0;pp<c.pp;pp++)for(let ep=0;ep<c.ep;ep++)data.ranks.push({rank:rank(dp,pp,ep),dp,stage:pp,tp:0,ep});
  const denseModelNodes=stage=>stage===0
    ? ['dv4/model/token_embedding','dv4/model/previous_decoder_layers','dv4/layer/2/mhc_attn','dv4/layer/2/mhc_ffn/deepseek_moe/shared_expert']
    : stage===c.pp-1
      ? ['dv4/model/remaining_decoder_layers','dv4/model/final_rmsnorm','dv4/model/lm_head','dv4/model/mtp']
      : ['dv4/model/remaining_decoder_layers'];
  const expertModelNodes=stage=>stage===0?['dv4/layer/2/mhc_ffn/deepseek_moe/routed_experts']:['dv4/model/remaining_decoder_layers'];
  for(let stage=0;stage<c.pp;stage++) {
    const layerRange=[...c.stageLayerRanges[stage]];
    data.parameterGroups.push({id:`pg/pp${stage}/dense`,label:`PP${stage} Dense / Norm / Head`,stage,layerRange,parameterClass:'dense',modelNodeIds:denseModelNodes(stage),
      distribution:{mode:'replicated',replicatedAxes:['dp','ep'],shardedAxes:[],shardCount:1},fidelity:'simulated',provenanceIds:['model-config','simulation-config']});
    data.parameterGroups.push({id:`pg/pp${stage}/routed-experts`,label:`PP${stage} Routed Experts`,stage,layerRange,parameterClass:'routed-expert',modelNodeIds:expertModelNodes(stage),
      distribution:{mode:'hybrid',replicatedAxes:['dp'],shardedAxes:['ep'],shardCount:c.ep},fidelity:'simulated',provenanceIds:['model-config','simulation-config']});
    const denseRanks=data.ranks.filter(r=>r.stage===stage).map(r=>r.rank);
    data.collectiveGroups.push({id:`collective/pp${stage}/dense`,label:`PP${stage} Dense Gradient Group`,kind:'dense-gradient',ranks:denseRanks,axes:['dp','ep'],fidelity:'simulated',provenanceIds:['simulation-config']});
    for(let ep=0;ep<c.ep;ep++) {
      const ranks=data.ranks.filter(r=>r.stage===stage&&r.ep===ep).map(r=>r.rank);
      data.collectiveGroups.push({id:`collective/pp${stage}/expert/ep${ep}`,label:`PP${stage} Expert Gradient Group · EP${ep}`,kind:'expert-gradient',ranks,axes:['dp'],fidelity:'simulated',provenanceIds:['simulation-config']});
    }
  }
  for(const group of data.parameterGroups) {
    const collective=group.parameterClass==='dense'
      ? data.collectiveGroups.find(g=>g.id===`collective/pp${group.stage}/dense`)
      : data.collectiveGroups.find(g=>g.id===`collective/pp${group.stage}/expert/ep0`);
    data.optimizerStateGroups.push({id:`opt-state/${group.id}`,label:`${group.label} · Optimizer State`,parameterGroupId:group.id,algorithm:null,
      distribution:{mode:'sharded',axes:[...collective.axes],shardCount:collective.ranks.length},fidelity:'simulated',provenanceIds:['st-optimizer-phases','simulation-config']});
  }
  const allParameterGroupIds=data.parameterGroups.map(g=>g.id), allOptimizerStateGroupIds=data.optimizerStateGroups.map(g=>g.id);
  data.runtimeGraph={id:'runtime/training-step',label:'训练运行时 · step 级',nodes:[
    {id:'runtime/gradient-reduce',label:'Gradient Reduce / Sync',kind:'collective',scope:'step',sourceCategories:['grads-reduce-scatter','layernorm-grads-all-reduce'],parameterGroupIds:allParameterGroupIds,optimizerStateGroupIds:[],fidelity:'simulated',provenanceIds:['st-optimizer-phases','simulation-config']},
    {id:'runtime/gradient-clip',label:'Global Norm / Clip',kind:'gradient-transform',scope:'step',sourceCategories:['optimizer-clip-main-grad'],parameterGroupIds:allParameterGroupIds,optimizerStateGroupIds:[],fidelity:'simulated',provenanceIds:['st-optimizer-phases','simulation-config']},
    {id:'runtime/optimizer-step',label:'Optimizer Step',kind:'optimizer',scope:'step',sourceCategories:['optimizer'],parameterGroupIds:allParameterGroupIds,optimizerStateGroupIds:allOptimizerStateGroupIds,fidelity:'simulated',provenanceIds:['st-optimizer-phases','simulation-config']},
    {id:'runtime/parameter-allgather',label:'Parameter AllGather',kind:'parameter-materialization',scope:'step',sourceCategories:['params-all-gather'],parameterGroupIds:allParameterGroupIds,optimizerStateGroupIds:[],fidelity:'simulated',provenanceIds:['st-optimizer-phases','simulation-config']}
  ],edges:[
    {id:'runtime/edge/loss-gradient-reduce',source:'dv4/model/loss',target:'runtime/gradient-reduce',semanticEdgeType:'gradient'},
    {id:'runtime/edge/reduce-clip',source:'runtime/gradient-reduce',target:'runtime/gradient-clip',semanticEdgeType:'gradient'},
    {id:'runtime/edge/clip-optimizer',source:'runtime/gradient-clip',target:'runtime/optimizer-step',semanticEdgeType:'control'},
    {id:'runtime/edge/optimizer-allgather',source:'runtime/optimizer-step',target:'runtime/parameter-allgather',semanticEdgeType:'parameter'}
  ]};
  const byId=new Map();
  function emit(label,kind,start,duration,placement,mb,graphNodeIds,dependsOn=[],extra={}) {
    const id=`${c.id}/e${data.events.length}`;
    const e={id,label,kind,start,end:start+duration,...placement,layer:null,microbatchId:mb,graphNodeIds,
      templateNodeIds:graphNodeIds.map(x=>x.replace(/dv4\/layer\/\d+/,'dv4/layer/2')),
      scope:mb===null?'step':'microbatch',runtimeNodeIds:[],parameterGroupIds:[],optimizerStateGroupIds:[],collectiveGroupId:null,shards:[],referenceEventCategory:null,
      stream:kind==='comm'?'collective':kind==='hold'?'lifetime':'compute', participants:[],
      relation:kind==='backward'?'backward-owner':kind==='comm'?'communication-participant':kind==='hold'?'activation-owner':'direct',
      sourceEventIds:[id],dependsOn:[...new Set(dependsOn)],fidelity:'simulated',provenanceIds:['model-config','simulation-config',...(kind==='backward'?['st-ratio']:[])],...extra};
    data.events.push(e);byId.set(id,e);return e;
  }
  const tasks=new Map(), completed=new Map(), transfers=new Map(), forwardLayers=new Map();
  const key=(dp,s,m,p)=>`${dp}/${s}/${m}/${p}`;
  for(let dp=0;dp<c.dp;dp++) for(let s=0;s<c.pp;s++) {
    const warm=Math.min(c.pp-s-1,M), sequence=[];
    for(let m=0;m<warm;m++)sequence.push([m,'F']);
    for(let m=0;m<M-warm;m++)sequence.push([m+warm,'F'],[m,'B']);
    for(let m=M-warm;m<M;m++)sequence.push([m,'B']);
    sequence.forEach(([m,p],i)=>{
      const deps=[];
      if(i)deps.push({key:key(dp,s,...sequence[i-1]),lag:0});
      if(p==='F'&&s>0)deps.push({key:key(dp,s-1,m,'F'),lag:c.ppTransferMs});
      if(p==='B') { deps.push({key:key(dp,s,m,'F'),lag:0}); if(s<c.pp-1)deps.push({key:key(dp,s+1,m,'B'),lag:c.ppTransferMs}); }
      tasks.set(key(dp,s,m,p),{dp,s,m,p,deps});
    });
  }
  while(completed.size<tasks.size) {
    let progress=false;
    for(const [id,t] of tasks) {
      if(completed.has(id)||t.deps.some(d=>!completed.has(d.key)))continue;
      progress=true;
      const {dp,s,m,p}=t, mb=`MB${String(m).padStart(2,'0')}`, owners=data.ranks.filter(r=>r.dp===dp&&r.stage===s);
      const dependencies=t.deps.flatMap(d=>{
        const previous=completed.get(d.key);
        if(!d.lag)return previous.tails;
        if(!transfers.has(d.key)) {
          const peer=tasks.get(d.key), participants=[...data.ranks.filter(r=>r.dp===dp&&(r.stage===peer.s||r.stage===s)).map(r=>r.rank)];
          const e=emit(`PP${peer.s} → PP${s} · ${p==='F'?'激活':'梯度'}`, 'comm',previous.end,d.lag,data.ranks.find(r=>r.dp===dp&&r.stage===peer.s),mb,
            [`dv4/layer/${c.stageLayerRanges[peer.s][p==='F'?1:0]}`,`dv4/layer/${c.stageLayerRanges[s][p==='F'?0:1]}`],previous.tails,{participants,communicationType:'pp',phase:p});
          transfers.set(d.key,e.id);
        }
        return [transfers.get(d.key)];
      });
      let time=Math.max(0,...dependencies.map(x=>byId.get(x).end)), tails=dependencies;
      const start=time, firstEvent=data.events.length, factor=p==='B'?ref.backwardForwardRatio:1;
      function op(label,weight,graphIds,layer=null,extra={}) {
        const generated=owners.map(owner=>emit(label,p==='F'?'forward':'backward',time,weight*c.layerForwardMs*factor,owner,mb,graphIds,tails,{layer,phase:p,operation:label,...extra}));
        time=generated[0].end; tails=generated.map(e=>e.id);
      }
      function collective(label,graphIds,layer) {
        const e=emit(label,'comm',time,c.epTransferMs,owners[0],mb,graphIds,tails,{layer,phase:p,participants:owners.map(r=>r.rank),communicationType:'ep'});
        time=e.end; tails=[e.id];
      }
      function head() {
        op(`输出头 / MTP / Loss · ${p==='F'?'前向':'反向'}概览`,1.26,
          ['final_rmsnorm','lm_head','output_tokens','mtp','loss'].map(node=>`dv4/model/${node}`),null,{granularity:'aggregate'});
      }
      if(p==='F'&&s===0)op('Token Embedding',.08,['dv4/model/input_tokens','dv4/model/token_embedding']);
      if(p==='B'&&s===c.pp-1)head();
      let layers=Array.from({length:c.stageLayerRanges[s][1]-c.stageLayerRanges[s][0]+1},(_,i)=>c.stageLayerRanges[s][0]+i);
      if(p==='B')layers.reverse();
      for(let i=0;i<layers.length;i++) {
        const l=layers[i];
        if(l!==c.detailLayer) {
          const group=[l];
          while(i+1<layers.length&&layers[i+1]!==c.detailLayer)group.push(layers[++i]);
          const range=[Math.min(...group),Math.max(...group)];
          const scope=range[1]<c.detailLayer?'dv4/model/previous_decoder_layers':'dv4/model/remaining_decoder_layers';
          op(`L${range[0]}–L${range[1]} · ${p==='F'?'前向':'反向'}概览`,group.length,[scope],null,{granularity:'aggregate',layerRange:range});
          continue;
        }
        const root=`dv4/layer/${l}`, attention=model.compressRatios[l]===4?'CSA':model.compressRatios[l]===128?'HCA':'SWA';
        const layerStart=time;
        const roles=[
          ['HC Pre · RMSNorm',.06,`${root}/mhc_attn/rmsnorm`],
          [attention,.38,`${root}/mhc_attn/hybrid_attention/${attention.toLowerCase()}`],
          ['Attention HC Post',.03,`${root}/mhc_attn/hc_post`],
          ['FFN HC Pre · RMSNorm',.04,`${root}/mhc_ffn/rmsnorm`],
          [l<model.numHashLayers?'Router · Hash IDs ×6':'Router · Top-6',.04,`${root}/mhc_ffn/deepseek_moe/router`],
          ['EP Dispatch',0,`${root}/mhc_ffn/deepseek_moe`],
          ['Routed Experts',.28,`${root}/mhc_ffn/deepseek_moe/routed_experts`],
          ['EP Combine',0,`${root}/mhc_ffn/deepseek_moe/combine`],
          ['Shared Expert',.10,`${root}/mhc_ffn/deepseek_moe/shared_expert`],
          ['Combine · HC Post',.07,`${root}/mhc_ffn/hc_post`]
        ];
        for(const [label,w,node] of p==='F'?roles:[...roles].reverse()) {
          const bindings=[node];
          if(label==='HC Pre · RMSNorm')bindings.push(`${root}/input_streams`,`${root}/mhc_attn/residual_mix_b`,`${root}/mhc_attn/hc_pre`);
          if(label==='Attention HC Post')bindings.push(`${root}/mhc_attn/merge`,`${root}/attention_output_streams`);
          if(label==='FFN HC Pre · RMSNorm')bindings.push(`${root}/mhc_ffn/residual_mix_b`,`${root}/mhc_ffn/hc_pre`);
          if(label===attention)bindings.push(`${root}/mhc_attn/hybrid_attention/attention_type`);
          if(label==='Combine · HC Post')bindings.push(`${root}/mhc_ffn/deepseek_moe/combine`,`${root}/mhc_ffn/merge`,`${root}/output_streams`);
          if(w)op(`${label} · L${l}${p==='B'?' ∇':''}`,w,bindings,l);
          else collective(`${label}${p==='B'?' ∇':''} · L${l}`,[node],l);
        }
        const lk=`${dp}/${s}/${m}/${l}`;
        if(p==='F')forwardLayers.set(lk,{start:layerStart,end:time,tails:[...tails]});
        else {
          const f=forwardLayers.get(lk);
          owners.forEach(owner=>emit(`Activation · L${l}`,'hold',f.end,time-f.end,owner,mb,[root],f.tails,{layer:l,releaseEventIds:[...tails]}));
        }
      }
      if(p==='F'&&s===c.pp-1)head();
      if(p==='B'&&s===0)op('Token Embedding ∇',.08,['dv4/model/input_tokens','dv4/model/token_embedding']);
      const source=data.events.slice(firstEvent).filter(e=>e.kind!=='hold');
      for(const owner of owners) {
        const subset=source.filter(e=>e.rank===owner.rank||e.participants.includes(owner.rank));
        data.summaries.push({ ...subset[0], id:`summary/${id}/${owner.rank}`,rank:owner.rank,ep:owner.ep,label:`${p}${m}`,kind:p==='F'?'forward':'backward',start,end:time,
          graphNodeIds:[...new Set(subset.flatMap(e=>e.graphNodeIds))],sourceEventIds:subset.map(e=>e.id),isSummary:true });
      }
      completed.set(id,{end:time,tails});
    }
    if(!progress)throw Error('Cyclic pipeline schedule');
  }
  // Keep the runtime lifecycle shaped like the measured ST categories while all
  // absolute durations, parameter ownership and shard placement remain simulated.
  const backwardTails=[...tasks].filter(([,t])=>t.p==='B').flatMap(([id])=>completed.get(id).tails);
  const syncStart=Math.max(...backwardTails.map(id=>byId.get(id).end));
  const syncIds=[];
  for(let s=0;s<c.pp;s++) {
    const dense=data.ranks.filter(r=>r.stage===s);
    const denseGroup=data.parameterGroups.find(g=>g.id===`pg/pp${s}/dense`), denseCollective=data.collectiveGroups.find(g=>g.id===`collective/pp${s}/dense`);
    const e=emit('Dense Gradient Reduce-Scatter','comm',syncStart,c.gradientSyncMs,dense[0],null,denseGroup.modelNodeIds,backwardTails,
      {scope:'parameter-group',runtimeNodeIds:['runtime/gradient-reduce'],parameterGroupIds:[denseGroup.id],collectiveGroupId:denseCollective.id,
        participants:denseCollective.ranks,communicationType:'dp-ep',referenceEventCategory:'grads-reduce-scatter',provenanceIds:['model-config','simulation-config','st-optimizer-phases']});syncIds.push(e.id);
    for(let ep=0;ep<c.ep;ep++) {
      const experts=dense.filter(r=>r.ep===ep);
      const expertGroup=data.parameterGroups.find(g=>g.id===`pg/pp${s}/routed-experts`), expertCollective=data.collectiveGroups.find(g=>g.id===`collective/pp${s}/expert/ep${ep}`);
      const g=emit('Expert Gradient Reduce-Scatter','comm',e.end,c.gradientSyncMs,experts[0],null,expertGroup.modelNodeIds,[e.id],
        {scope:'parameter-group',runtimeNodeIds:['runtime/gradient-reduce'],parameterGroupIds:[expertGroup.id],collectiveGroupId:expertCollective.id,
          participants:expertCollective.ranks,communicationType:'dp',referenceEventCategory:'grads-reduce-scatter',provenanceIds:['model-config','simulation-config','st-optimizer-phases']});syncIds.push(g.id);
    }
  }
  const clipStart=Math.max(...syncIds.map(id=>byId.get(id).end)), clipIds=[];
  for(const owner of data.ranks) {
    const parameterGroupIds=[`pg/pp${owner.stage}/dense`,`pg/pp${owner.stage}/routed-experts`];
    const graphNodeIds=[...new Set(parameterGroupIds.flatMap(id=>data.parameterGroups.find(g=>g.id===id).modelNodeIds))];
    const clip=emit('Global Norm / Clip','optimizer',clipStart,c.gradientClipMs,owner,null,graphNodeIds,syncIds,
      {scope:'step',runtimeNodeIds:['runtime/gradient-clip'],parameterGroupIds,stream:'optimizer',relation:'runtime-owner',referenceEventCategory:'optimizer-clip-main-grad',
        provenanceIds:['model-config','simulation-config','st-optimizer-phases']});
    clipIds.push(clip.id);
  }
  const optimizerStart=Math.max(...clipIds.map(id=>byId.get(id).end)), optimizerIds=[];
  for(const owner of data.ranks) {
    const parameterGroupIds=[`pg/pp${owner.stage}/dense`,`pg/pp${owner.stage}/routed-experts`];
    const optimizerStateGroupIds=parameterGroupIds.map(id=>`opt-state/${id}`);
    const graphNodeIds=[...new Set(parameterGroupIds.flatMap(id=>data.parameterGroups.find(g=>g.id===id).modelNodeIds))];
    const shards=optimizerStateGroupIds.map((optimizerStateGroupId,index)=>{
      const collective=index===0?data.collectiveGroups.find(g=>g.id===`collective/pp${owner.stage}/dense`)
        :data.collectiveGroups.find(g=>g.id===`collective/pp${owner.stage}/expert/ep${owner.ep}`);
      return {optimizerStateGroupId,index:collective.ranks.indexOf(owner.rank),count:collective.ranks.length,axes:[...collective.axes],ownerRanks:[owner.rank]};
    });
    const event=emit('Optimizer Step','optimizer',optimizerStart,c.optimizerMs,owner,null,graphNodeIds,[clipIds[owner.rank]],
      {scope:'parameter-group',runtimeNodeIds:['runtime/optimizer-step'],parameterGroupIds,optimizerStateGroupIds,shards,stream:'optimizer',relation:'runtime-owner',
        referenceEventCategory:'optimizer',provenanceIds:['model-config','simulation-config','st-optimizer-phases']});
    optimizerIds.push(event.id);
  }
  const gatherStart=Math.max(...optimizerIds.map(id=>byId.get(id).end));
  for(let s=0;s<c.pp;s++) {
    const denseGroup=data.parameterGroups.find(g=>g.id===`pg/pp${s}/dense`), denseCollective=data.collectiveGroups.find(g=>g.id===`collective/pp${s}/dense`);
    emit('Dense Parameter AllGather','comm',gatherStart,c.parameterAllGatherMs,data.ranks.find(r=>r.rank===denseCollective.ranks[0]),null,denseGroup.modelNodeIds,optimizerIds,
      {scope:'parameter-group',runtimeNodeIds:['runtime/parameter-allgather'],parameterGroupIds:[denseGroup.id],collectiveGroupId:denseCollective.id,
        participants:denseCollective.ranks,communicationType:'dp-ep',referenceEventCategory:'params-all-gather',provenanceIds:['model-config','simulation-config','st-optimizer-phases']});
    for(let ep=0;ep<c.ep;ep++) {
      const expertGroup=data.parameterGroups.find(g=>g.id===`pg/pp${s}/routed-experts`), collective=data.collectiveGroups.find(g=>g.id===`collective/pp${s}/expert/ep${ep}`);
      emit('Expert Parameter AllGather','comm',gatherStart,c.parameterAllGatherMs,data.ranks.find(r=>r.rank===collective.ranks[0]),null,expertGroup.modelNodeIds,optimizerIds,
        {scope:'parameter-group',runtimeNodeIds:['runtime/parameter-allgather'],parameterGroupIds:[expertGroup.id],collectiveGroupId:collective.id,
          participants:collective.ranks,communicationType:'dp',referenceEventCategory:'params-all-gather',provenanceIds:['model-config','simulation-config','st-optimizer-phases']});
    }
  }
  data.stepBounds.end=Math.max(...data.events.map(e=>e.end));
  data.generationHash=crypto.createHash('sha256').update(JSON.stringify({model,c,ref})).digest('hex');
  return data;
}
module.exports={generate,reference};
if(require.main===module) {
  const data=generate();
  require('./timeline-contract.js').validate(data,require('./training-timeline.schema.json'));
  const json=JSON.stringify(data);
  if(process.argv[2]==='--write') {
    require('node:fs').writeFileSync(require('node:path').join(__dirname,'data/deepseek-v4-pro.timeline.json'),json+'\n');
    console.log(`Generated ${data.events.length} events / ${data.summaries.length} summaries; ${json.length} bytes`);
  } else process.stdout.write(json);
}
