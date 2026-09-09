(function(root) {
  'use strict';
  // Deliberately supports only the JSON Schema keywords used by our versioned contract.
  // Unknown keywords fail closed so schema changes cannot silently escape validation.
  function validate(data,schema) {
    const fail=m=>{throw new Error(`Timeline contract: ${m}`);};
    function check(value,s,path) {
      if(s.$ref) { if(s.$ref!=='#/$defs/event')fail('unsupported ref'); return check(value,schema.$defs.event,path); }
      const supported=['$schema','$id','$defs','$ref','type','required','properties','items','enum','minimum'];
      Object.keys(s).forEach(k=>{if(!supported.includes(k))fail(`unsupported schema keyword ${k}`);});
      if(s.type) {
        const types=Array.isArray(s.type)?s.type:[s.type];
        const valid=types.some(t=>t==='null'?value===null:t==='integer'?Number.isInteger(value):t==='number'?Number.isFinite(value):t==='array'?Array.isArray(value):t==='object'?value!==null&&typeof value==='object'&&!Array.isArray(value):typeof value===t);
        if(!valid)fail(`${path}: type`);
      }
      if(s.enum&&!s.enum.includes(value))fail(`${path}: enum`);
      if(s.minimum!==undefined&&value<s.minimum)fail(`${path}: minimum`);
      for(const key of s.required||[])if(!(key in value))fail(`${path}.${key}: missing`);
      for(const [key,child]of Object.entries(s.properties||{}))if(key in value)check(value[key],child,`${path}.${key}`);
      if(s.items)value.forEach((v,i)=>check(v,s.items,`${path}[${i}]`));
    }
    check(data,schema,'root');
    const ranks=new Map(data.ranks.map(r=>[r.rank,r])), ids=new Map(), provenance=new Set(data.provenance.map(p=>p.id));
    const uniqueRegistry=(items,label)=>{
      const map=new Map();
      for(const item of items) { if(map.has(item.id))fail(`duplicate ${label} ${item.id}`);map.set(item.id,item); }
      return map;
    };
    const parameterGroups=uniqueRegistry(data.parameterGroups,'parameter group');
    const optimizerStateGroups=uniqueRegistry(data.optimizerStateGroups,'optimizer state group');
    const collectiveGroups=uniqueRegistry(data.collectiveGroups,'collective group');
    const runtimeNodes=uniqueRegistry(data.runtimeGraph.nodes,'runtime node');
    const modelNodeIds=new Set([...data.parameterGroups.flatMap(group=>group.modelNodeIds),...data.events.flatMap(event=>event.graphNodeIds)]);
    const checkProvenance=(item,label)=>{if(item.provenanceIds.some(id=>!provenance.has(id)))fail(`${label} provenance ${item.id}`);};
    for(const group of data.parameterGroups) {
      if(!data.stages.some(stage=>stage.id===group.stage)||group.layerRange.length!==2||group.layerRange[0]>group.layerRange[1])fail(`parameter group placement ${group.id}`);
      checkProvenance(group,'parameter group');
    }
    for(const state of data.optimizerStateGroups) {
      if(!parameterGroups.has(state.parameterGroupId))fail(`optimizer state parameter group ${state.id}`);
      checkProvenance(state,'optimizer state');
    }
    for(const group of data.collectiveGroups) {
      if(!group.ranks.length||group.ranks.some(rank=>!ranks.has(rank)))fail(`collective ranks ${group.id}`);
      checkProvenance(group,'collective group');
    }
    for(const node of data.runtimeGraph.nodes) {
      if(node.parameterGroupIds.some(id=>!parameterGroups.has(id))||node.optimizerStateGroupIds.some(id=>!optimizerStateGroups.has(id)))fail(`runtime references ${node.id}`);
      checkProvenance(node,'runtime node');
    }
    const runtimeEdgeIds=new Set(), runtimeEndpoints=new Set([...runtimeNodes.keys(),...modelNodeIds]);
    for(const edge of data.runtimeGraph.edges) {
      if(runtimeEdgeIds.has(edge.id)||!runtimeEndpoints.has(edge.source)||!runtimeEndpoints.has(edge.target))fail(`runtime edge ${edge.id}`);
      runtimeEdgeIds.add(edge.id);
    }
    for(const e of data.events) {
      if(ids.has(e.id))fail(`duplicate ${e.id}`); ids.set(e.id,e);
      const r=ranks.get(e.rank);if(!r||['dp','stage','tp','ep'].some(k=>e[k]!==r[k]))fail(`placement ${e.id}`);
      if(e.end<e.start||e.participants.some(r=>!ranks.has(r)))fail(`interval/participants ${e.id}`);
      if(e.provenanceIds.some(p=>!provenance.has(p)))fail(`provenance ${e.id}`);
      if(e.runtimeNodeIds.some(id=>!runtimeNodes.has(id))||e.parameterGroupIds.some(id=>!parameterGroups.has(id))||e.optimizerStateGroupIds.some(id=>!optimizerStateGroups.has(id)))fail(`registry reference ${e.id}`);
      if(e.collectiveGroupId!==null) {
        const group=collectiveGroups.get(e.collectiveGroupId);if(!group)fail(`collective reference ${e.id}`);
        if(JSON.stringify([...e.participants].sort((a,b)=>a-b))!==JSON.stringify([...group.ranks].sort((a,b)=>a-b)))fail(`collective participants ${e.id}`);
      }
      if(e.shards.some(shard=>!optimizerStateGroups.has(shard.optimizerStateGroupId)||shard.index>=shard.count||shard.ownerRanks.some(rank=>!ranks.has(rank))))fail(`shard reference ${e.id}`);
      if((e.scope==='microbatch'||e.scope==='layer')!==(e.microbatchId!==null))fail(`event scope ${e.id}`);
      if(e.fidelity!==data.fidelity)fail(`fidelity ${e.id}`);
      if(e.sourceEventIds.length!==1||e.sourceEventIds[0]!==e.id)fail(`source identity ${e.id}`);
    }
    for(const e of data.events)for(const dep of e.dependsOn) {
      const previous=ids.get(dep); if(!previous||dep===e.id||previous.end>e.start+1e-8)fail(`causal dependency ${e.id} <- ${dep}`);
    }
    const visited=new Set(),active=new Set();
    function visit(id) {
      if(active.has(id))fail(`dependency cycle ${id}`);if(visited.has(id))return;
      active.add(id);ids.get(id).dependsOn.forEach(visit);active.delete(id);visited.add(id);
    }
    data.events.forEach(e=>visit(e.id));
    for(const s of data.summaries)for(const id of s.sourceEventIds) {
      const e=ids.get(id);if(!e||e.start<s.start-1e-8||e.end>s.end+1e-8||e.microbatchId!==s.microbatchId)fail(`summary ${s.id}`);
    }
    if(!Number.isFinite(data.stepBounds.start)||!Number.isFinite(data.stepBounds.end)||data.events.some(e=>e.start<data.stepBounds.start-1e-8||e.end>data.stepBounds.end+1e-8))fail('step bounds');
    return data;
  }
  const api={validate}; if(typeof module!=='undefined')module.exports=api;else root.TimelineContract=api;
})(globalThis);
