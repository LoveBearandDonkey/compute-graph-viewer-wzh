(function attachPtoModelArchitecture3dDeck(global){
  'use strict';

  const VIEWS=new Set(['iso','front','right']);
  const THEMES=new Set(['dark','light']);
  const COLOR_FALLBACKS={
    embedding:'#14B8A6',norm:'#38BDF8',attention:'#3B82F6',linear:'#4F46E5',head:'#7C3AED',mlp:'#A855F7',act:'#8B5CF6',gate:'#F59E0B',moe:'#EA580C',comm:'#06B6D4',decoder:'#0D9488',
    input:'#A855F7',output:'#38BDF8',parameter:'#3B82F6',state:'#8B5CF6'
  };
  const VIEW_POSES={iso:{rx:-18,ry:-34},front:{rx:0,ry:0},right:{rx:0,ry:-90}};
  const SIDE_ROWS=[
    {label:'Input RMSNorm',ids:['attn_norm']},{label:'Attention TP/SP AllGather',ids:['attn_all_gather']},
    {label:'Q / KV Latent Linear',ids:['q_a_proj','kv_a_proj']},{label:'Q / KV Causal Conv1D',ids:['q_causal_conv','kv_causal_conv']},
    {label:'Q / KV Residual Add',ids:['q_residual_add','kv_residual_add']},{label:'Q / KV LayerNorm',ids:['q_a_norm','kv_a_norm']},
    {label:'Q / KV Up Linear',ids:['q_b_proj','kv_b_proj']},{label:'Query / Key / Value',ids:['query_tensor','key_tensor']},
    {label:'Sparse FlashAttention',ids:['attention_core']},{label:'Output Causal Conv1D',ids:['o_causal_conv']},
    {label:'Output Residual Add',ids:['o_residual_add']},{label:'Output Projection',ids:['o_proj']},
    {label:'Attention TP/SP Reduce-Scatter',ids:['attn_reduce_scatter']},{label:'Post Attention RMSNorm',ids:['post_attention_norm']},
    {label:'mHC Attention Merge',ids:['mhc_attention_post']},{label:'Pre-MLP RMSNorm',ids:['pre_mlp_norm']},
    {label:'Dense FFN TP/SP AllGather',ids:['moe_all_gather']},{label:'Dense Gate / Up Linear',ids:['dense_gate_up']},
    {label:'Dense SiLU × Multiply',ids:['dense_silu']},{label:'Dense Down Linear',ids:['dense_down']},
    {label:'Dense FFN TP/SP Reduce-Scatter',ids:['moe_reduce_scatter']},{label:'Router · Top-8',ids:['gate']},
    {label:'EP Dispatch · fused A2A',ids:['a2a_dispatch']},{label:'Expert Pool',ids:['expert_pool']},
    {label:'Shared Expert',ids:['shared_expert']},{label:'EP Combine · fused A2A',ids:['a2a_combine']},
    {label:'Routed + Shared Add',ids:['moe_branch_add']},{label:'Post-MLP RMSNorm',ids:['post_mlp_norm']},
    {label:'mHC FFN Merge',ids:['ffn_residual_add']},{label:'Block Post RMSNorm',ids:['block_post_norm']}
  ];
  const OPENPANGU_FLASH={
    id:'openpangu-flash',label:'openPangu-2.0-Flash',layerCount:46,depthGap:46,frontLayer:23,
    firstMoeLayer:2,denseLayers:[0,1],dsaLayers:[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45],
    blockPostLayers:[0,4,9,14,19,24,29,34,39],routedExperts:256,topK:8,
    stageRanges:[[0,11],[12,22],[23,34],[35,45]],
    representativeLayers:[0,12,23,35],sideRows:SIDE_ROWS
  };
  const PRESETS={'openpangu-flash':OPENPANGU_FLASH};
  const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
  const esc=(value)=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const qs=(value,root=document)=>typeof value==='string'?root.querySelector(value):value;

  function nodeHtml(node,extra=''){
    const compat=String(extra).split(/\s+/).filter(Boolean).map(name=>name==='is-tiny'?'opv-cssdeck-node--tiny':name==='is-wide'?'opv-cssdeck-node--wide':name==='is-add'?'opv-cssdeck-node--add':name==='is-compact-add'?'opv-cssdeck-node--compact-add':'').filter(Boolean).join(' ');
    return `<button type="button" class="pto-model-deck__node opv-cssdeck-node ${esc(extra)} ${compat}" data-node="${esc(node.id)}" data-op="${esc(node.op||'linear')}" style="left:${node.x}px;top:${node.y}px;width:${node.w}px;height:${node.h}px">${esc(node.label)}</button>`;
  }
  function center(node,side='mid'){
    if(side==='top')return{x:node.x+node.w/2,y:node.y};
    if(side==='bottom')return{x:node.x+node.w/2,y:node.y+node.h};
    return{x:node.x+node.w/2,y:node.y+node.h/2};
  }
  function edgesHtml(nodes,edges,{width=720,height=1180}={}){
    const paths=edges.map(edge=>{
      const a=nodes[edge[0]],b=nodes[edge[1]];if(!a||!b)return'';
      const p0=center(a,edge[3]||'bottom'),p1=center(b,edge[4]||'top'),route=edge[5]||{},mid=(p0.y+p1.y)/2;
      const d=route.mode==='elbow'&&Number.isFinite(route.viaX)
        ?`M${p0.x} ${p0.y}L${route.viaX} ${p0.y}L${route.viaX} ${p1.y}L${p1.x} ${p1.y}`
        :Math.abs(p0.x-p1.x)<10?`M${p0.x} ${p0.y}L${p1.x} ${p1.y}`:`M${p0.x} ${p0.y}C${p0.x} ${mid},${p1.x} ${mid},${p1.x} ${p1.y}`;
      return `<path data-kind="${esc(edge[2]||'activation')}" data-edge="${esc(edge[2]||'activation')}" data-source="${esc(edge[0])}" data-target="${esc(edge[1])}" d="${d}"></path>`;
    }).join('');
    return `<svg class="pto-model-deck__edges opv-cssgraph__edges" viewBox="0 0 ${width} ${height}" aria-hidden="true">${paths}</svg>`;
  }
  function graphNode(parts,nodes,id,label,op,x,y,w,h,extra=''){const node={id,label,op,x,y,w,h};nodes[id]=node;parts.push(nodeHtml(node,extra));}
  function clusterHtml(label,x,y,w,h,color){return `<div class="pto-model-deck__cluster opv-cssgraph__cluster" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px;--cluster-color:${color}"><span>${esc(label)}</span></div>`;}
  function expertPoolHtml(nodes){const box={x:300,y:700,w:140,h:52},expanded={x:260,y:688,w:236,h:126};nodes.expert_pool={...box,id:'expert_pool',op:'moe'};return `<div class="pto-model-deck__experts opv-cssdeck-experts is-collapsed" data-node="expert_pool" data-op="moe" role="button" tabindex="0" aria-expanded="false" aria-label="Expert Pool · 72 experts" data-collapsed-x="${box.x}" data-collapsed-y="${box.y}" data-collapsed-w="${box.w}" data-collapsed-h="${box.h}" data-expanded-x="${expanded.x}" data-expanded-y="${expanded.y}" data-expanded-w="${expanded.w}" data-expanded-h="${expanded.h}" style="left:${box.x}px;top:${box.y}px;width:${box.w}px;height:${box.h}px"></div>`;}
  function layerHtml(layer,config){
    const stage=config.stageRanges.findIndex(([lo,hi])=>layer>=lo&&layer<=hi);
    const first=config.representativeLayers.includes(layer);
    const dense=layer<config.firstMoeLayer,stageRange=config.stageRanges[stage],blockPost=config.blockPostLayers.includes(layer);
    const attn=config.dsaLayers.includes(layer)?'DSA':'SWA',nodes={},parts=[];
    parts.push(`<div class="pto-model-deck__layer-label opv-cssgraph__layer-label">L${layer}<span>PP${stage} · L${stageRange[0]}-${stageRange[1]} · ${dense?'Dense':'MoE'} · ${attn}${blockPost?' · block-post':''}</span></div>`);
    parts.push(clusterHtml('Sparse MLA Attention · TP/SP',72,48,576,570,'var(--pto-model-deck-attention)'));
    parts.push(clusterHtml(dense?'Dense FFN · TP/SP':'MoE FFN · EP dispatch_combine',92,650,536,390,dense?'var(--pto-model-deck-linear)':'var(--pto-model-deck-moe)'));
    graphNode(parts,nodes,'mhc_state_in','X_l · mHC state ×4','mhc-state',500,14,190,30,'is-tiny');
    graphNode(parts,nodes,'attn_norm','Input RMSNorm','norm',270,68,180,36);
    graphNode(parts,nodes,'attn_all_gather','TP/SP AllGather','comm',270,116,180,30,'is-tiny');
    graphNode(parts,nodes,'q_a_proj','Q Latent Linear','linear',98,154,176,32);graphNode(parts,nodes,'kv_a_proj','KV Latent Linear','linear',446,154,176,32);
    graphNode(parts,nodes,'q_causal_conv','Q Causal Conv1D','act',98,194,176,28,'is-tiny');graphNode(parts,nodes,'kv_causal_conv','KV Causal Conv1D','act',446,194,176,28,'is-tiny');
    graphNode(parts,nodes,'q_residual_add','+','add',172,228,24,24,'is-add is-compact-add');graphNode(parts,nodes,'kv_residual_add','+','add',520,228,24,24,'is-add is-compact-add');
    graphNode(parts,nodes,'q_a_norm','Q LayerNorm','norm',98,256,176,28,'is-tiny');graphNode(parts,nodes,'kv_a_norm','KV LayerNorm','norm',446,256,176,28,'is-tiny');
    graphNode(parts,nodes,'q_b_proj','Q Up Linear','linear',98,290,176,32);graphNode(parts,nodes,'kv_b_proj','KV Up Linear','linear',446,290,176,32);
    graphNode(parts,nodes,'query_tensor','Query','linear',156,328,118,24,'is-tiny');graphNode(parts,nodes,'key_tensor','Key / Value','linear',446,328,130,24,'is-tiny');
    graphNode(parts,nodes,'attention_core','Sparse FlashAttention','attention',250,366,220,38,'is-wide');
    graphNode(parts,nodes,'o_causal_conv','Output Causal Conv1D','act',270,414,180,28,'is-tiny');graphNode(parts,nodes,'o_residual_add','+','add',348,448,24,24,'is-add is-compact-add');
    graphNode(parts,nodes,'o_proj','Output Projection','linear',270,476,180,30,'is-tiny');graphNode(parts,nodes,'attn_reduce_scatter','TP/SP Reduce-Scatter','comm',270,514,180,26,'is-tiny');
    graphNode(parts,nodes,'post_attention_norm','Post Attention RMSNorm','norm',270,548,180,30,'is-tiny');graphNode(parts,nodes,'mhc_attention_post','mHC Attention Merge','attention',270,586,180,30,'is-tiny');
    graphNode(parts,nodes,'pre_mlp_norm','Pre-MLP RMSNorm','norm',270,624,180,32);
    if(dense){
      graphNode(parts,nodes,'moe_all_gather','TP/SP AllGather','comm',270,680,180,28,'is-tiny');graphNode(parts,nodes,'dense_gate_up','Gate / Up Linear','linear',262,718,196,32);
      graphNode(parts,nodes,'dense_silu','SiLU × Multiply','act',270,760,180,30);graphNode(parts,nodes,'dense_down','Dense Down Linear','linear',264,800,192,32);
      graphNode(parts,nodes,'moe_reduce_scatter','TP/SP Reduce-Scatter','comm',270,842,180,28,'is-tiny');
    }else{
      graphNode(parts,nodes,'gate','Router · Top-8','gate',98,678,150,32);graphNode(parts,nodes,'a2a_dispatch','EP Dispatch · fused A2A','comm',98,720,150,28,'is-tiny');
      parts.push(expertPoolHtml(nodes));graphNode(parts,nodes,'a2a_combine','EP Combine · fused A2A','comm',293,824,180,28,'is-tiny');
      graphNode(parts,nodes,'shared_expert','Shared Expert','mlp',508,700,140,52);graphNode(parts,nodes,'moe_branch_add','+','add',344,866,32,28,'is-add');
    }
    graphNode(parts,nodes,'post_mlp_norm','Post-MLP RMSNorm','norm',270,914,180,32);graphNode(parts,nodes,'ffn_residual_add','mHC FFN Merge','attention',270,954,180,30,'is-tiny');
    if(blockPost)graphNode(parts,nodes,'block_post_norm','Block Post RMSNorm','norm',270,994,180,30,'is-tiny');
    graphNode(parts,nodes,'mhc_state_out','X_{l+1} · mHC state ×4','mhc-state',500,1044,190,28,'is-tiny');
    const edges=[
      ['mhc_state_in','attn_norm','state-spine'],['attn_norm','attn_all_gather','comm'],['attn_all_gather','q_a_proj'],['attn_all_gather','kv_a_proj'],
      ['q_a_proj','q_causal_conv'],['q_a_proj','q_residual_add','residual'],['q_causal_conv','q_residual_add'],['q_residual_add','q_a_norm'],['q_a_norm','q_b_proj'],['q_b_proj','query_tensor'],
      ['kv_a_proj','kv_causal_conv'],['kv_a_proj','kv_residual_add','residual'],['kv_causal_conv','kv_residual_add'],['kv_residual_add','kv_a_norm'],['kv_a_norm','kv_b_proj'],['kv_b_proj','key_tensor'],
      ['query_tensor','attention_core'],['key_tensor','attention_core'],['attention_core','o_causal_conv'],['attention_core','o_residual_add','residual'],['o_causal_conv','o_residual_add'],
      ['o_residual_add','o_proj'],['o_proj','attn_reduce_scatter','comm'],['attn_reduce_scatter','post_attention_norm','comm'],['post_attention_norm','mhc_attention_post'],
      ['mhc_state_in','mhc_attention_post','residual','right','right',{mode:'elbow',viaX:668}],['mhc_attention_post','pre_mlp_norm']
    ];
    if(dense)edges.push(['pre_mlp_norm','moe_all_gather','comm'],['moe_all_gather','dense_gate_up'],['dense_gate_up','dense_silu'],['dense_silu','dense_down'],['dense_down','moe_reduce_scatter'],['moe_reduce_scatter','post_mlp_norm','comm']);
    else edges.push(['pre_mlp_norm','gate'],['gate','a2a_dispatch','comm'],['a2a_dispatch','expert_pool','comm'],['expert_pool','a2a_combine','comm'],['a2a_combine','moe_branch_add'],['pre_mlp_norm','shared_expert'],['shared_expert','moe_branch_add'],['moe_branch_add','post_mlp_norm']);
    edges.push(['post_mlp_norm','ffn_residual_add'],['mhc_attention_post','ffn_residual_add','residual','right','right',{mode:'elbow',viaX:668}]);
    if(blockPost)edges.push(['ffn_residual_add','block_post_norm'],['block_post_norm','mhc_state_out','state-spine']);else edges.push(['ffn_residual_add','mhc_state_out','state-spine']);
    return `<section class="pto-model-deck__layer opv-cssdeck-card${layer===config.frontLayer?' is-front-layer':''}" data-layer="${layer}" data-stage="${stage}" data-stage-role="${first?'first':'repeat'}" data-stage-sample="true" style="--deck-opacity:${(1-.38*(layer/45)).toFixed(3)};transform:translate3d(0,0,${-layer*config.depthGap}px)"><div class="pto-model-deck__graph opv-cssgraph">${edgesHtml(nodes,edges)}${parts.join('')}</div></section>`;
  }
  function staticHtml(kind,config){
    const input=kind==='input',nodes={},parts=[];
    if(input){
      graphNode(parts,nodes,'token_ids','Token IDs','input',20,34,112,30,'is-tiny');graphNode(parts,nodes,'positions','Position IDs','input',148,34,112,30,'is-tiny');
      graphNode(parts,nodes,'attention_context','Attention Context','input',276,34,150,30,'is-tiny');graphNode(parts,nodes,'embedding_weight','Embedding Weight','parameter',526,34,154,30,'is-tiny');
      graphNode(parts,nodes,'embedding','Parallel Embedding','embedding',270,82,180,32,'is-tiny');nodes.decoder_entry_depth={x:595,y:154,w:0,h:0};nodes.decoder_entry_front={x:360,y:178,w:0,h:0};
      const edges=[['token_ids','embedding'],['positions','embedding'],['attention_context','embedding'],['embedding_weight','embedding','parameter'],['embedding','decoder_entry_depth','model-spine-depth'],['embedding','decoder_entry_front','model-spine-front']];
      return `<section class="pto-model-deck__static pto-model-deck__static--input opv-cssdeck-static opv-cssdeck-static--input" style="transform:translate3d(0,0,${config.depthGap*1.55}px)"><div class="pto-model-deck__static-title opv-cssdeck-static__title">Model input · source checked</div>${edgesHtml(nodes,edges,{height:178})}${parts.join('')}</section>`;
    }
    graphNode(parts,nodes,'final_norm','Final RMSNorm','norm',270,34,180,32,'is-tiny');graphNode(parts,nodes,'lm_head_weight','LM Head Weight','parameter',70,82,140,30,'is-tiny');
    graphNode(parts,nodes,'lm_head','LM Head','head',270,82,180,32,'is-tiny');graphNode(parts,nodes,'logits_allgather','Logits All-Gather','comm',270,128,180,32,'is-tiny');graphNode(parts,nodes,'logits','Logits','output',270,174,180,32,'is-tiny');
    parts.push(clusterHtml('Multi Token Predictor · L46–L48 ×3',200,244,320,274,'var(--pto-model-deck-attention)'));
    graphNode(parts,nodes,'mtp_input_norms','MTP Input Norms','norm',270,268,180,30,'is-tiny');graphNode(parts,nodes,'mtp_eh_proj','EH Projection','linear',270,314,180,30,'is-tiny');
    graphNode(parts,nodes,'mtp_decoder_layer','MTP Decoder ×3','decoder',270,360,180,30,'is-tiny');graphNode(parts,nodes,'mtp_head_weight','MTP Head Weight','parameter',70,406,140,30,'is-tiny');
    graphNode(parts,nodes,'mtp_shared_head','MTP Shared Head','head',270,406,180,30,'is-tiny');graphNode(parts,nodes,'mtp_logits','MTP Logits','output',270,452,180,30,'is-tiny');
    nodes.decoder_exit_depth={x:595,y:-66,w:0,h:0};nodes.decoder_exit_front={x:360,y:-126,w:0,h:0};nodes.decoder_exit_block={x:360,y:-86,w:0,h:0};
    const edges=[['decoder_exit_depth','final_norm','model-spine-depth'],['decoder_exit_front','final_norm','model-spine-front'],['decoder_exit_block','final_norm','model-spine-block'],['final_norm','lm_head'],['lm_head','logits_allgather','comm'],['logits_allgather','logits'],['lm_head_weight','lm_head','parameter'],['final_norm','mtp_input_norms','activation','right','right',{mode:'elbow',viaX:548}],['mtp_input_norms','mtp_eh_proj'],['mtp_eh_proj','mtp_decoder_layer'],['mtp_decoder_layer','mtp_shared_head'],['mtp_head_weight','mtp_shared_head','parameter'],['mtp_shared_head','mtp_logits']];
    return `<section class="pto-model-deck__static pto-model-deck__static--output opv-cssdeck-static opv-cssdeck-static--output" style="transform:translate3d(0,0,${-(config.layerCount+.5)*config.depthGap}px)"><div class="pto-model-deck__static-title opv-cssdeck-static__title">Main output + MTP tail · source checked</div>${edgesHtml(nodes,edges,{height:526})}${parts.join('')}</section>`;
  }
  function shellHtml(config,options={}){
    const chrome=options.showChrome===false?'':`<div class="pto-model-deck__title">${esc(config.label)}<span>CSS 3D · ${config.layerCount} layers</span></div>
      <div class="pto-model-deck__toolbar" data-stage-ui>
        <div class="pto-model-deck__control" role="group" aria-label="视图切换">${['iso','front','right'].map(view=>`<button class="pto-model-deck__button" type="button" data-deck-view="${view}">${view==='iso'?'3D':view==='front'?'正视':'侧视'}</button>`).join('')}</div>
        <div class="pto-model-deck__control"><button class="pto-model-deck__button" type="button" data-theme-toggle>浅色</button></div>
        <div class="pto-model-deck__control"><button class="pto-model-deck__button" type="button" data-deck-fit>适配</button><span class="pto-model-deck__readout" data-deck-readout>100%</span></div>
      </div>`;
    return `${chrome}
      <div class="pto-model-deck__viewport">
        <div class="pto-model-deck__scene opv-cssdeck__scene">${staticHtml('input',config)}${Array.from({length:config.layerCount},(_,layer)=>layerHtml(layer,config)).join('')}${staticHtml('output',config)}</div>
        <svg class="pto-model-deck__interlayer-spine" aria-hidden="true"></svg>
        <svg class="pto-model-deck__side-guides" aria-label="侧视 PP 分割与 residual state 主干"></svg>
        <div class="pto-model-deck__pp-groups" aria-label="PP 分组标注"></div>
        <div class="pto-model-deck__side-labels" aria-label="侧视算子标注"></div>
      </div>`;
  }

  function normalizeConfig(options){
    const preset=typeof options.preset==='string'?PRESETS[options.preset]:options.preset;
    const source=options.config||preset||OPENPANGU_FLASH;
    return {...OPENPANGU_FLASH,...source,sideRows:source.sideRows||SIDE_ROWS};
  }
  function mount(rootInput,options={}){
    const root=qs(rootInput); if(!root)return null;
    const config=normalizeConfig(options);
    root.classList.add('pto-model-deck'); root.dataset.sharedPattern='model-architecture-3d-deck';root.innerHTML=shellHtml(config,options);
    root.style.setProperty('--pto-model-deck-depth-gap',`${config.depthGap}px`);
    const viewport=root.querySelector('.pto-model-deck__viewport');
    const scene=root.querySelector('.pto-model-deck__scene');
    const interlayerSpine=root.querySelector('.pto-model-deck__interlayer-spine');
    const sideGuides=root.querySelector('.pto-model-deck__side-guides');
    const ppGroups=root.querySelector('.pto-model-deck__pp-groups');
    const sideLabels=root.querySelector('.pto-model-deck__side-labels');
    const readout=root.querySelector('[data-deck-readout]');
    const initialTheme=THEMES.has(options.initialTheme)?options.initialTheme:(document.documentElement.dataset.theme==='light'?'light':'dark');
    const state={view:VIEWS.has(options.initialView)?options.initialView:'iso',theme:initialTheme,zoom:Number(options.initialZoom)||.5,rx:-18,ry:-34,panX:0,panY:0,selected:null};
    let drag=null,raf=0,destroyed=false;

    function applySemanticPalette(){
      const shared=global.PtoModelGraphvizPattern?.modelArchitectureColormap?.({nodes:[],clusters:[]},{theme:state.theme});
      const semantic=shared?.semanticColors||{};
      const io=shared?.ioColors||{};
      const value=(entry,fallback)=>typeof entry==='string'?entry:(entry?.raw||entry?.color||fallback);
      const colors={
        embedding:value(semantic['sem:embedding'],COLOR_FALLBACKS.embedding),norm:value(semantic['sem:norm'],COLOR_FALLBACKS.norm),attention:value(semantic['sem:attention'],COLOR_FALLBACKS.attention),
        linear:value(semantic['sem:linear'],COLOR_FALLBACKS.linear),head:value(semantic['sem:head'],COLOR_FALLBACKS.head),mlp:value(semantic['sem:mlp'],COLOR_FALLBACKS.mlp),act:value(semantic['sem:act'],COLOR_FALLBACKS.act),
        gate:value(semantic['sem:gate'],COLOR_FALLBACKS.gate),moe:value(semantic['sem:moe'],COLOR_FALLBACKS.moe),comm:value(semantic['sem:comm'],COLOR_FALLBACKS.comm),decoder:value(semantic['module:decoder'],COLOR_FALLBACKS.decoder),
        input:value(io.input,COLOR_FALLBACKS.input),output:value(io.output,COLOR_FALLBACKS.output),parameter:value(io.parameter,COLOR_FALLBACKS.parameter),state:value(io.state,COLOR_FALLBACKS.state)
      };
      if(state.theme==='light')colors.input=colors.output=colors.parameter=colors.state='#D7D7D7';
      Object.entries(colors).forEach(([key,color])=>root.style.setProperty(`--pto-model-deck-${key}`,color));
    }

    function pivot(){return state.pivot||{x:0,y:0,z:-(config.layerCount-1)*config.depthGap/2};}
    function transformValue(){const p=pivot();return `scale(${state.zoom}) rotateX(${state.rx}deg) rotateY(${state.ry}deg) translate3d(${-p.x}px,${-p.y}px,${-p.z}px)`;}
    function apply(){scene.style.left=`calc(50% + ${state.panX}px)`;scene.style.top=`calc(50% + ${state.panY}px)`;scene.style.transform=transformValue();if(readout)readout.textContent=`${Math.round(state.zoom*100)}%`;scheduleOverlay();}
    function syncButtons(){
      root.dataset.view=state.view;
      root.querySelectorAll('[data-deck-view]').forEach(button=>{const active=button.dataset.deckView===state.view;button.classList.toggle('is-active',active);button.setAttribute('aria-pressed',String(active));});
      const themeButton=root.querySelector('[data-theme-toggle]');if(themeButton){const light=state.theme==='light';themeButton.textContent=light?'深色':'浅色';themeButton.title=light?'切换深色模式':'切换浅色模式';themeButton.setAttribute('aria-label',themeButton.title);themeButton.setAttribute('aria-pressed',String(light));}
    }
    function setView(view){
      state.view=VIEWS.has(view)?view:'iso';Object.assign(state,VIEW_POSES[state.view]);state.panX=0;state.panY=0;
      syncButtons();syncExpertExpansion();fit();options.onViewChange?.(state.view,api);return api;
    }
    function setTheme(theme){state.theme=THEMES.has(theme)?theme:'dark';document.documentElement.dataset.theme=state.theme;applySemanticPalette();syncButtons();scheduleOverlay();options.onThemeChange?.(state.theme,api);return api;}
    function setZoom(value){state.zoom=clamp(Number(value)||state.zoom,.12,1.2);apply();options.onZoomChange?.(state.zoom,api);return api;}
    function fit(){
      const width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight);
      const raw=state.view==='front'?Math.min(width/850,height/1810):state.view==='right'?Math.min(width/3000,height/1900):Math.min(width/1850,height/2050);
      state.zoom=clamp(raw,.12,state.view==='front'?1.05:.86);apply();return api;
    }
    function setFrontLayer(layer){
      const next=clamp(Number(layer)||0,0,config.layerCount-1);
      root.querySelectorAll('.pto-model-deck__layer').forEach(card=>card.classList.toggle('is-front-layer',Number(card.dataset.layer)===next));
      syncExpertExpansion();scheduleOverlay();return api;
    }
    function syncExpertEdges(card){
      const svg=card.querySelector('.pto-model-deck__edges');if(!svg)return;
      svg.querySelectorAll('path[data-source="expert_pool"],path[data-target="expert_pool"]').forEach(path=>{
        const source=card.querySelector(`[data-node="${CSS.escape(path.dataset.source)}"]`),target=card.querySelector(`[data-node="${CSS.escape(path.dataset.target)}"]`);if(!source||!target)return;
        const point=(node,side)=>({x:node.offsetLeft+node.offsetWidth/2,y:side==='top'?node.offsetTop:node.offsetTop+node.offsetHeight});
        const p0=point(source,'bottom'),p1=point(target,'top'),mid=(p0.y+p1.y)/2;
        path.setAttribute('d',`M${p0.x} ${p0.y}C${p0.x} ${mid},${p1.x} ${mid},${p1.x} ${p1.y}`);
      });
    }
    function syncExpertExpansion(){
      root.querySelectorAll('.pto-model-deck__layer').forEach(card=>{
        const pool=card.querySelector('.pto-model-deck__experts');if(!pool)return;
        pool.classList.remove('is-expanded');pool.classList.add('is-collapsed');pool.setAttribute('aria-expanded','false');
        ['x','y','w','h'].forEach((key,index)=>{pool.style[index<2?(key==='x'?'left':'top'):(key==='w'?'width':'height')]=`${pool.dataset[`collapsed${key.toUpperCase()}`]}px`;});
        if(pool.childElementCount)pool.replaceChildren();
        syncExpertEdges(card);
      });
    }
    function selectNode(nodeId,layer){
      root.querySelectorAll('.pto-model-deck__node.is-selected,.pto-model-deck__experts.is-selected,.pto-model-deck__layer.is-selected').forEach(el=>el.classList.remove('is-selected'));
      const scope=Number.isFinite(Number(layer))?root.querySelector(`.pto-model-deck__layer[data-layer="${CSS.escape(String(layer))}"]`):root;
      const node=scope?.querySelector(`[data-node="${CSS.escape(String(nodeId))}"]`);node?.classList.add('is-selected');node?.closest('.pto-model-deck__layer')?.classList.add('is-selected');state.selected=node?{nodeId,layer}:null;
      interlayerSpine.querySelectorAll('[data-source-layer]').forEach(path=>{const selectedLayer=Number(layer);path.classList.toggle('is-selected',Number.isFinite(selectedLayer)&&(Number(path.dataset.sourceLayer)===selectedLayer||Number(path.dataset.targetLayer)===selectedLayer));});
      options.onNodeSelect?.(state.selected,node,api);return node;
    }
    function project(point,matrix,metrics){const t=new DOMPoint(point.x,point.y,point.z,1).matrixTransform(matrix),den=Math.max(1,metrics.perspective-t.z),scale=metrics.perspective/den;return{x:metrics.originX+(metrics.sceneLeft+t.x-metrics.originX)*scale,y:metrics.originY+(metrics.sceneTop+t.y-metrics.originY)*scale,z:t.z};}
    function projectionContext(){
      const width=viewport.clientWidth,height=viewport.clientHeight;if(!width||!height||!global.DOMPoint||(typeof global.DOMMatrixReadOnly==='undefined'&&typeof global.DOMMatrix==='undefined'))return null;
      const Matrix=global.DOMMatrixReadOnly||global.DOMMatrix,matrix=new Matrix(transformValue()),perspective=parseFloat(getComputedStyle(viewport).perspective)||9000;
      return{width,height,matrix,metrics:{perspective,originX:width/2,originY:height/2,sceneLeft:width/2+state.panX,sceneTop:height/2+state.panY}};
    }
    function interlayerGeometry(){
      const z=(layer)=>-layer*config.depthGap,sideX=408,stateX=235,outputY=468,inputY=-561;
      const links=Array.from({length:config.layerCount-1},(_,layer)=>({sourceLayer:layer,targetLayer:layer+1,stageBoundary:config.stageRanges.some(([lo])=>lo===layer+1),points:[{x:stateX,y:outputY,z:z(layer)},{x:sideX,y:outputY,z:z(layer)},{x:sideX,y:inputY,z:z(layer+1)},{x:stateX,y:inputY,z:z(layer+1)}]}));
      links.unshift({sourceLayer:-1,targetLayer:0,kind:'input',points:[{x:0,y:-602,z:config.depthGap*1.55},{x:sideX,y:-602,z:config.depthGap*1.55},{x:sideX,y:inputY,z:0},{x:stateX,y:inputY,z:0}]});
      links.push({sourceLayer:config.layerCount-1,targetLayer:config.layerCount,kind:'output',points:[{x:stateX,y:outputY,z:z(config.layerCount-1)},{x:sideX,y:outputY,z:z(config.layerCount-1)},{x:sideX,y:570,z:-(config.layerCount+.5)*config.depthGap},{x:0,y:570,z:-(config.layerCount+.5)*config.depthGap}]});
      return links;
    }
    function renderInterlayerSpine(){
      interlayerSpine.innerHTML='';if(!['iso','right'].includes(state.view))return;const ctx=projectionContext();if(!ctx)return;
      interlayerSpine.setAttribute('viewBox',`0 0 ${ctx.width} ${ctx.height}`);
      interlayerGeometry().forEach(link=>{const points=link.points.map(point=>project(point,ctx.matrix,ctx.metrics)),path=document.createElementNS('http://www.w3.org/2000/svg','path'),selectedLayer=Number(state.selected?.layer);path.dataset.sourceLayer=String(link.sourceLayer);path.dataset.targetLayer=String(link.targetLayer);if(link.kind)path.dataset.kind=link.kind;if(link.stageBoundary)path.dataset.stageBoundary='true';if(Number.isFinite(selectedLayer)&&(link.sourceLayer===selectedLayer||link.targetLayer===selectedLayer))path.classList.add('is-selected');path.setAttribute('d',`M${points[0].x.toFixed(1)},${points[0].y.toFixed(1)}C${points[1].x.toFixed(1)},${points[1].y.toFixed(1)} ${points[2].x.toFixed(1)},${points[2].y.toFixed(1)} ${points[3].x.toFixed(1)},${points[3].y.toFixed(1)}`);interlayerSpine.appendChild(path);});
    }
    function renderSideGuides(){
      sideGuides.innerHTML='';if(state.view!=='right')return;
      const width=viewport.clientWidth,height=viewport.clientHeight,cards=Array.from(root.querySelectorAll('.pto-model-deck__layer[data-layer]')),base=viewport.getBoundingClientRect();if(!width||!height||!cards.length)return;
      sideGuides.setAttribute('viewBox',`0 0 ${width} ${height}`);
      const ffnBand=(lo,hi,nodeIds,kind,label)=>{
        const bandCards=cards.filter(card=>Number(card.dataset.layer)>=lo&&Number(card.dataset.layer)<=hi),nodes=bandCards.flatMap(card=>nodeIds.map(id=>card.querySelector(`[data-node="${id}"]`)).filter(Boolean));if(!bandCards.length||!nodes.length)return;
        const layerRects=bandCards.map(card=>card.getBoundingClientRect()),nodeRects=nodes.map(node=>node.getBoundingClientRect()),x1=Math.min(...layerRects.map(rect=>rect.left))-base.left-7,x2=Math.max(...layerRects.map(rect=>rect.right))-base.left+7,y1=Math.min(...nodeRects.map(rect=>rect.top))-base.top-9,y2=Math.max(...nodeRects.map(rect=>rect.bottom))-base.top+9;
        const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');rect.classList.add('pto-model-deck__side-ffn-band',`is-${kind}`);rect.setAttribute('x',x1.toFixed(1));rect.setAttribute('y',y1.toFixed(1));rect.setAttribute('width',Math.max(8,x2-x1).toFixed(1));rect.setAttribute('height',Math.max(8,y2-y1).toFixed(1));rect.setAttribute('rx','8');sideGuides.appendChild(rect);
        const text=document.createElementNS('http://www.w3.org/2000/svg','text');text.classList.add('pto-model-deck__side-ffn-label',`is-${kind}`);text.setAttribute('x',((x1+x2)/2).toFixed(1));text.setAttribute('y',(y1-8).toFixed(1));text.textContent=label;sideGuides.appendChild(text);
      };
      ffnBand(0,Math.min(1,config.layerCount-1),['moe_all_gather','dense_gate_up','dense_silu','dense_down','moe_reduce_scatter'],'dense','Dense FFN · L0–L1 · ×2');
      if(config.layerCount>config.firstMoeLayer)ffnBand(config.firstMoeLayer,config.layerCount-1,['gate','a2a_dispatch','expert_pool','shared_expert','a2a_combine','moe_branch_add'],'moe',`MoE FFN · L${config.firstMoeLayer}–L${config.layerCount-1} · ×${config.layerCount-config.firstMoeLayer}`);
      const statePoints=cards.map(card=>card.querySelector('[data-node="mhc_state_in"]')).filter(Boolean).map(node=>{const rect=node.getBoundingClientRect();return{x:rect.left+rect.width/2-base.left,y:rect.top+rect.height/2-base.top};}).sort((a,b)=>a.x-b.x);if(!statePoints.length)return;
      const y=statePoints.reduce((sum,point)=>sum+point.y,0)/statePoints.length,residualOffsets=[-6,-2,2,6],projectedCenter=node=>{const rect=node?.getBoundingClientRect();return rect?{x:rect.left+rect.width/2-base.left,y:rect.top+rect.height/2-base.top}:null;},inputPoint=projectedCenter(root.querySelector('.pto-model-deck__static--input [data-node="embedding"]')),outputPoint=projectedCenter(root.querySelector('.pto-model-deck__static--output [data-node="final_norm"]'));
      residualOffsets.forEach((offset,index)=>{const railY=y+offset,first=statePoints[0],last=statePoints[statePoints.length-1];
        if(inputPoint){const inputPath=document.createElementNS('http://www.w3.org/2000/svg','path');inputPath.classList.add('pto-model-deck__side-residual-connector');inputPath.dataset.stateRail=String(index);inputPath.setAttribute('d',`M${inputPoint.x.toFixed(1)},${inputPoint.y.toFixed(1)}C${(inputPoint.x+24).toFixed(1)},${inputPoint.y.toFixed(1)} ${(first.x-24).toFixed(1)},${railY.toFixed(1)} ${first.x.toFixed(1)},${railY.toFixed(1)}`);sideGuides.appendChild(inputPath);}
        if(outputPoint){const outputPath=document.createElementNS('http://www.w3.org/2000/svg','path');outputPath.classList.add('pto-model-deck__side-residual-connector');outputPath.dataset.stateRail=String(index);outputPath.setAttribute('d',`M${last.x.toFixed(1)},${railY.toFixed(1)}C${(last.x+24).toFixed(1)},${railY.toFixed(1)} ${(outputPoint.x-24).toFixed(1)},${outputPoint.y.toFixed(1)} ${outputPoint.x.toFixed(1)},${outputPoint.y.toFixed(1)}`);sideGuides.appendChild(outputPath);}
        const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.classList.add('pto-model-deck__side-residual-spine');path.dataset.stateRail=String(index);path.setAttribute('d',`M${first.x.toFixed(1)},${railY.toFixed(1)}L${last.x.toFixed(1)},${railY.toFixed(1)}`);sideGuides.appendChild(path);statePoints.forEach(point=>{const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');dot.classList.add('pto-model-deck__side-residual-dot');dot.dataset.stateRail=String(index);dot.setAttribute('cx',point.x.toFixed(1));dot.setAttribute('cy',railY.toFixed(1));dot.setAttribute('r','1.75');sideGuides.appendChild(dot);});});
      const label=document.createElementNS('http://www.w3.org/2000/svg','text');label.classList.add('pto-model-deck__side-residual-label');label.setAttribute('x',(statePoints[0].x+8).toFixed(1));label.setAttribute('y',(y-10).toFixed(1));label.textContent='mHC residual state ×4';sideGuides.appendChild(label);
    }
    // 4 段 PP 分组的简单标注:灰字 + 灰色短分割线,贴在侧视层堆顶部,常显不随任何开关切换。
    function renderPpGroups(){
      ppGroups.innerHTML='';if(state.view!=='right')return;
      const cards=Array.from(root.querySelectorAll('.pto-model-deck__layer[data-layer]'));if(!cards.length)return;
      const base=ppGroups.getBoundingClientRect(),cardRects=cards.map(card=>card.getBoundingClientRect());
      const top=Math.max(70,Math.min(...cardRects.map(rect=>rect.top))-base.top-14);
      config.stageRanges.forEach(([lo,hi],stage)=>{
        const stageCards=cards.filter(card=>Number(card.dataset.layer)>=lo&&Number(card.dataset.layer)<=hi),rects=stageCards.map(card=>card.getBoundingClientRect());if(!rects.length)return;
        const left=Math.min(...rects.map(rect=>rect.left))-base.left,right=Math.max(...rects.map(rect=>rect.right))-base.left;
        if(stage>0){
          const divider=document.createElement('div');divider.className='pto-model-deck__pp-group-divider';divider.style.left=`${left}px`;divider.style.top=`${top}px`;ppGroups.appendChild(divider);
        }
        const label=document.createElement('div');label.className='pto-model-deck__pp-group-label';label.textContent=`PP${stage} · L${lo}–L${hi}`;label.style.left=`${(left+right)/2}px`;label.style.top=`${top}px`;ppGroups.appendChild(label);
      });
    }
    function renderSideLabels(){
      sideLabels.innerHTML='';if(options.showSideLabels===false||state.view!=='right')return;
      const base=sideLabels.getBoundingClientRect(),allCards=Array.from(root.querySelectorAll('.pto-model-deck__layer[data-layer]'));if(!allCards.length)return;
      const representative=allCards.find(card=>Number(card.dataset.layer)===config.frontLayer)||allCards[0],cards=[representative,...allCards.filter(card=>card!==representative)];
      const cardRects=allCards.map(card=>card.getBoundingClientRect()),g={left:Math.min(...cardRects.map(rect=>rect.left))-base.left,right:Math.max(...cardRects.map(rect=>rect.right))-base.left};
      const rows=config.sideRows.map(row=>{const nodes=row.ids.flatMap(id=>cards.map(card=>card.querySelector(`[data-node="${CSS.escape(id)}"]`)).filter(Boolean));if(!nodes.length)return null;const rects=nodes.map(node=>node.getBoundingClientRect()),y=rects.reduce((sum,rect)=>sum+rect.top+rect.height/2,0)/rects.length-base.top,color=getComputedStyle(nodes[0]).backgroundColor;return{row,nodes,y,color};}).filter(Boolean).sort((a,b)=>a.y-b.y);
      // x 只由模型层边缘 + 固定间距决定,不再向视口边界回夹:回夹会在缩放/平移使模型贴近
      // 容器边缘时把标注挤回模型上方,盖住层内容。宁可让标注被 side-labels 的 overflow:hidden
      // 裁掉一截,也不盖模型。
      const labelWidth=148,placeRight=base.width-g.right>=labelWidth+22,x=placeRight?g.right+14:g.left-14;
      rows.forEach((item,index)=>{const prev=index?Math.abs(item.y-rows[index-1].y):Infinity,next=index<rows.length-1?Math.abs(rows[index+1].y-item.y):Infinity,rule=document.createElement('button');rule.type='button';rule.className='pto-model-deck__side-rule';rule.style.cssText=`left:${g.left}px;top:${item.y}px;width:${Math.max(8,g.right-g.left)}px;height:${clamp(Math.min(prev,next)-2,8,18)}px;--side-color:${item.color}`;rule.addEventListener('click',()=>selectNode(item.row.ids[0]));const label=document.createElement('div');label.className=`pto-model-deck__side-label${placeRight?'':' is-left'}`;label.textContent=item.row.label;label.style.cssText=`left:${x}px;top:${item.y}px;--side-color:${item.color}`;sideLabels.append(rule,label);});
    }
    // options.onOverlay:每帧覆盖层重绘的收尾钩子(training-monitoring-v2 用它画对齐每层的逐层指标曲线)。
    // 在所有内置覆盖层画完后调用,与 side-guides/pp/annotations 共享同一套 viewport 屏幕坐标系,
    // 因此消费方只要按相同方式读 layer card 的 getBoundingClientRect() 就能与本体逐帧对齐。
    function renderOverlays(){raf=0;if(destroyed)return;renderInterlayerSpine();renderSideGuides();renderPpGroups();renderSideLabels();options.onOverlay?.({view:state.view,viewport,root,config,api});}
    function scheduleOverlay(){if(raf)return;raf=requestAnimationFrame(renderOverlays);}
    function pointerDown(event){if(event.button!==0||event.target.closest('[data-stage-ui],button'))return;drag={id:event.pointerId,x:event.clientX,y:event.clientY,rx:state.rx,ry:state.ry,panX:state.panX,panY:state.panY,pan:state.view!=='iso'||event.metaKey||event.ctrlKey};viewport.setPointerCapture?.(event.pointerId);viewport.classList.add('is-grabbing');}
    function pointerMove(event){if(!drag||event.pointerId!==drag.id)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(drag.pan){state.panX=drag.panX+dx;state.panY=drag.panY+dy;}else{state.ry=clamp(drag.ry+dx*.24,-82,82);state.rx=clamp(drag.rx-dy*.24,-74,74);}apply();}
    function pointerUp(event){if(!drag||event.pointerId!==drag.id)return;drag=null;viewport.classList.remove('is-grabbing');try{viewport.releasePointerCapture?.(event.pointerId);}catch(_){}}
    function wheel(event){if(event.target.closest('[data-stage-ui]'))return;event.preventDefault();setZoom(state.zoom*Math.exp(-event.deltaY*.0012));}
    function nodeClick(event){const node=event.target.closest('.pto-model-deck__node,.pto-model-deck__experts');if(!node)return;event.stopPropagation();selectNode(node.dataset.node,Number(node.closest('.pto-model-deck__layer')?.dataset.layer));}
    function resize(){fit();}
    root.querySelectorAll('[data-deck-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.deckView)));
    root.querySelector('[data-theme-toggle]')?.addEventListener('click',()=>setTheme(state.theme==='light'?'dark':'light'));
    root.querySelector('[data-deck-fit]')?.addEventListener('click',fit);
    const externallyManaged=options.externallyManaged===true;
    if(!externallyManaged){viewport.addEventListener('pointerdown',pointerDown);viewport.addEventListener('pointermove',pointerMove);viewport.addEventListener('pointerup',pointerUp);viewport.addEventListener('pointercancel',pointerUp);viewport.addEventListener('wheel',wheel,{passive:false});scene.addEventListener('click',nodeClick);}
    const resizeObserver=!externallyManaged&&global.ResizeObserver?new ResizeObserver(resize):null;resizeObserver?.observe(root);
    function setPose(pose={}){if(VIEWS.has(pose.view))state.view=pose.view;if(Number.isFinite(Number(pose.rx)))state.rx=Number(pose.rx);if(Number.isFinite(Number(pose.ry)))state.ry=Number(pose.ry);if(Number.isFinite(Number(pose.zoom)))state.zoom=clamp(Number(pose.zoom),.12,1.35);if(Number.isFinite(Number(pose.panX)))state.panX=Number(pose.panX);if(Number.isFinite(Number(pose.panY)))state.panY=Number(pose.panY);if(pose.pivot&&['x','y','z'].every(key=>Number.isFinite(Number(pose.pivot[key]))))state.pivot={x:Number(pose.pivot.x),y:Number(pose.pivot.y),z:Number(pose.pivot.z)};syncButtons();syncExpertExpansion();apply();return api;}
    const api={root,state,config,setView,setTheme,setZoom,setPose,refresh:scheduleOverlay,fit,setFrontLayer,selectNode,destroy(){destroyed=true;cancelAnimationFrame(raf);resizeObserver?.disconnect();if(!externallyManaged){viewport.removeEventListener('pointerdown',pointerDown);viewport.removeEventListener('pointermove',pointerMove);viewport.removeEventListener('pointerup',pointerUp);viewport.removeEventListener('pointercancel',pointerUp);viewport.removeEventListener('wheel',wheel);scene.removeEventListener('click',nodeClick);}}};
    setTheme(state.theme);Object.assign(state,VIEW_POSES[state.view]);syncExpertExpansion();if(externallyManaged){syncButtons();apply();}else requestAnimationFrame(fit);return api;
  }

  global.PtoModelArchitecture3dDeck={PRESETS,VIEW_POSES,render:mount,mount};
})(window);
