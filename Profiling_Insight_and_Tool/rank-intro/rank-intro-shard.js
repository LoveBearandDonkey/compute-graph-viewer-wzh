(function(){
  'use strict';

  var page=document.querySelector('.page');
  var visual=document.getElementById('shardVisual');
  if(!page||!visual)return;
  var title=document.getElementById('shardTitle');
  var body=document.getElementById('shardBody');
  var number=document.getElementById('shardStepNumber');
  var rankQuantifier=document.getElementById('shardRankQuantifier');
  var minRanks=document.getElementById('shardMinRanks');
  var rankBasis=document.getElementById('shardRankBasis');
  var epToolbar=document.getElementById('shardEpToolbar');
  var epControlsHost=document.getElementById('shardEpControls');
  var steps=document.getElementById('shardSteps');
  var prev=document.getElementById('shardPrev');
  var next=document.getElementById('shardNext');
  var stage=0;
  var selectedEpRank=null;
  var epMode='cut';
  var epDegree=2;
  var DP=4, CP=2, TP=2, PP=4;
  var controllers=[];
  var matrixObserver=null;
  var activeTransition=null;
  var captions=[
    {name:'输入 BSH',body:'一个 Step 的输入是 B=8 个样本。每个样本是一张 S×H = 2048×2560 的矩阵。B：矩阵编号；S：纵向 token 行；H：横向 hidden 列。',fact:'8 × [2048, 2560]'},
    {name:'DP × 4：分样本',body:'沿 B 维把 8 个样本分给四个 DP 分支：DP0 持有 B0、B1，DP1 持有 B2、B3，DP2 持有 B4、B5，DP3 持有 B6、B7；每张矩阵的 S 和 H 不变。',fact:'每个 DP：2 × [2048, 2560]'},
    {name:'CP × 2：分 token',body:'每个 DP 再沿 S 维分成 CP0、CP1。八张源矩阵同时横向划线，再各自展开两支。缩略图保留完整 16×20 网格：CP0 的上半部着绿，CP1 的下半部着绿。',fact:'每个 DP×CP：2 × [1024, 2560]'},
    {name:'TP × 2：分 hidden',body:'每个 CP 分支再沿 H 维分成 TP0、TP1。十六张源矩阵同时纵向划线，再各自展开两支；绿色区域变为原矩阵的四分之一。CP 和 TP 分别切 S 和 H，互换先后仍得到相同分片。',fact:'每个 DP×CP×TP：2 × [1024, 1280]'},
    {name:'输入 Rank',body:'每条 DP→CP→TP 路径对应一个 Rank，共 4×2×2=16 个。每个 Rank 接收两个样本在对应 token 区间和 hidden 区间的分片。',fact:'16 个 Rank；每 Rank：2 × [1024, 1280]'},
    {name:'PP × 4：分模型层',body:'DP0–DP3 各自独立运行；每组 DP 都经过 PP0–PP3 四段，每段保留 4 个 CP×TP 坐标。以 48 层为例，PP0 负责 L0–11，PP1 负责 L12–23，依次传递微批激活。',fact:'4 组 DP × 4 段 PP × 4 个 CP×TP Rank = 64 个 Rank'},
    {name:'EP：分专家',body:'选择 EP 切入或 EP 正交，再选择 EP=2、4 或 8。切入口径从已有 DP Rank 划出专家组；正交口径为每个 DP 副本增加独立的 EP 坐标。',fact:''}
  ];

  function cells(cp,tp){
    var result=[];
    for(var row=0;row<2048;row+=128){
      for(var column=0;column<2560;column+=128){
        var owned=(cp==null||Math.floor(row/1024)===cp)&&(tp==null||Math.floor(column/1280)===tp);
        result.push({row:row,column:column,rowSpan:128,columnSpan:128,style:'value',tone:owned?'output':'neutral',states:owned?[]:['muted']});
      }
    }
    return result;
  }
  var cellCache={'cpnull-tpnull':cells(null,null)};
  function matrixScene(cp,tp){
    var key='cp'+cp+'-tp'+tp;
    if(!cellCache[key])cellCache[key]=cells(cp,tp);
    var cuts=[];
    if(cp!=null)cuts.push({axis:'rows',at:1024,tone:'output',color:'--cp'});
    if(tp!=null)cuts.push({axis:'columns',at:1280,tone:'output',color:'--tp'});
    return {extent:{rows:2048,columns:2560},axes:{rows:'S · token',columns:'H · hidden'},cells:cellCache[key],cuts:cuts};
  }

  function destroy(){
    if(matrixObserver){matrixObserver.disconnect();matrixObserver=null;}
    controllers.forEach(function(c){c.destroy();});controllers=[];
  }
  function mountCanvases(){
    if(!window.PtoMatrixCanvas)return;
    function mount(canvas){
      if(canvas.dataset.shardMounted)return;
      canvas.dataset.shardMounted='true';
      var cp=canvas.hasAttribute('data-cp')?Number(canvas.dataset.cp):null;
      var tp=canvas.hasAttribute('data-tp')?Number(canvas.dataset.tp):null;
      var context=canvas.getContext('2d');
      if(!context.__shardCornerScale){
        // MatrixCanvas uses a fixed 4px outer radius; keep it proportional as Shard previews shrink.
        var arcTo=context.arcTo;
        context.arcTo=function(x1,y1,x2,y2,radius){
          var scaled=Math.max(.35,canvas.getBoundingClientRect().height*4/146);
          return arcTo.call(this,x1,y1,x2,y2,radius===4?Math.min(radius,scaled):radius);
        };
        context.__shardCornerScale=true;
      }
      controllers.push(window.PtoMatrixCanvas.render(canvas,matrixScene(cp,tp),{ariaLabel:canvas.getAttribute('aria-label'),showAxes:false,showGrid:true,interactive:false,showTooltip:false,minZoom:.0001,padding:{top:2,right:2,bottom:2,left:2},cellStrokeAlpha:.12}));
      canvas.setAttribute('role','img');
      canvas.tabIndex=-1;
    }
    var canvases=visual.querySelectorAll('canvas[data-shard-matrix]');
    if(stage===6&&epMode==='ortho'&&window.IntersectionObserver){
      var observer=new IntersectionObserver(function(entries){
        if(matrixObserver!==observer)return;
        entries.forEach(function(entry){if(entry.isIntersecting){observer.unobserve(entry.target);mount(entry.target);}});
      },{root:visual,rootMargin:'160px 0px'});
      matrixObserver=observer;
      canvases.forEach(function(canvas){matrixObserver.observe(canvas);});
    }else canvases.forEach(mount);
  }
  function input(){
    var html='<div class="shard-input-group"><div class="shard-input-row">';
    for(var b=0;b<8;b++){
      html+='<div class="shard-input-matrix"><div class="shard-input-caption"><strong>B'+b+'</strong></div><div class="shard-input-chart"><span class="shard-axis-s">S ↓<br>2048</span><div class="shard-input-host"><canvas data-shard-matrix="input" data-fragment-key="b'+b+'" aria-label="B'+b+'，S 2048 行、H 2560 列"></canvas></div><span class="shard-axis-h">H → 2560</span></div></div>';
    }
    return html+'</div></div>';
  }
  function node(col,start,span,kind,html,added){
    return '<div class="shard-tree-node is-'+kind+(added?' is-new':'')+'" style="grid-column:'+col+';grid-row:'+start+' / span '+span+'">'+html+'</div>';
  }
  function brace(col,start,span,label,added,axis){
    var marginRows=axis==='dp'?span/8:span/4;
    return '<div class="shard-tree-brace is-'+axis+(added?' is-new':'')+'" style="grid-column:'+col+';grid-row:'+start+' / span '+span+';--brace-margin-rows:'+marginRows+'" role="img" aria-label="'+label+'"><svg viewBox="0 0 28 100" preserveAspectRatio="none" aria-hidden="true"><path d="M26 0 C12 0 12 24 12 34 C12 44 9 50 2 50"/><path d="M2 50 C9 50 12 56 12 66 C12 76 12 100 26 100"/></svg></div>';
  }
  function source(){
    return '<strong>输入 BSH</strong><span>B0–B7</span><small>[8, 2048, 2560]</small>';
  }
  function fragment(b,cp,tp){
    var key='b'+b+(cp==null?'':'-cp'+cp)+(tp==null?'':'-tp'+tp);
    var rows=cp==null?2048:1024,columns=tp==null?2560:1280;
    return '<div class="shard-fragment"><span class="shard-fragment-label">B'+b+'</span><div class="shard-fragment-host"><canvas data-shard-matrix="fragment" data-fragment-key="'+key+'"'+(cp==null?'':' data-cp="'+cp+'"')+(tp==null?'':' data-tp="'+tp+'"')+' aria-label="B'+b+'，完整 16×20 网格；持有 S '+rows+' 行、H '+columns+' 列的分片"></canvas></div></div>';
  }
  function pair(dp,cp,tp){return '<div class="shard-fragment-pair">'+fragment(dp*2,cp,tp)+fragment(dp*2+1,cp,tp)+'</div>';}
  function rankCard(rank,dp,cp,tp,pp,row,ep){
    var local=(dp*CP+cp)*TP+tp;
    var placement=row==null?'':' style="grid-column:7;grid-row:'+row+'"';
    return '<div class="shard-rank-slot"'+placement+' data-rank="'+rank+'" data-dp="'+dp+'" data-cp="'+cp+'" data-tp="'+tp+'" data-pp="'+(pp==null?0:pp)+'" data-ep="'+(ep==null?0:ep)+'" role="group" aria-label="Rank '+rank+'，DP'+dp+'，CP'+cp+'，TP'+tp+'，两张输入矩阵分片"><div class="shard-rank-label"><strong>R'+rank+'</strong><span>TP'+tp+'</span></div><div class="shard-tree-node is-rank shard-rank-card" data-rank-local="'+local+'"'+(pp==null?'':' data-pp="'+pp+'"')+'>'+pair(dp,cp,tp)+'</div></div>';
  }
  function tree(){
    var count=stage===1?DP:stage===2?DP*CP:DP*CP*TP;
    var html='<div class="shard-tree" data-stage="'+stage+'" style="--tree-rows:'+count+'">';
    html+=node(1,1,count,'root',source(),false);
    html+=brace(2,1,count,'沿 B 维分成 DP0 至 DP3',stage===1,'dp');
    for(var dp=0;dp<DP;dp++){
      var dpSpan=count/DP,dpStart=dp*dpSpan+1;
      html+=node(3,dpStart,dpSpan,'dp','<strong>DP'+dp+'</strong><span>B'+(dp*2)+' · B'+(dp*2+1)+'</span>'+(stage===1?pair(dp,null,null):'<small>2 × [2048, 2560]</small>'),stage===1);
      if(stage<2)continue;
      html+=brace(4,dpStart,dpSpan,'DP'+dp+' 沿 S 维分成 CP0 和 CP1',stage===2,'cp');
      for(var cp=0;cp<2;cp++){
        var cpSpan=count/(DP*CP),cpStart=dpStart+cp*cpSpan;
        html+=node(5,cpStart,cpSpan,'cp','<strong>CP'+cp+'</strong><span>S['+(cp*1024)+'~'+((cp+1)*1024-1)+']</span>'+(stage===2?pair(dp,cp,null):'<small>2 × [1024, 2560]</small>'),stage===2);
        if(stage<3)continue;
        html+=brace(6,cpStart,cpSpan,'DP'+dp+' CP'+cp+' 沿 H 维分成 TP0 和 TP1',stage===3,'tp');
        for(var tp=0;tp<2;tp++){
          var rank=(dp*CP+cp)*TP+tp,rankStart=cpStart+tp;
          if(stage===4)html+=rankCard(rank,dp,cp,tp,null,rankStart);
          else html+=node(7,rankStart,1,'tp','<strong>TP'+tp+'</strong><span>H['+(tp*1280)+'~'+((tp+1)*1280-1)+']</span>'+pair(dp,cp,tp),stage===3);
        }
      }
    }
    return html+'</div>';
  }
  function pipeline(){
    var html='<div class="shard-pipeline" data-stage="'+stage+'"><div class="shard-pp-guide">';
    for(var pp=0;pp<PP;pp++){
      html+='<div class="shard-pp-heading"><strong>PP'+pp+'</strong><span>L'+pp*12+'–'+(pp*12+11)+'</span></div>';
    }
    html+='</div>';
    for(var dp=0;dp<DP;dp++){
      var lanes=stage===6&&epMode==='ortho'?epDegree:1;
      for(var lane=0;lane<lanes;lane++){
      var ep=epMode==='cut'?dp%epDegree:lane;
      var expertStart=ep*(64/epDegree),expertEnd=expertStart+64/epDegree-1;
      html+='<div class="shard-dp-lane" role="group" aria-label="DP'+dp+(stage===6?'，EP'+ep:'')+' 的 16 个 Rank"><div class="shard-dp-heading"><strong>DP'+dp+'</strong><span>B'+(dp*2)+' · B'+(dp*2+1)+'</span><small>EP'+ep+' · E'+expertStart+'–'+expertEnd+'</small></div><div class="shard-dp-columns">';
      for(var pp=0;pp<PP;pp++){
        html+='<div class="shard-dp-column">';
        for(var cp=0;cp<CP;cp++){
          html+='<div class="shard-cp-pair"><span class="shard-cp-label">CP'+cp+'</span><div class="shard-tp-pair">';
          for(var tp=0;tp<TP;tp++){
            var rank=epMode==='ortho'&&stage===6?pp*(DP*epDegree*CP*TP)+(dp*epDegree+ep)*CP*TP+cp*TP+tp:pp*(DP*CP*TP)+(dp*CP+cp)*TP+tp;
            html+=rankCard(rank,dp,cp,tp,pp,null,ep);
          }
          html+='</div></div>';
        }
        html+='</div>';
      }
      html+='</div></div>';
      }
    }
    return html+'<div class="shard-pipeline-tail">'+pipelineTail()+'</div></div>';
  }
  function pipelineTail(){return stage===6?'Router 按专家位置在所选 EP 组内分发 token，计算结果返回源 Rank':'PP 相邻段顺序传递微批激活';}
  function rankCountGuidance(){
    var total=DP*CP*TP*PP*(epMode==='ortho'?epDegree:1);
    var description=epMode==='cut'?'EP '+epDegree+' 从 DP 4 中切出，形成 '+(DP/epDegree)+' 个数据副本，每个 EP 组跨 '+epDegree+' 个 DP 位置；Rank 总数仍为 64。':'EP '+epDegree+' 与 DP 4 正交：四个 DP 副本各有 '+epDegree+' 个 EP 位置，Rank 总数乘以 '+epDegree+'。';
    return '<p><strong>当前拓扑：</strong>'+description+'</p><p><strong>单个作业：</strong>DP 4 × CP 2 × TP 2 × PP 4'+(epMode==='ortho'?' × EP '+epDegree:'')+' = '+total+' 个 Rank。固定这些并行度时，每个 Rank 都有明确坐标。</p>';
  }
  function epControls(){
    return '<div class="c4s"><span class="lb">EP 口径</span><div class="ctl choice" role="group" aria-label="EP 口径">'+
      '<button type="button" data-shard-ep-mode="cut" aria-pressed="'+(epMode==='cut')+'">EP 切入</button><button type="button" data-shard-ep-mode="ortho" aria-pressed="'+(epMode==='ortho')+'">EP 正交</button></div></div>'+
      '<div class="c4s"><span class="lb">EP 并行度（64个专家）</span><div class="ctl choice" role="group" aria-label="EP 并行度（64个专家）">'+[2,4,8].map(function(degree){return '<button type="button" data-shard-ep-degree="'+degree+'" aria-pressed="'+(epDegree===degree)+'"'+(epMode==='cut'&&degree===8?' disabled title="切入口径的 EP 不能超过 DP=4"':'')+'>EP '+degree+'</button>';}).join('')+'</div></div>';
  }
  function epGroupKey(slot){
    return [slot.dataset.pp,slot.dataset.cp,slot.dataset.tp,epMode==='cut'?Math.floor(Number(slot.dataset.dp)/epDegree):slot.dataset.dp].join(':');
  }
  function updateEpSelection(){
    var pipelineView=visual.querySelector('.shard-pipeline');
    if(!pipelineView)return;
    var slots=Array.from(pipelineView.querySelectorAll('.shard-rank-slot'));
    var chosen=slots.find(function(slot){return Number(slot.dataset.rank)===selectedEpRank;});
    var key=chosen?epGroupKey(chosen):null;
    var members=key?slots.filter(function(slot){return epGroupKey(slot)===key;}):[];
    slots.forEach(function(slot){
      var rank=Number(slot.dataset.rank);
      var selected=stage===6&&rank===selectedEpRank;
      var paired=stage===6&&key===epGroupKey(slot)&&!selected;
      slot.classList.toggle('is-ep-selected',selected);
      slot.classList.toggle('is-ep-partner',paired);
      if(!slot.dataset.rankDescription)slot.dataset.rankDescription=slot.getAttribute('aria-label');
      if(stage===6){
        slot.setAttribute('role','button');
        slot.tabIndex=0;
        slot.setAttribute('aria-pressed',String(selected));
        slot.setAttribute('aria-label',slot.dataset.rankDescription+'；'+(selected?'当前选中，EP 组：'+members.map(function(item){return 'R'+item.dataset.rank;}).join('、'):paired?'与当前选中 R'+selectedEpRank+' 同属 EP 组':'点选查看 EP 组'));
      }else{
        slot.setAttribute('role','group');
        slot.removeAttribute('tabindex');
        slot.removeAttribute('aria-pressed');
        slot.setAttribute('aria-label',slot.dataset.rankDescription);
      }
    });
    var readout=document.getElementById('shardEpReadout');
    if(readout)readout.textContent=chosen?'R'+selectedEpRank+' 的 EP 组：'+members.map(function(slot){return 'R'+slot.dataset.rank;}).join(' · '):'点选任意 Rank，查看完整 EP 组。';
  }
  function render(){
    destroy();
    var current=captions[stage];
    number.textContent=String(stage+1).padStart(2,'0')+' / '+String(captions.length).padStart(2,'0');
    var factors=[],rankCount=1;
    if(stage>=1){factors.push('DP 4');rankCount*=DP;}
    if(stage>=2){factors.push('CP 2');rankCount*=CP;}
    if(stage>=3){factors.push('TP 2');rankCount*=TP;}
    if(stage>=5){factors.push('PP 4');rankCount*=PP;}
    if(stage===6&&epMode==='ortho'){factors.push('EP '+epDegree);rankCount*=epDegree;}
    rankQuantifier.textContent=stage===6?'只能用':'需要至少';
    minRanks.textContent=String(rankCount);
    rankBasis.textContent='（'+(factors.length?factors.join(' × '):'尚未切分')+(stage===6&&epMode==='cut'?'；EP '+epDegree+' 切入 DP':'')+'）';
    title.textContent=stage===6?'EP × '+epDegree+'：'+(epMode==='cut'?'切入 DP':'与 DP 正交'):current.name;
    var experts=Array.from({length:epDegree},function(_,ep){var first=ep*64/epDegree;return 'EP'+ep+'：E'+first+'–'+(first+64/epDegree-1);}).join('；');
    body.innerHTML=stage===6?'<p>'+current.body+'</p>'+rankCountGuidance()+'<p>每个 MoE 层有 64 个专家，ETP=1；每个 EP 位置持有 '+(64/epDegree)+' 个：'+experts+'。Dense 层沿 DP×CP×TP 运行。</p><p>'+pipelineTail()+'</p>':'<p>'+current.body+'</p><div class="shard-fact">'+current.fact+'</div>';
    epToolbar.hidden=stage!==6;
    epControlsHost.innerHTML=stage===6?epControls():'';
    visual.dataset.step=String(stage);
    visual.innerHTML=stage>=5?pipeline():stage===0?input():tree();
    updateEpSelection();
    var stageColors=['--success','--dp','--cp','--tp','--fg','--pp','--ep'];
    steps.innerHTML=captions.map(function(item,i){return '<button type="button" class="'+(i===stage?'cur':i<stage?'done':'')+'" style="--c:var('+stageColors[i]+')" data-shard-step="'+i+'"'+(i===stage?' aria-current="step"':'')+'><span class="lbl">'+String(i+1).padStart(2,'0')+' · '+item.name+'</span><span class="bar"></span></button>';}).join('');
    prev.disabled=stage===0;
    next.disabled=stage===captions.length-1;
    mountCanvases();
  }
  function cancelTransition(){
    if(!activeTransition)return;
    activeTransition.timers.forEach(clearTimeout);
    activeTransition.animations.forEach(function(animation){animation.cancel();});
    activeTransition.nodes.forEach(function(node){node.remove();});
    visual.querySelectorAll('.shard-tree-node.is-new,.shard-tree-brace.is-new').forEach(function(node){node.classList.remove('is-new');});
    activeTransition=null;
    visual.classList.remove('is-transitioning','is-transition-visible');
  }
  function copyCanvases(from,to){
    var originals=from.querySelectorAll('canvas'),copies=to.querySelectorAll('canvas');
    originals.forEach(function(canvas,i){
      var copy=copies[i];
      copy.width=canvas.width;copy.height=canvas.height;
      copy.getContext('2d').drawImage(canvas,0,0);
    });
  }
  function matrixArea(canvas){
    var rect=canvas.getBoundingClientRect();
    var innerWidth=Math.max(1,rect.width-4),innerHeight=Math.max(1,rect.height-4);
    var height=Math.min(innerHeight,innerWidth*2048/2560),width=height*2560/2048;
    return {left:rect.left+(rect.width-width)/2,top:rect.top+(rect.height-height)/2,width:width,height:height};
  }
  function morphPiece(source,target,cropMatrix){
    var rect=cropMatrix?matrixArea(source):source.getBoundingClientRect();
    var dest=cropMatrix?matrixArea(target):target.getBoundingClientRect();
    var piece=document.createElement('div');
    piece.className='shard-flying-piece'+(cropMatrix?' is-matrix-bounds':'');
    piece.style.left=rect.left+'px';piece.style.top=rect.top+'px';piece.style.width=rect.width+'px';piece.style.height=rect.height+'px';
    [source,target].forEach(function(canvas,i){
      var snapshot=document.createElement('canvas');
      snapshot.className=i?'shard-piece-owned':'shard-piece-before';
      if(cropMatrix){
        var area=i?dest:rect,canvasRect=canvas.getBoundingClientRect();
        var pixelX=canvas.width/canvasRect.width,pixelY=canvas.height/canvasRect.height;
        snapshot.width=Math.max(1,Math.round(area.width*pixelX));
        snapshot.height=Math.max(1,Math.round(area.height*pixelY));
        snapshot.getContext('2d').drawImage(canvas,(area.left-canvasRect.left)*pixelX,(area.top-canvasRect.top)*pixelY,area.width*pixelX,area.height*pixelY,0,0,snapshot.width,snapshot.height);
      }else{
        snapshot.width=canvas.width;snapshot.height=canvas.height;
        snapshot.getContext('2d').drawImage(canvas,0,0);
        var cutOverlay=canvas.parentElement.querySelector('.rank-matrix-canvas-compat');
        if(cutOverlay)snapshot.getContext('2d').drawImage(cutOverlay,0,0,snapshot.width,snapshot.height);
        var animatedCut=i?null:canvas.parentElement.querySelector('.shard-transition-cut');
        if(animatedCut){
          var cutRect=animatedCut.getBoundingClientRect(),fullRect=canvas.getBoundingClientRect();
          var cutContext=snapshot.getContext('2d');
          cutContext.fillStyle=getComputedStyle(animatedCut).backgroundColor;
          cutContext.fillRect((cutRect.left-fullRect.left)*snapshot.width/fullRect.width,(cutRect.top-fullRect.top)*snapshot.height/fullRect.height,cutRect.width*snapshot.width/fullRect.width,cutRect.height*snapshot.height/fullRect.height);
        }
      }
      piece.appendChild(snapshot);
    });
    document.body.appendChild(piece);
    var motion=piece.animate([
      {left:rect.left+'px',top:rect.top+'px',width:rect.width+'px',height:rect.height+'px'},
      {left:dest.left+'px',top:dest.top+'px',width:dest.width+'px',height:dest.height+'px'}
    ],{duration:680,easing:'cubic-bezier(.2,.72,.2,1)',fill:'forwards'});
    var handoff={duration:240,delay:440,fill:'forwards'};
    var fadeBefore=piece.firstChild.animate([{opacity:1},{opacity:0}],handoff);
    var fadeOwned=piece.lastChild.animate([{opacity:0},{opacity:1}],handoff);
    return {piece:piece,motions:[motion,fadeBefore,fadeOwned]};
  }
  function transitionTo(nextStage){
    var oldScene=visual.querySelector('.shard-tree,.shard-input-group');
    if(!oldScene){stage=nextStage;render();return;}
    var oldRect=oldScene.getBoundingClientRect();
    var ghost=oldScene.cloneNode(true);
    ghost.classList.add('shard-scene-ghost');
    ghost.style.left=oldRect.left+'px';ghost.style.top=oldRect.top+'px';
    ghost.style.width=oldRect.width+'px';ghost.style.height=oldRect.height+'px';
    ghost.style.margin='0';
    document.body.appendChild(ghost);
    copyCanvases(oldScene,ghost);
    var state={nodes:[ghost],timers:[],animations:[]};
    activeTransition=state;
    var axis=nextStage===2?'cp':nextStage===3?'tp':null;
    if(axis)ghost.querySelectorAll('canvas[data-fragment-key]').forEach(function(canvas){
      var line=document.createElement('i');
      line.className='shard-transition-cut is-'+axis;
      line.setAttribute('aria-hidden','true');
      canvas.parentElement.appendChild(line);
    });
    stage=nextStage;
    visual.classList.add('is-transitioning');
    render();
    state.timers.push(setTimeout(function(){
      if(activeTransition!==state)return;
      var targets=new Map();
      visual.querySelectorAll('canvas[data-fragment-key]').forEach(function(canvas){targets.set(canvas.dataset.fragmentKey,canvas);});
      ghost.querySelectorAll('canvas[data-fragment-key]').forEach(function(canvas){
        for(var part=0;part<(axis?2:1);part++){
          var key=canvas.dataset.fragmentKey+(axis?'-'+axis+part:'');
          var target=targets.get(key);
          if(!target)continue;
          var flying=morphPiece(canvas,target,nextStage===1);
          state.nodes.push(flying.piece);state.animations.push.apply(state.animations,flying.motions);
        }
      });
      visual.classList.add('is-transition-visible');
      ghost.style.visibility='hidden';
      state.timers.push(setTimeout(function(){if(activeTransition===state)cancelTransition();},700));
    },axis?650:80));
  }
  function transitionToPP(){
    var oldTree=visual.querySelector('.shard-tree');
    if(!oldTree){stage=5;render();return;}
    var oldRect=oldTree.getBoundingClientRect(),ghost=oldTree.cloneNode(true);
    ghost.classList.add('shard-scene-ghost');
    ghost.style.left=oldRect.left+'px';ghost.style.top=oldRect.top+'px';
    ghost.style.width=oldRect.width+'px';ghost.style.height=oldRect.height+'px';ghost.style.margin='0';
    document.body.appendChild(ghost);
    copyCanvases(oldTree,ghost);
    var state={nodes:[ghost],timers:[],animations:[]};
    activeTransition=state;
    stage=5;
    visual.classList.add('is-transitioning');
    render();
    state.timers.push(setTimeout(function(){
      if(activeTransition!==state)return;
      var sources=new Map();
      ghost.querySelectorAll('.shard-rank-card').forEach(function(card){sources.set(card.dataset.rankLocal,card);});
      visual.querySelectorAll('.shard-pipeline .shard-rank-card').forEach(function(target){
        var source=sources.get(target.dataset.rankLocal);
        if(!source)return;
        var origin=source.getBoundingClientRect(),dest=target.getBoundingClientRect();
        var flying=source.cloneNode(true);
        copyCanvases(source,flying);
        flying.classList.add('shard-flying-rank');
        flying.style.left=origin.left+'px';flying.style.top=origin.top+'px';
        document.body.appendChild(flying);
        state.nodes.push(flying);
        state.animations.push(flying.animate([
          {left:origin.left+'px',top:origin.top+'px'},
          {left:dest.left+'px',top:dest.top+'px'}
        ],{duration:720,easing:'cubic-bezier(.2,.72,.2,1)',fill:'forwards'}));
      });
      visual.classList.add('is-transition-visible');
      ghost.style.visibility='hidden';
      state.timers.push(setTimeout(function(){if(activeTransition===state)cancelTransition();},740));
    },80));
  }
  function setStage(value){
    var nextStage=Math.max(0,Math.min(captions.length-1,value));
    if(activeTransition)cancelTransition();
    if(nextStage===stage)return;
    selectedEpRank=nextStage===6?0:null;
    if(stage===4&&nextStage===5&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches){transitionToPP();return;}
    if((stage===0&&nextStage===1)||(stage===1&&nextStage===2)||(stage===2&&nextStage===3)||(stage===3&&nextStage===4)){
      if(!window.matchMedia('(prefers-reduced-motion: reduce)').matches){transitionTo(nextStage);return;}
    }
    stage=nextStage;render();
  }
  prev.addEventListener('click',function(){setStage(stage-1);});
  next.addEventListener('click',function(){setStage(stage+1);});
  steps.addEventListener('click',function(event){var button=event.target.closest('[data-shard-step]');if(button)setStage(Number(button.dataset.shardStep));});
  epControlsHost.addEventListener('click',function(event){
    if(stage!==6)return;
    var modeButton=event.target.closest('[data-shard-ep-mode]');
    var degreeButton=event.target.closest('[data-shard-ep-degree]');
    if(modeButton){
      var mode=modeButton.dataset.shardEpMode;
      if(mode===epMode)return;
      epMode=mode;
      if(epMode==='cut'&&epDegree>DP)epDegree=DP;
    }else if(degreeButton&&!degreeButton.disabled){
      var degree=Number(degreeButton.dataset.shardEpDegree);
      if(degree===epDegree)return;
      epDegree=degree;
    }else return;
    selectedEpRank=0;
    render();
  });
  visual.addEventListener('click',function(event){
    if(stage!==6)return;
    var slot=event.target.closest('.shard-pipeline .shard-rank-slot');
    if(!slot||!visual.contains(slot))return;
    var rank=Number(slot.dataset.rank);
    selectedEpRank=selectedEpRank===rank?null:rank;
    updateEpSelection();
  });
  visual.addEventListener('keydown',function(event){
    if(stage!==6||(event.key!=='Enter'&&event.key!==' '))return;
    var slot=event.target.closest('.shard-pipeline .shard-rank-slot');
    if(!slot||!visual.contains(slot))return;
    event.preventDefault();
    slot.click();
  });
  window.RIShard={enter:function(){cancelTransition();render();},leave:function(){cancelTransition();destroy();}};
  if(page.dataset.tab==='shard')render();
})();
