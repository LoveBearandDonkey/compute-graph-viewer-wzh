(function(){
  'use strict';

  var page=document.querySelector('.page');
  var visual=document.getElementById('shardVisual');
  if(!page||!visual)return;
  var title=document.getElementById('shardTitle');
  var body=document.getElementById('shardBody');
  var number=document.getElementById('shardStepNumber');
  var minRanks=document.getElementById('shardMinRanks');
  var rankBasis=document.getElementById('shardRankBasis');
  var steps=document.getElementById('shardSteps');
  var prev=document.getElementById('shardPrev');
  var next=document.getElementById('shardNext');
  var stage=0;
  var controllers=[];
  var activeTransition=null;
  var captions=[
    {name:'输入 BSH',body:'一个 Step 的输入是 B=4 个样本。每个样本是一张 S×H = 2048×2560 的矩阵；纵向是 token，横向是 hidden。',fact:'4 × [2048, 2560]'},
    {name:'DP × 2：分样本',body:'沿 B 维把 4 个样本分给两个 DP 分支。DP0 持有 B0、B1，DP1 持有 B2、B3；每张矩阵的 S 和 H 不变。',fact:'每个 DP：2 × [2048, 2560]'},
    {name:'CP × 2：分 token',body:'每个 DP 再沿 S 维分成 CP0、CP1。四张源矩阵同时横向划线，再各自展开两支。缩略图始终保留完整 16×20 网格：CP0 的上半部着绿，CP1 的下半部着绿。',fact:'每个 DP×CP：2 × [1024, 2560]'},
    {name:'TP × 2：分 hidden',body:'每个 CP 分支再沿 H 维分成 TP0、TP1。八张源矩阵同时纵向划线，再各自展开两支；绿色区域变为原矩阵的四分之一。这里先画 CP 后画 TP 只是讲解顺序：两者分别切 S 和 H，互换先后仍得到同样的八个分片。',fact:'每个 DP×CP×TP：2 × [1024, 1280]'},
    {name:'输入 Rank',body:'每条 DP→CP→TP 路径对应一个 Rank，共 2×2×2=8 个。每个 Rank 接收两个样本在对应 token 区间和 hidden 区间的分片。',fact:'8 个 Rank；每 Rank：2 × [1024, 1280]'},
    {name:'PP × 4：分模型层',body:'DP0、DP1 各自独立运行；每组 DP 都经过 PP0–PP3 四段，每段保留 4 个 CP×TP 坐标。以 48 层为例，PP0 负责 L0–11，PP1 负责 L12–23，依次传递微批激活。',fact:'2 组 DP × 4 段 PP × 4 个 CP×TP Rank = 32 个 Rank'},
    {name:'EP × 2：分专家',body:'本示例令 EP=2，从 DP=2 的两支切出：固定 PP、CP、TP 后，DP0 与 DP1 两张 Rank 组成一个 EP 组。以每个 MoE 层有 64 个专家、ETP=1 为例，EP0 持有 E0–31，EP1 持有 E32–63。Router 按专家选择动态分发 token；例如 R0 的 token 可以发给 R4 上的专家，计算后再合并回原 token 位置。Dense 层仍按原来的 DP×CP×TP 运行。',fact:'EP=2 从 DP=2 切出 · Rank 总数仍为 32'}
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

  function destroy(){controllers.forEach(function(c){c.destroy();});controllers=[];}
  function mountCanvases(){
    if(!window.PtoMatrixCanvas)return;
    visual.querySelectorAll('canvas[data-shard-matrix]').forEach(function(canvas){
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
    });
  }
  function input(){
    var html='<div class="shard-input-group"><div class="shard-input-row">';
    for(var b=0;b<4;b++){
      html+='<div class="shard-input-matrix"><div class="shard-input-caption"><strong>B'+b+'</strong></div><div class="shard-input-chart"><span class="shard-axis-s">S ↓<br>2048</span><div class="shard-input-host"><canvas data-shard-matrix="input" data-fragment-key="b'+b+'" aria-label="B'+b+'，S 2048 行、H 2560 列"></canvas></div><span class="shard-axis-h">H → 2560</span></div></div>';
    }
    return html+'</div></div>';
  }
  function node(col,start,span,kind,html,added){
    return '<div class="shard-tree-node is-'+kind+(added?' is-new':'')+'" style="grid-column:'+col+';grid-row:'+start+' / span '+span+'">'+html+'</div>';
  }
  function brace(col,start,span,label,added,axis){
    return '<div class="shard-tree-brace is-'+axis+(added?' is-new':'')+'" style="grid-column:'+col+';grid-row:'+start+' / span '+span+';--brace-quarter:'+span/4+'" role="img" aria-label="'+label+'"><svg viewBox="0 0 28 100" preserveAspectRatio="none" aria-hidden="true"><path d="M26 0 C12 0 12 24 12 34 C12 44 9 50 2 50"/><path d="M2 50 C9 50 12 56 12 66 C12 76 12 100 26 100"/></svg></div>';
  }
  function source(){
    return '<strong>输入 BSH</strong><span>B0 · B1 · B2 · B3</span><small>[4, 2048, 2560]</small>';
  }
  function fragment(b,cp,tp){
    var key='b'+b+(cp==null?'':'-cp'+cp)+(tp==null?'':'-tp'+tp);
    var rows=cp==null?2048:1024,columns=tp==null?2560:1280;
    return '<div class="shard-fragment"><span class="shard-fragment-label">B'+b+'</span><div class="shard-fragment-host"><canvas data-shard-matrix="fragment" data-fragment-key="'+key+'"'+(cp==null?'':' data-cp="'+cp+'"')+(tp==null?'':' data-tp="'+tp+'"')+' aria-label="B'+b+'，完整 16×20 网格；持有 S '+rows+' 行、H '+columns+' 列的分片"></canvas></div></div>';
  }
  function pair(dp,cp,tp){return '<div class="shard-fragment-pair">'+fragment(dp*2,cp,tp)+fragment(dp*2+1,cp,tp)+'</div>';}
  function rankCard(rank,dp,cp,tp,pp,row){
    var local=(dp*2+cp)*2+tp;
    var placement=row==null?'':' style="grid-column:7;grid-row:'+row+'"';
    return '<div class="shard-rank-slot"'+placement+' role="group" aria-label="Rank '+rank+'，DP'+dp+'，CP'+cp+'，TP'+tp+'，两张输入矩阵分片"><div class="shard-rank-label"><strong>R'+rank+'</strong><span>TP'+tp+'</span></div><div class="shard-tree-node is-rank shard-rank-card" data-rank-local="'+local+'"'+(pp==null?'':' data-pp="'+pp+'"')+'>'+pair(dp,cp,tp)+'</div></div>';
  }
  function tree(){
    var count=stage===1?2:stage===2?4:8;
    var html='<div class="shard-tree" data-stage="'+stage+'" style="--tree-rows:'+count+'">';
    html+=node(1,1,count,'root',source(),false);
    html+=brace(2,1,count,'沿 B 维分成 DP0 和 DP1',stage===1,'dp');
    for(var dp=0;dp<2;dp++){
      var dpSpan=count/2,dpStart=dp*dpSpan+1;
      html+=node(3,dpStart,dpSpan,'dp','<strong>DP'+dp+'</strong><span>B'+(dp*2)+' · B'+(dp*2+1)+'</span>'+(stage===1?pair(dp,null,null):'<small>2 × [2048, 2560]</small>'),stage===1);
      if(stage<2)continue;
      html+=brace(4,dpStart,dpSpan,'DP'+dp+' 沿 S 维分成 CP0 和 CP1',stage===2,'cp');
      for(var cp=0;cp<2;cp++){
        var cpSpan=count/4,cpStart=dpStart+cp*cpSpan;
        html+=node(5,cpStart,cpSpan,'cp','<strong>CP'+cp+'</strong><span>S['+(cp*1024)+'~'+((cp+1)*1024-1)+']</span>'+(stage===2?pair(dp,cp,null):'<small>2 × [1024, 2560]</small>'),stage===2);
        if(stage<3)continue;
        html+=brace(6,cpStart,cpSpan,'DP'+dp+' CP'+cp+' 沿 H 维分成 TP0 和 TP1',stage===3,'tp');
        for(var tp=0;tp<2;tp++){
          var rank=(dp*2+cp)*2+tp,rankStart=cpStart+tp;
          if(stage===4)html+=rankCard(rank,dp,cp,tp,null,rankStart);
          else html+=node(7,rankStart,1,'tp','<strong>TP'+tp+'</strong><span>H['+(tp*1280)+'~'+((tp+1)*1280-1)+']</span>'+pair(dp,cp,tp),stage===3);
        }
      }
    }
    return html+'</div>';
  }
  function pipeline(){
    var html='<div class="shard-pipeline" data-stage="'+stage+'"><div class="shard-pipeline-intro">'+pipelineIntro()+'</div><div class="shard-pp-guide">';
    for(var pp=0;pp<4;pp++){
      html+='<div class="shard-pp-heading"><strong>PP'+pp+'</strong><span>L'+pp*12+'–'+(pp*12+11)+'</span></div>';
    }
    html+='</div>';
    for(var dp=0;dp<2;dp++){
      html+='<div class="shard-dp-lane" role="group" aria-label="DP'+dp+' 独立运行的 16 个 Rank"><div class="shard-dp-heading"><strong>DP'+dp+'</strong><span>B'+(dp*2)+' · B'+(dp*2+1)+'</span><small>EP'+dp+' · E'+(dp*32)+'–'+(dp*32+31)+'</small></div><div class="shard-dp-columns">';
      for(var pp=0;pp<4;pp++){
        html+='<div class="shard-dp-column">';
        for(var cp=0;cp<2;cp++){
          html+='<div class="shard-cp-pair"><span class="shard-cp-label">CP'+cp+'</span><div class="shard-tp-pair">';
          for(var tp=0;tp<2;tp++){
            var rank=pp*8+(dp*2+cp)*2+tp;
            html+=rankCard(rank,dp,cp,tp,pp);
          }
          html+='</div></div>';
        }
        html+='</div>';
      }
      html+='</div></div>';
    }
    return html+'<div class="shard-pipeline-tail">'+pipelineTail()+'</div></div>';
  }
  function pipelineIntro(){return stage===6?'同一 PP/CP/TP 位置上，DP0 与 DP1 配成 EP 组；两组 DP 仍各自运行':'两组 DP 独立运行；每组按 PP0→PP3 经过不同模型层';}
  function pipelineTail(){return stage===6?'例：R0 的 token 可发到 R4 的 E32–63，专家计算后再合并回原 token 位置':'PP 相邻段顺序传递微批激活';}
  function render(){
    destroy();
    var current=captions[stage];
    number.textContent=String(stage+1).padStart(2,'0')+' / '+String(captions.length).padStart(2,'0');
    var factors=[],rankCount=1;
    if(stage>=1){factors.push('DP 2');rankCount*=2;}
    if(stage>=2){factors.push('CP 2');rankCount*=2;}
    if(stage>=3){factors.push('TP 2');rankCount*=2;}
    if(stage>=5){factors.push('PP 4');rankCount*=4;}
    minRanks.textContent=String(rankCount);
    rankBasis.textContent='（'+(factors.length?factors.join(' × '):'尚未切分')+(stage===6?'；EP 复用 DP':'')+'）';
    title.textContent=current.name;
    body.innerHTML='<p>'+current.body+'</p><div class="shard-fact">'+current.fact+'</div>';
    visual.dataset.step=String(stage);
    if(stage>=5){
      var existing=visual.querySelector('.shard-pipeline');
      if(existing){
        existing.dataset.stage=String(stage);
        existing.querySelector('.shard-pipeline-intro').textContent=pipelineIntro();
        existing.querySelector('.shard-pipeline-tail').textContent=pipelineTail();
      }
      else visual.innerHTML=pipeline();
    }else visual.innerHTML=stage===0?input():tree();
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
    if(stage===4&&nextStage===5&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches){transitionToPP();return;}
    if((stage===0&&nextStage===1)||(stage===1&&nextStage===2)||(stage===2&&nextStage===3)||(stage===3&&nextStage===4)){
      if(!window.matchMedia('(prefers-reduced-motion: reduce)').matches){transitionTo(nextStage);return;}
    }
    stage=nextStage;render();
  }
  prev.addEventListener('click',function(){setStage(stage-1);});
  next.addEventListener('click',function(){setStage(stage+1);});
  steps.addEventListener('click',function(event){var button=event.target.closest('[data-shard-step]');if(button)setStage(Number(button.dataset.shardStep));});
  window.RIShard={enter:function(){cancelTransition();render();},leave:function(){cancelTransition();destroy();}};
  if(page.dataset.tab==='shard')render();
})();
