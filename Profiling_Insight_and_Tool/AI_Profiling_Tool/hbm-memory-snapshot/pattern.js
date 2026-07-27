(function (global) {
  "use strict";
  function esc(v){return String(v==null?"":v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;");}
  function kind(v){
    var k=String(v||"").toLowerCase();
    if(k==="temp")return "workspace";
    if(k==="resident")return "parameters";
    return k||"workspace";
  }
  function label(k){return ({activation:"激活",parameters:"参数",gradients:"梯度",optimizer:"优化器",workspace:"临时空间"})[k]||k;}
  function render(container,snapshot,options){
    options=options||{}; var s=snapshot.summary||{}, rows=snapshot.lifetimes||[], selected=options.initialSelectedId||rows[0]?.id;
    var maxT=Math.max(1,...rows.map(function(x){return Number(x.freeMs)||0;}));
    container.innerHTML='<section class="pto-hbm-snapshot"><div class="pto-hbm-snapshot__evidence">'+
      fact("HBM 峰值",(s.peakGB||0)+"/"+(s.capacityGB||0)+" GB",true)+fact("空闲总量",(s.freeGB||0)+" GB")+
      fact("最大连续空闲块",(s.largestFreeBlockGB||0)+" GB",true)+fact("下一次申请",(s.requestedGB||0)+" GB",true)+
      '</div><div class="pto-hbm-snapshot__verdict"><b>为什么 OOM：</b>空闲总量足够，但最大连续块 '+esc(s.largestFreeBlockGB)+' GB 小于请求 '+esc(s.requestedGB)+' GB；碎片率 '+esc(s.fragmentationRatio)+'%。</div>'+
      '<div class="pto-hbm-snapshot__body"><div class="pto-hbm-snapshot__plots">'+
      '<section class="pto-hbm-snapshot__section"><div class="pto-hbm-snapshot__title">关键分配的地址位置 <span class="pto-hbm-snapshot__hint">色块=关键块 · 斜纹=未展开地址段/空洞 · 点击查看来源</span></div>'+
      '<div class="pto-hbm-snapshot__address">'+rows.map(function(x){return '<button class="pto-hbm-snapshot__address-block" data-id="'+esc(x.id)+'" data-kind="'+kind(x.kind)+'" style="left:'+(x.offsetGB/s.capacityGB*100)+'%;width:'+(x.sizeGB/s.capacityGB*100)+'%" title="'+esc(x.name)+' · '+esc(x.sizeGB)+' GB"></button>';}).join("")+'</div>'+
      '<div class="pto-hbm-snapshot__axis"><span>0 GB</span><span>'+esc(s.capacityGB/2)+' GB</span><span>'+esc(s.capacityGB)+' GB</span></div>'+
      '<div class="pto-hbm-snapshot__legend">'+["activation","parameters","gradients","optimizer","workspace"].map(function(k){return '<span><i style="background:var(--hbm-'+k+')"></i>'+label(k)+'</span>';}).join("")+'<span><i class="is-gap"></i>未展开/空洞</span></div></section>'+
      '<section class="pto-hbm-snapshot__section"><div class="pto-hbm-snapshot__title">生命周期 <span class="pto-hbm-snapshot__hint">左=forward 开始 · 右=backward 结束</span></div><div class="pto-hbm-snapshot__rows">'+
      rows.slice().sort(function(a,b){return (b.freeMs-b.allocMs)-(a.freeMs-a.allocMs);}).map(function(x){var d=x.freeMs-x.allocMs;return '<div class="pto-hbm-snapshot__row"><span class="pto-hbm-snapshot__row-label" title="'+esc(x.name)+'">'+esc(x.name)+'</span><div class="pto-hbm-snapshot__track"><button class="pto-hbm-snapshot__life pto-hbm-snapshot__address-block" data-id="'+esc(x.id)+'" data-kind="'+kind(x.kind)+'" style="left:'+(x.allocMs/maxT*100)+'%;width:'+(d/maxT*100)+'%"></button></div><span class="pto-hbm-snapshot__duration">'+esc(d)+' ms</span></div>';}).join("")+
      '</div><div class="pto-hbm-snapshot__axis"><span>0 ms</span><span>'+Math.round(maxT/2)+' ms</span><span>'+maxT+' ms</span></div></section></div><aside class="pto-hbm-snapshot__detail"></aside></div></section>';
    var root=container.firstElementChild, detail=root.querySelector(".pto-hbm-snapshot__detail");
    function select(id){
      selected=id; root.querySelectorAll("[data-id]").forEach(function(el){el.classList.toggle("is-selected",el.dataset.id===id);});
      var x=rows.find(function(v){return String(v.id)===String(id);})||rows[0]; if(!x)return;
      detail.innerHTML='<h3>'+esc(x.name)+'</h3><div class="pto-hbm-snapshot__detail-kind">'+label(kind(x.kind))+' · '+esc(x.source||"未知来源")+(x.line?":"+x.line:"")+'</div>'+
        '<div class="pto-hbm-snapshot__kv"><div><span>地址</span><b>'+esc(x.offsetGB)+'–'+esc((x.offsetGB+x.sizeGB).toFixed(3))+' GB</b></div><div><span>大小</span><b>'+esc(x.sizeGB)+' GB</b></div><div><span>申请</span><b>'+esc(x.allocMs)+' ms</b></div><div><span>释放</span><b>'+esc(x.freeMs)+' ms</b></div><div><span>持有</span><b>'+esc(x.freeMs-x.allocMs)+' ms</b></div></div>'+
        '<div class="pto-hbm-snapshot__stack"><b>调用栈</b><br>'+esc(x.stack||"暂无调用栈")+'</div><div class="pto-hbm-snapshot__actions"><button class="btn btn-ghost btn-sm" data-action="timeline">Timeline</button><button class="btn btn-ghost btn-sm" data-action="source">源码</button></div>';
    }
    root.addEventListener("click",function(e){var block=e.target.closest("[data-id]");if(block)select(block.dataset.id);var action=e.target.closest("[data-action]");if(!action)return;var x=rows.find(function(v){return String(v.id)===String(selected);});if(action.dataset.action==="timeline")options.onOpenTimeline?.(x);else options.onOpenSource?.(x);});
    select(selected);
    return {select:select,resize:function(){},destroy:function(){container.innerHTML="";}};
  }
  function fact(name,value,danger){return '<div class="pto-hbm-snapshot__fact'+(danger?" is-danger":"")+'"><span>'+name+'</span><strong>'+esc(value)+'</strong></div>';}
  global.PtoHbmMemorySnapshot={render:render};
})(window);
