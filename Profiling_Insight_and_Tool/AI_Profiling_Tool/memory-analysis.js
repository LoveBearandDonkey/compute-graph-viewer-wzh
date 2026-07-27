(function () {
  "use strict";
  var data = null;
  var viewer = null;

  function css(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }
  function canvas(id) {
    var el = document.getElementById(id), r = el && el.getBoundingClientRect();
    if (!el || !r || r.width < 10) return null;
    var dpr = window.devicePixelRatio || 1;
    el.width = Math.round(r.width * dpr); el.height = Math.round(r.height * dpr);
    var c = el.getContext("2d"); c.setTransform(dpr,0,0,dpr,0,0);
    return {c:c,w:r.width,h:r.height};
  }
  function drawTrend() {
    var g=canvas("memoryTrendChart"); if(!g||!data)return;
    var c=g.c,w=g.w,h=g.h,p={l:42,r:18,t:16,b:28}, rows=data.trend;
    var x=function(v){return p.l+(v-4000)/(12003-4000)*(w-p.l-p.r);};
    var y=function(v){return p.t+(64-v)/14*(h-p.t-p.b);};
    c.strokeStyle=css("--border-subtle","#ddd"); c.fillStyle=css("--foreground-secondary","#666"); c.font="11px sans-serif";
    [52,56,60,64].forEach(function(v){c.beginPath();c.moveTo(p.l,y(v));c.lineTo(w-p.r,y(v));c.stroke();c.fillText(v+"GB",3,y(v)+4);});
    c.beginPath(); rows.forEach(function(r,i){i?c.lineTo(x(r.step),y(r.reservedGB)):c.moveTo(x(r.step),y(r.reservedGB));});
    c.strokeStyle=css("--danger","#d33");c.lineWidth=2;c.stroke();
    c.fillStyle=css("--danger","#d33");c.beginPath();c.arc(x(12000),y(64),4,0,Math.PI*2);c.fill();c.fillText("OOM · step 12003",Math.max(p.l,x(12000)-105),y(64)+18);
  }
  function drawComposition() {
    var g=canvas("memoryCompositionChart"); if(!g||!data)return;
    var c=g.c,w=g.w,h=g.h, y=15, colors=["#d97706","#2563eb","#16a34a","#7c3aed","#64748b"];
    data.composition.forEach(function(r,i){var bw=(w-120)*r.gb/36.2;c.fillStyle=colors[i];c.fillRect(105,y,bw,22);c.fillStyle=css("--foreground","#111");c.font="11px sans-serif";c.fillText(r.label,4,y+15);c.fillText(r.gb+" GB",Math.min(w-42,110+bw),y+15);y+=39;});
  }
  function openTimeline(x) {
    openTab("timeline");
    var pane=document.getElementById("tab-timeline"), note=document.getElementById("memoryTimelineEvidence");
    if(pane&&!note){note=document.createElement("div");note.id="memoryTimelineEvidence";note.className="inspector-soft-card is-danger";note.style.margin="12px";pane.prepend(note);}
    if(note&&x)note.innerHTML="<strong>"+x.name+"</strong><br>"+x.allocMs+" ms 申请 "+x.sizeGB+" GB → "+x.freeMs+" ms 释放；持有 "+(x.freeMs-x.allocMs)+" ms。";
  }
  function openSource(x) {
    openTab("code");
    window.dispatchEvent(new CustomEvent("memory:open-code",{detail:{path:x&&x.source,line:x&&x.line}}));
  }
  function openTab(name) { var b=document.querySelector('.tab[data-tab="'+name+'"]'); if(b)b.click(); }
  function render() {
    if(!data)return;
    var s=data.summary;
    document.getElementById("memoryPeak").textContent=s.peakGB+"/"+s.capacityGB+" GB";
    document.getElementById("memoryLargestFree").textContent=s.largestFreeBlockGB+" GB / 空闲 "+s.freeGB+" GB";
    document.getElementById("memoryFragRatio").textContent=s.fragmentationRatio+"%";
    document.getElementById("memoryAllocRatio").textContent=s.allocApiRatio+"% · "+s.allocApiMs+" ms";
    drawTrend(); drawComposition();
    if(!viewer&&window.PtoHbmMemorySnapshot)viewer=window.PtoHbmMemorySnapshot.render(document.getElementById("memoryReuseViewer"),data,{
      initialSelectedId:"expert-dispatch",onOpenTimeline:openTimeline,onOpenSource:openSource
    });
  }
  fetch("data/openpangu-2.0-flash.memory-snapshot.json").then(function(r){return r.json();}).then(function(d){data=d;window.OPENPANGU_MEMORY_SNAPSHOT=d;render();});
  window.addEventListener("resize",function(){drawTrend();drawComposition();viewer&&viewer.resize&&viewer.resize();});
  window.addEventListener("msnext:ready",function(){
    var p=new URLSearchParams(location.search), tab=p.get("tab"), issue=p.get("issue");
    if(tab)openTab(tab);
    if(issue==="mem-oom") setTimeout(function(){openTab("memory");render();},0);
    document.getElementById("memoryTabBtn")?.addEventListener("click",function(){setTimeout(render,0);});
    document.getElementById("memoryTimelineBtn")?.addEventListener("click",function(){
      openTimeline(data&&data.lifetimes.find(function(x){return x.id==="expert-dispatch";}));
    });
    document.getElementById("memoryCodeBtn")?.addEventListener("click",function(){openSource(data&&data.lifetimes.find(function(x){return x.id==="expert-dispatch";}));});
  });
}());
