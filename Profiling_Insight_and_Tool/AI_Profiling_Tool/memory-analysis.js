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
    var c=g.c,w=g.w,h=g.h,p={l:46,r:18,t:16,b:28},rows=data.trend;
    var capacity=Number(data.summary.capacityGB)||64,threshold=capacity*.95,yMin=50;
    var minStep=rows[0].step,maxStep=rows[rows.length-1].step;
    var x=function(v){return p.l+(v-minStep)/(maxStep-minStep)*(w-p.l-p.r);};
    var y=function(v){return p.t+(capacity-v)/(capacity-yMin)*(h-p.t-p.b);};
    var primary=css("--primary","#4369ef"),danger=css("--danger","#d33"),warning=css("--warning","#d88a00");
    function linePath(){
      c.beginPath();
      rows.forEach(function(r,i){i?c.lineTo(x(r.step),y(r.reservedGB)):c.moveTo(x(r.step),y(r.reservedGB));});
    }
    function areaPath(){
      c.beginPath();c.moveTo(x(rows[0].step),h-p.b);
      rows.forEach(function(r){c.lineTo(x(r.step),y(r.reservedGB));});
      c.lineTo(x(rows[rows.length-1].step),h-p.b);c.closePath();
    }
    c.strokeStyle=css("--border-subtle","#ddd");c.fillStyle=css("--foreground-secondary","#666");c.font="11px sans-serif";c.lineWidth=1;
    [52,56,60,64].forEach(function(v){c.beginPath();c.moveTo(p.l,y(v));c.lineTo(w-p.r,y(v));c.stroke();c.fillText(v+"GB",3,y(v)+4);});
    c.textAlign="center";
    var lastLabelX=-Infinity;
    rows.forEach(function(r){var px=x(r.step);if(px-lastLabelX<32)return;lastLabelX=px;c.fillText("step "+r.step,px,h-p.b+16);});
    c.textAlign="start";
    c.save();c.setLineDash([4,4]);c.strokeStyle=warning;c.beginPath();c.moveTo(p.l,y(threshold));c.lineTo(w-p.r,y(threshold));c.stroke();c.restore();
    c.fillStyle=warning;c.font="10px sans-serif";c.fillText("95% 阈值 · "+threshold.toFixed(1)+" GB",p.l+5,y(threshold)-5);
    c.save();c.globalAlpha=.13;c.fillStyle=primary;areaPath();c.fill();c.restore();
    c.lineJoin="round";c.lineCap="round";c.strokeStyle=primary;c.lineWidth=2;linePath();c.stroke();
    c.save();c.beginPath();c.rect(p.l,p.t,w-p.l-p.r,Math.max(0,y(threshold)-p.t));c.clip();
    c.globalAlpha=.16;c.fillStyle=danger;areaPath();c.fill();c.globalAlpha=1;c.strokeStyle=danger;c.lineWidth=2;linePath();c.stroke();c.restore();
    var incident=rows[rows.length-2]||rows[rows.length-1];
    c.fillStyle=danger;c.beginPath();c.arc(x(incident.step),y(incident.reservedGB),4,0,Math.PI*2);c.fill();
    c.font="11px sans-serif";c.fillText("OOM · step "+(data.incidentStep||12003),Math.max(p.l,x(incident.step)-105),y(incident.reservedGB)+18);
  }
  function drawComposition() {
    var g=canvas("memoryCompositionChart"); if(!g||!data)return;
    var c=g.c,w=g.w,h=g.h, y=15, colors=["#d97706","#3478d4","#159c58","#8b5cc7","#738095"];
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
    document.getElementById("memoryLargestFree").innerHTML=s.largestFreeBlockGB+' GB <span class="memory-analysis__aux">/ 空闲 '+s.freeGB+" GB</span>";
    document.getElementById("memoryFragRatio").textContent=s.fragmentationRatio+"%";
    document.getElementById("memoryAllocRatio").textContent=s.allocApiRatio+"% · "+s.allocApiMs+" ms";
    drawTrend(); drawComposition();
    if(!viewer&&window.PtoHbmMemorySnapshot)viewer=window.PtoHbmMemorySnapshot.render(document.getElementById("memoryReuseViewer"),data,{
      onOpenTimeline:openTimeline,onOpenSource:openSource
    });
  }
  fetch("data/openpangu-2.0-flash.memory-snapshot.json").then(function(r){return r.json();}).then(function(d){data=d;window.OPENPANGU_MEMORY_SNAPSHOT=d;render();});
  window.addEventListener("resize",function(){drawTrend();drawComposition();viewer&&viewer.resize&&viewer.resize();});
  window.addEventListener("msnext:ready",function(){
    var p=new URLSearchParams(location.search), tab=p.get("tab"), issue=p.get("issue");
    if(tab)openTab(tab);
    if(issue==="mem-oom") setTimeout(function(){openTab("memory");render();},0);
    document.getElementById("memoryTabBtn")?.addEventListener("click",function(){setTimeout(render,0);});
  });
}());
