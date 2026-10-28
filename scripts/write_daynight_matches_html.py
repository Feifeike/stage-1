from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads(args.matches.read_text())
    for sample in samples:
        sample["frontImage"] = Path(sample["frontImage"]).resolve().as_uri()
        sample["satelliteImage"] = Path(sample["satelliteImage"]).resolve().as_uri()
    summary = json.loads(args.metrics.read_text())
    data = json.dumps({"summary": summary, "samples": samples}, ensure_ascii=True).replace("<", "\\u003c")
    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Stage-1 day-night matches</title>
<style>body{margin:0;padding:18px;background:#f5f7fb;color:#172033;font-family:Arial,sans-serif}.page{max-width:1800px;margin:auto;display:grid;gap:14px}.card{background:#fff;border:1px solid #d0d5dd;border-radius:10px;padding:14px}.toolbar{display:grid;grid-template-columns:minmax(260px,1fr) auto auto minmax(260px,1fr);gap:10px;align-items:end}select,button{height:34px;border:1px solid #d0d5dd;border-radius:6px;background:#fff;padding:0 9px}button{cursor:pointer}label{display:block;color:#667085;font-size:12px;margin-bottom:5px}.viewer{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:14px}.title{margin:0 0 7px;color:#667085;font-size:13px}.canvas{position:relative;overflow:hidden;border:1px solid #d0d5dd;border-radius:7px}.canvas img{display:block;width:100%;height:auto}svg{position:absolute;inset:0;width:100%;height:100%}.stats{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}.stat{background:#f8fafc;border-radius:6px;padding:7px;color:#667085;font-size:12px}.stat strong{display:block;color:#172033;font-size:14px;margin-top:3px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border-bottom:1px solid #eaecf0;padding:6px 5px;text-align:right}th:first-child,td:first-child{text-align:left}.legend{color:#667085;font-size:12px;margin-top:8px}@media(max-width:1000px){.viewer{grid-template-columns:1fr}.toolbar{grid-template-columns:1fr 1fr}.stats{grid-template-columns:repeat(3,1fr)}}</style></head><body><div class="page">
<div class="card"><h2 style="margin:0 0 4px">Stage-1 day-night front-to-satellite matching</h2><div style="color:#667085;font-size:13px">N=20 · predicted anchor camera z ≤ 80 m · 400 sampled frames</div></div>
<div class="card toolbar"><div><label for="sampleSelect">选择帧</label><select id="sampleSelect"></select></div><button id="prev">上一帧</button><button id="next">下一帧</button><div id="info" style="color:#667085;font-size:13px"></div></div><div class="card stats" id="summary"></div>
<div class="card viewer"><section><p class="title">前视图：选中的距离内锚点</p><div class="canvas"><img id="front"><svg id="frontSvg"></svg></div></section><section><p class="title">卫星图：GT / coarse / refine</p><div class="canvas"><img id="sat"><svg id="satSvg"></svg></div></section></div>
<div class="card"><div class="legend">绿色 GT；橙色 coarse；红色 refine。点击前视图锚点查看对应卫星位置。</div><div id="details"></div></div></div><script>
const data=__DATA__;let si=0,pi=0;const sel=document.getElementById('sampleSelect'),front=document.getElementById('front'),sat=document.getElementById('sat'),fs=document.getElementById('frontSvg'),ss=document.getElementById('satSvg'),info=document.getElementById('info'),details=document.getElementById('details');
function clear(x){while(x.firstChild)x.removeChild(x.firstChild)}function svg(n,a){let x=document.createElementNS('http://www.w3.org/2000/svg',n);for(const[k,v]of Object.entries(a))x.setAttribute(k,v);return x}function cross(s,x,y,c,r,w){s.appendChild(svg('path',{d:`M ${x-r} ${y} L ${x+r} ${y} M ${x} ${y-r} L ${x} ${y+r}`,stroke:c,'stroke-width':w,'stroke-linecap':'round'}))}function dot(s,x,y,c,r,click){let e=svg('circle',{cx:x,cy:y,r:r,fill:c,stroke:'#fff','stroke-width':1.5,cursor:'pointer'});e.onclick=click;s.appendChild(e)}function fmt(x){return x===null||x===undefined?'-':Number(x).toFixed(2)}
function draw(){let s=data.samples[si],p=s.anchors.length?s.anchors[pi%s.anchors.length]:null;sel.value=si;front.src=s.frontImage;sat.src=s.satelliteImage;fs.setAttribute('viewBox',`0 0 ${s.frontSize.width} ${s.frontSize.height}`);ss.setAttribute('viewBox',`0 0 ${s.satelliteSize.width} ${s.satelliteSize.height}`);clear(fs);clear(ss);info.textContent=`${si+1} / ${data.samples.length} · ${s.sampleId} · anchors ${s.anchors.length}`;s.anchors.forEach((a,i)=>dot(fs,a.x,a.y,i===pi?'#d84315':'rgba(216,67,21,.65)',i===pi?8:5,()=>{pi=i;draw()}));if(p){if(p.gt_x!==null)cross(ss,p.gt_x,p.gt_y,'#20c997',8,3);cross(ss,p.coarse_x,p.coarse_y,'#f59f00',7,2);cross(ss,p.pred_x,p.pred_y,'#fa2b2b',8,3);if(p.gt_x!==null)ss.appendChild(svg('line',{x1:p.gt_x,y1:p.gt_y,x2:p.pred_x,y2:p.pred_y,stroke:'rgba(52,152,219,.9)','stroke-width':2,'stroke-dasharray':'5 4'}))}details.innerHTML=p?`<table><thead><tr><th>anchor</th><th>matching score</th><th>certainty</th><th>pred z(m)</th><th>coarse(px)</th><th>refine(px)</th><th>GT</th></tr></thead><tbody>${s.anchors.map((a,i)=>`<tr style="background:${i===pi?'#fff4e6':'transparent'}"><td>${i+1}</td><td>${fmt(a.matching_score)}</td><td>${fmt(a.certainty ?? a.confidence)}</td><td>${fmt(a.anchor_z_m)}</td><td>${fmt(a.coarse_error_px)}</td><td>${fmt(a.refine_error_px)}</td><td>${a.gt_valid}</td></tr>`).join('')}</tbody></table>`:'<div>该帧没有 z≤80m 的有效预测锚点</div>'}
function showSummary(){let s=data.summary,items=[['frames',s.samples_evaluated],['GT queries',s.refine_valid_queries],['coarse px',fmt(s.coarse_pixel_error_global)],['refine px',fmt(s.refine_pixel_error_global)],['coarse R@1',fmt(s.coarse_recall_at_1)],['refine ≤10px',fmt(s.refine_within_10px_fraction)]];document.getElementById('summary').innerHTML=items.map(x=>`<div class="stat">${x[0]}<strong>${x[1]}</strong></div>`).join('')}
;data.samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${i+1} · ${s.sampleId}`;sel.appendChild(o)});sel.onchange=()=>{si=Number(sel.value);pi=0;draw()};document.getElementById('prev').onclick=()=>{si=(si+data.samples.length-1)%data.samples.length;pi=0;draw()};document.getElementById('next').onclick=()=>{si=(si+1)%data.samples.length;pi=0;draw()};showSummary();draw();
</script></body></html>""".replace("__DATA__", data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
