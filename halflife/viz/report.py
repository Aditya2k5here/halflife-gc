"""Build ``results/report.html`` -- a single self-contained page, no CDN, no build step.

Everything is inlined because a results page that stops rendering the day a CDN
changes a URL is not a result. The charts are plain SVG generated at load time
from the embedded JSON, and the heap player is a canvas replaying the event tape
captured by :mod:`halflife.viz.film`.
"""

import json
import os

from ..eval.run import RESULTS_DIR
from .film import capture_pair

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>halflife-gc &mdash; results</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --line:#262b35; --ink:#e7eaf0; --dim:#98a2b3;
    --eden:#f0a03c; --old:#5aa9e6; --pre:#8b7ff0; --dead:#2a2f3a;
    --good:#4ec9a5; --bad:#e0655f;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:40px 20px 80px}
  h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
  h2{font-size:17px;margin:44px 0 6px;letter-spacing:-.01em}
  p{color:var(--dim);margin:6px 0 14px;max-width:70ch}
  .lede{color:var(--ink);font-size:15px}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;
       background:#1d222b;padding:1px 5px;border-radius:4px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
         padding:16px 18px;margin:14px 0}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);font-size:13px}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--dim);font-weight:500}
  .win{color:var(--good)} .loss{color:var(--bad)}
  .legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--dim);font-size:12px;margin:8px 0 0}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
      vertical-align:-1px}
  .films{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:780px){.films{grid-template-columns:1fr}}
  canvas{width:100%;height:auto;image-rendering:pixelated;background:#0b0d11;
         border:1px solid var(--line);border-radius:6px;display:block}
  .filmhead{display:flex;justify-content:space-between;align-items:baseline;
            font-size:13px;margin-bottom:6px}
  .filmhead b{font-weight:600}
  .controls{display:flex;gap:12px;align-items:center;margin-top:12px;color:var(--dim);
            font-size:13px}
  button{background:#232936;color:var(--ink);border:1px solid var(--line);
         border-radius:6px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}
  button:hover{background:#2b3342}
  input[type=range]{accent-color:var(--old)}
  .foot{color:var(--dim);font-size:12px;margin-top:40px;border-top:1px solid var(--line);
        padding-top:14px}
</style>
</head>
<body>
<div class="wrap">

<h1>halflife-gc</h1>
<p class="lede">A generational garbage collector that predicts, before it allocates an object,
whether that object will still be alive when the nursery fills up &mdash; and puts the ones
that will straight into the old generation, so they are never copied.</p>
<p>Every number below is a deterministic event count from a simulator, not a wall-clock
measurement. <b>GC work</b> = bytes&nbsp;traced + 2&times;bytes&nbsp;copied +
0.1&times;bytes&nbsp;swept. Lower is better. The model is trained on a
<i>profiling run</i> of each program and scored on a <i>different run</i> with different
input size and data.</p>

<h2>GC work, relative to a stock generational collector</h2>
<p>100% is the stock generational collector on that workload. <span class="win">Below 100%</span>
is a win. The oracle reads true object lifetimes off a recorded trace and is unshippable
by construction &mdash; it is here to say how much there was to win at all.</p>
<div class="panel" id="bars"></div>

<h2>How much of the available headroom the predictor recovered</h2>
<p>The gap between the stock collector and the oracle is the prize. This is the share of it
each policy actually collected.</p>
<div class="panel" id="recovered"></div>

<h2>The policy family: how long must an object live before pretenuring pays?</h2>
<p>Pretenure an object if it will live at least <code>K</code> nurseries.
<code>K&nbsp;=&nbsp;0</code> pretenures everything; large <code>K</code> pretenures nothing.
Each curve is the perfect-knowledge collector swept over <code>K</code>, which shows both
where the optimum sits and how sharp it is &mdash; a flat curve means the exact threshold
barely matters, a sharp one means a predictor has to be right about <i>when</i>, not just
<i>whether</i>.</p>
<div class="panel" id="kcurve"></div>

<h2>The heap, playing</h2>
<p>The same program, the same allocations, in the same order &mdash; twice. On the left, every
object is born in the nursery and the survivors are copied out. On the right, the oracle puts
the long-lived ones straight into the old generation. Watch the nursery: on the left it keeps
handing objects over; on the right almost nothing survives it, which is exactly what a nursery
is supposed to look like.</p>
<div class="controls">
  <button id="play">Pause</button>
  <label>speed <input id="speed" type="range" min="1" max="60" value="18"></label>
  <span id="clock"></span>
</div>
<div class="films" id="films"></div>
<div class="legend">
  <span><i class="sw" style="background:var(--eden)"></i>nursery</span>
  <span><i class="sw" style="background:var(--old)"></i>old generation (promoted)</span>
  <span><i class="sw" style="background:var(--pre)"></i>old generation (pretenured at birth)</span>
  <span><i class="sw" style="background:var(--dead)"></i>free</span>
</div>

<h2>Prediction quality vs collector outcome</h2>
<p>These two things come apart, and that is the interesting part. A model can have a
respectable AUC and still make the collector worse, because the two mistakes do not cost the
same: failing to pretenure a survivor costs one copy, while pretenuring garbage puts it in a
space that is only collected by a full trace.</p>
<div class="panel"><table id="model"></table></div>

<h2>Every number</h2>
<div class="panel" style="overflow-x:auto"><table id="full"></table></div>

<div class="foot" id="foot"></div>
</div>

<script id="results-data" type="application/json">__RESULTS__</script>
<script id="film-data" type="application/json">__FILM__</script>
<script>
const RESULTS = JSON.parse(document.getElementById('results-data').textContent);
const FILM = JSON.parse(document.getElementById('film-data').textContent);
const SVGNS = 'http://www.w3.org/2000/svg';

const COLORS = {
  'mark-sweep':'#6b7280', 'semispace':'#8a6d3b', 'generational':'#98a2b3',
  'gen+always':'#7a6ea8', 'gen+sitetable':'#e0a44b', 'gen+learned':'#4ec9a5',
  'gen+transfer':'#3f8f78', 'gen+oracle':'#5aa9e6'
};
const ORDER = ['mark-sweep','semispace','generational','gen+always','gen+sitetable',
               'gen+learned','gen+transfer','gen+oracle'];

function el(tag, attrs, kids){
  const n = document.createElementNS(SVGNS, tag);
  for(const k in (attrs||{})) n.setAttribute(k, attrs[k]);
  (kids||[]).forEach(c => n.appendChild(c));
  return n;
}
function txt(s){ return document.createTextNode(s); }
function fmtPct(x){ return (x*100).toFixed(0) + '%'; }

function rowsFor(w){
  const map = {};
  w.collectors.forEach(c => map[c.label] = c.gc_work);
  const t = RESULTS.transfer && RESULTS.transfer.per_workload[w.name];
  if(t) map['gen+transfer'] = t.stats.gc_work;
  return ORDER.filter(l => l in map).map(l => ({label:l, work:map[l]}));
}

// ---------------------------------------------------------------- bar chart
function drawBars(){
  const host = document.getElementById('bars');
  const W = 980, rowH = 19, pad = 150, groupPad = 26;
  let y = 14;
  const svg = el('svg', {viewBox:'0 0 '+W+' 10', width:'100%'});
  RESULTS.workloads.forEach(w => {
    const rows = rowsFor(w);
    const base = rows.find(r => r.label === 'generational').work;
    // Clamp the axis: semispace is off the scale on some workloads and would
    // squash every interesting difference into a few pixels.
    const CLIP = 2.0;
    const max = Math.min(Math.max(...rows.map(r => r.work / base), 1) * 1.06, CLIP);
    svg.appendChild(el('text',{x:0,y:y,fill:'#e7eaf0','font-size':13,
      'font-weight':600},[txt(w.name)]));
    y += 8;
    rows.forEach(r => {
      const rel = r.work / base;
      const clipped = rel > max;
      const wpx = Math.max(1, (W - pad - 60) * Math.min(rel, max) / max);
      svg.appendChild(el('text',{x:0,y:y+11,fill:'#98a2b3','font-size':12},[txt(r.label)]));
      svg.appendChild(el('rect',{x:pad,y:y+1,width:wpx,height:rowH-6,rx:2,
        fill:COLORS[r.label]||'#666'}));
      if(clipped){
        svg.appendChild(el('path',{d:'M'+(pad+wpx-6)+' '+(y+1)+'l6 '+((rowH-6)/2)+'l-6 '+((rowH-6)/2),
          fill:'none',stroke:'#0f1115','stroke-width':2}));
      }
      svg.appendChild(el('text',{x:pad+wpx+7,y:y+12,'font-size':11,
        fill: rel < 0.999 ? '#4ec9a5' : (rel > 1.001 ? '#e0655f' : '#98a2b3')},
        [txt(fmtPct(rel))]));
      y += rowH;
    });
    const x100 = pad + (W - pad - 60) / max;
    svg.appendChild(el('line',{x1:x100,y1:y-rows.length*rowH,x2:x100,y2:y,
      stroke:'#3a4150','stroke-dasharray':'2 3'}));
    y += groupPad;
  });
  svg.setAttribute('viewBox','0 0 '+W+' '+y);
  host.appendChild(svg);
}

// ------------------------------------------------------- recovered headroom
function drawRecovered(){
  const host = document.getElementById('recovered');
  const W = 980, rowH = 34, pad = 150, barW = W - pad - 90;
  let y = 6;
  const svg = el('svg', {viewBox:'0 0 '+W+' 10', width:'100%'});
  RESULTS.workloads.forEach(w => {
    const rec = w.recovered['gen+learned'];
    const site = w.recovered['gen+sitetable'];
    svg.appendChild(el('text',{x:0,y:y+16,fill:'#e7eaf0','font-size':13},[txt(w.name)]));
    svg.appendChild(el('rect',{x:pad,y:y+4,width:barW,height:14,rx:3,fill:'#232936'}));
    if(rec !== null){
      const clamped = Math.max(-1, Math.min(1.2, rec));
      const wpx = Math.abs(clamped) * barW / 1.2;
      svg.appendChild(el('rect',{x: clamped>=0 ? pad : Math.max(pad, pad - wpx),
        y:y+4, width:wpx, height:14, rx:3,
        fill: clamped>=0 ? '#4ec9a5' : '#e0655f'}));
    }
    if(site !== null){
      const c = Math.max(-1, Math.min(1.2, site));
      svg.appendChild(el('line',{x1:pad + c*barW/1.2, x2:pad + c*barW/1.2,
        y1:y, y2:y+22, stroke:'#e0a44b','stroke-width':2}));
    }
    svg.appendChild(el('text',{x:pad+barW+8,y:y+16,'font-size':12,
      fill: (rec||0) >= 0 ? '#4ec9a5' : '#e0655f'},
      [txt(rec === null ? 'n/a' : fmtPct(rec))]));
    y += rowH;
  });
  svg.setAttribute('viewBox','0 0 '+W+' '+(y+6));
  host.appendChild(svg);
  const l = document.createElement('div');
  l.className = 'legend';
  l.innerHTML = '<span><i class="sw" style="background:#4ec9a5"></i>learned model</span>' +
                '<span><i class="sw" style="background:#e0a44b"></i>per-site table (tick)</span>';
  host.appendChild(l);
}

// ---------------------------------------------------------------- K curves
function drawK(){
  const host = document.getElementById('kcurve');
  const W = 980, H = 260, m = {l:56, r:120, t:14, b:34};
  const svg = el('svg',{viewBox:'0 0 '+W+' '+H, width:'100%'});
  const ks = RESULTS.k_values;
  const x = i => m.l + i * (W - m.l - m.r) / (ks.length - 1);
  const palette = ['#5aa9e6','#4ec9a5','#e0a44b','#e0655f','#8b7ff0','#98a2b3'];

  svg.appendChild(el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#2a3040'}));
  ks.forEach((k,i) => {
    svg.appendChild(el('text',{x:x(i),y:H-m.b+16,fill:'#98a2b3','font-size':11,
      'text-anchor':'middle'},[txt('K=' + k)]));
  });
  svg.appendChild(el('text',{x:m.l-44,y:m.t+10,fill:'#98a2b3','font-size':11},
    [txt('GC work')]));
  svg.appendChild(el('text',{x:m.l-44,y:H-m.b,fill:'#98a2b3','font-size':11},[txt('0')]));

  RESULTS.workloads.forEach((w,wi) => {
    const curve = w.policy_family.eval_curve;
    const base = w.collectors.find(c => c.label==='generational').gc_work;
    const y = v => m.t + (H-m.t-m.b) * (1 - Math.min(1.35, v/base)/1.35);
    let d = '';
    curve.forEach((pt,i) => { d += (i?'L':'M') + x(i) + ' ' + y(pt.gc_work); });
    svg.appendChild(el('path',{d:d, fill:'none', stroke:palette[wi%palette.length],
      'stroke-width':2}));
    curve.forEach((pt,i) => {
      svg.appendChild(el('circle',{cx:x(i),cy:y(pt.gc_work),r:2.5,
        fill:palette[wi%palette.length]}));
    });
    const last = curve[curve.length-1];
    svg.appendChild(el('text',{x:W-m.r+8, y:y(last.gc_work)+4, 'font-size':11,
      fill:palette[wi%palette.length]},[txt(w.name)]));
  });
  // the stock collector is the K=infinity member of the family
  svg.appendChild(el('line',{x1:m.l,y1:m.t+(H-m.t-m.b)*(1-1/1.35),
    x2:W-m.r,y2:m.t+(H-m.t-m.b)*(1-1/1.35),stroke:'#3a4150','stroke-dasharray':'3 4'}));
  svg.appendChild(el('text',{x:W-m.r+8,y:m.t+(H-m.t-m.b)*(1-1/1.35)-4,'font-size':10,
    fill:'#98a2b3'},[txt('stock = 100%')]));
  host.appendChild(svg);
}

// ------------------------------------------------------------- heap player
const CELL = 5, COLS = 132, EDEN_ROWS = 26, OLD_ROWS = 18, GAP_ROWS = 2;
const ROWS = EDEN_ROWS + GAP_ROWS + OLD_ROWS;

class HeapView {
  constructor(host, film){
    this.film = film;
    const head = document.createElement('div');
    head.className = 'filmhead';
    head.innerHTML = '<b>' + film.policy + '</b><span id="s"></span>';
    this.stat = head.querySelector('#s');
    const c = document.createElement('canvas');
    c.width = COLS*CELL; c.height = ROWS*CELL;
    host.appendChild(head); host.appendChild(c);
    this.ctx = c.getContext('2d');
    this.reset();
  }
  reset(){
    this.i = 0;
    this.slot = new Map();          // seq -> cell index
    this.cellState = new Uint8Array(COLS*ROWS);  // 0 free 1 eden 2 old 3 pretenured
    this.freeEden = []; this.freeOld = [];
    for(let r = EDEN_ROWS-1; r >= 0; r--)
      for(let c = COLS-1; c >= 0; c--) this.freeEden.push(r*COLS+c);
    for(let r = ROWS-1; r >= EDEN_ROWS+GAP_ROWS; r--)
      for(let c = COLS-1; c >= 0; c--) this.freeOld.push(r*COLS+c);
    this.flash = 0; this.gcs = 0; this.bytes = 0; this.promoted = 0;
    this.paint(true);
  }
  put(seq, pool, state){
    const free = pool === 0 ? this.freeEden : this.freeOld;
    const cell = free.pop();
    if(cell === undefined) return;   // grid full: stop drawing, keep simulating
    this.slot.set(seq, cell);
    this.cellState[cell] = state;
    this.dirty = true;
  }
  drop(seq){
    const cell = this.slot.get(seq);
    if(cell === undefined) return;
    this.slot.delete(seq);
    (cell < EDEN_ROWS*COLS ? this.freeEden : this.freeOld).push(cell);
    this.cellState[cell] = 0;
    this.dirty = true;
  }
  step(n){
    const ev = this.film.events;
    let done = 0;
    while(this.i < ev.length && done < n){
      const e = ev[this.i++]; done++;
      if(e[0] === 'a'){
        this.bytes = e[5];
        if(e[3] === 0) this.put(e[1], 0, 1);
        else this.put(e[1], 1, 3);
      } else if(e[0] === 'f'){
        this.drop(e[1]);
      } else if(e[0] === 'm'){
        this.drop(e[1]); this.put(e[1], 1, 2); this.promoted++;
      } else if(e[0] === 'g'){
        this.gcs++; this.flash = 6; break;   // pause a beat on a collection
      }
    }
    return this.i < ev.length;
  }
  paint(force){
    const ctx = this.ctx;
    ctx.fillStyle = '#0b0d11';
    ctx.fillRect(0,0,COLS*CELL,ROWS*CELL);
    ctx.fillStyle = '#14181f';
    ctx.fillRect(0,0,COLS*CELL,EDEN_ROWS*CELL);
    ctx.fillStyle = '#111520';
    ctx.fillRect(0,(EDEN_ROWS+GAP_ROWS)*CELL,COLS*CELL,OLD_ROWS*CELL);
    const cols = ['#2a2f3a', getCss('--eden'), getCss('--old'), getCss('--pre')];
    for(let i = 0; i < this.cellState.length; i++){
      const s = this.cellState[i];
      if(!s) continue;
      ctx.fillStyle = cols[s];
      ctx.fillRect((i%COLS)*CELL, Math.floor(i/COLS)*CELL, CELL-1, CELL-1);
    }
    ctx.fillStyle = '#0b0d11';
    ctx.fillRect(0, EDEN_ROWS*CELL, COLS*CELL, CELL*GAP_ROWS);
    if(this.flash > 0){
      ctx.fillStyle = 'rgba(240,160,60,' + (this.flash/90) + ')';
      ctx.fillRect(0,0,COLS*CELL,EDEN_ROWS*CELL);
      this.flash--;
    }
    ctx.font = '9px ui-sans-serif,system-ui,sans-serif';
    ctx.fillStyle = '#5b6473';
    ctx.fillText('nursery', 4, 11);
    ctx.fillText('old generation', 4, (EDEN_ROWS+GAP_ROWS)*CELL + 11);
    this.stat.textContent = (this.bytes/1024).toFixed(0) + ' KB allocated  ·  '
      + this.gcs + ' collections  ·  ' + this.promoted + ' promoted';
  }
}
function getCss(v){
  return getComputedStyle(document.documentElement).getPropertyValue(v).trim();
}

let views = [], playing = true;
function initFilms(){
  const host = document.getElementById('films');
  [FILM.baseline, FILM.oracle].forEach(f => {
    const d = document.createElement('div');
    host.appendChild(d);
    views.push(new HeapView(d, f));
  });
  document.getElementById('play').onclick = () => {
    playing = !playing;
    document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  };
  requestAnimationFrame(tick);
}
function tick(){
  if(playing){
    const n = +document.getElementById('speed').value * 12;
    let any = false;
    views.forEach(v => { if(v.step(n)) any = true; });
    views.forEach(v => v.paint());
    document.getElementById('clock').textContent =
      FILM.baseline.workload + (FILM.baseline.truncated ? ' (first slice of the run)' : '');
    if(!any) views.forEach(v => v.reset());
  }
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ tables
function drawModelTable(){
  const t = document.getElementById('model');
  t.innerHTML = '<tr><th>workload</th><th>AUC (held out)</th><th>precision</th>'
    + '<th>recall</th><th>target rate</th><th>learned vs stock</th></tr>';
  RESULTS.workloads.forEach(w => {
    const m = w.model.test_metrics;
    const learned = w.collectors.find(c => c.label==='gen+learned').gc_work;
    const base = w.collectors.find(c => c.label==='generational').gc_work;
    const rel = learned/base;
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>'+w.name+'</td>'
      + '<td>'+(m.auc===null?'-':m.auc.toFixed(3))+'</td>'
      + '<td>'+m.precision.toFixed(3)+'</td>'
      + '<td>'+m.recall.toFixed(3)+'</td>'
      + '<td>'+(m.base_rate*100).toFixed(1)+'%</td>'
      + '<td class="'+(rel<1?'win':'loss')+'">'+fmtPct(rel)+'</td>';
    t.appendChild(tr);
  });
}

function drawFullTable(){
  const t = document.getElementById('full');
  t.innerHTML = '<tr><th>workload</th><th>collector</th><th>GC work</th><th>traced</th>'
    + '<th>copied</th><th>swept</th><th>minor</th><th>major</th><th>pretenured B</th></tr>';
  RESULTS.workloads.forEach(w => {
    w.collectors.forEach(c => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>'+w.name+'</td><td>'+c.label+'</td>'
        + '<td>'+Math.round(c.gc_work).toLocaleString()+'</td>'
        + '<td>'+c.bytes_traced.toLocaleString()+'</td>'
        + '<td>'+c.bytes_copied.toLocaleString()+'</td>'
        + '<td>'+c.bytes_swept.toLocaleString()+'</td>'
        + '<td>'+c.minor_collections+'</td><td>'+c.major_collections+'</td>'
        + '<td>'+(c.pretenured_bytes===undefined?'-':c.pretenured_bytes.toLocaleString())+'</td>';
      t.appendChild(tr);
    });
  });
}

document.getElementById('foot').textContent =
  'nursery ' + (RESULTS.config.nursery_bytes/1024) + ' KB  ·  old gen ' +
  (RESULTS.config.old_bytes/1024) + ' KB  ·  flat heap ' +
  (RESULTS.config.flat_heap_bytes/1024) + ' KB  ·  lifetime tracing granularity ' +
  RESULTS.config.trace_granularity + ' B  ·  experiment ran in ' +
  RESULTS.generated_seconds + ' s  ·  regenerate with `make results && make report`';

drawBars(); drawRecovered(); drawK(); drawModelTable(); drawFullTable(); initFilms();
</script>
</body>
</html>
"""


def build_report(results_path=None, out_path=None, film_workload="particles"):
    results_path = results_path or os.path.join(RESULTS_DIR, "results.json")
    out_path = out_path or os.path.join(RESULTS_DIR, "report.html")
    with open(results_path) as fh:
        results = json.load(fh)

    entry = next(w for w in results["workloads"] if w["name"] == film_workload)
    film = capture_pair(film_workload, entry["policy_family"]["k_oracle"],
                        max_events=14000)

    html = (TEMPLATE
            .replace("__RESULTS__", json.dumps(results, separators=(",", ":")))
            .replace("__FILM__", json.dumps(film, separators=(",", ":"))))
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


def build_summary(results_path=None, out_path=None):
    """A markdown results table, so the README never quotes a number by hand."""
    results_path = results_path or os.path.join(RESULTS_DIR, "results.json")
    out_path = out_path or os.path.join(RESULTS_DIR, "summary.md")

    with open(results_path) as fh:
        r = json.load(fh)

    lines = []
    lines.append("<!-- generated by `python -m halflife.viz.report`; do not edit -->")
    lines.append("")
    lines.append("## GC work relative to a stock generational collector")
    lines.append("")
    lines.append("Lower is better. 100% is the stock generational collector. "
                 "`gen+oracle` reads true lifetimes off a recorded trace and cannot be "
                 "shipped; it is the ceiling, not a proposal.")
    lines.append("")
    labels = ["mark-sweep", "semispace", "generational", "gen+always",
              "gen+sitetable", "gen+learned", "gen+oracle"]
    lines.append("| workload | " + " | ".join(labels) + " | headroom recovered |")
    lines.append("|" + "---|" * (len(labels) + 2))
    for w in r["workloads"]:
        by = {c["label"]: c["gc_work"] for c in w["collectors"]}
        base = by["generational"]
        cells = []
        for lab in labels:
            v = by.get(lab)
            if v is None:
                cells.append("-")
            else:
                rel = 100.0 * v / base
                cells.append(("%.2f%%" if rel < 1 else "%.0f%%") % rel)
        rec = w["recovered"]["gen+learned"]
        cells.append("n/a" if rec is None else "**%.0f%%**" % (100 * rec))
        lines.append("| `%s` | %s |" % (w["name"], " | ".join(cells)))
    lines.append("")

    lines.append("## Workload profile")
    lines.append("")
    lines.append("| workload | objects | allocated | survive one nursery | "
                 "immortal | sites | best K | K chosen from profile |")
    lines.append("|" + "---|" * 8)
    for w in r["workloads"]:
        t = w["eval_trace"]
        pf = w["policy_family"]
        lines.append(
            "| `%s` | %s | %s KB | %.1f%% | %s | %d | %.2f | %.2f |"
            % (w["name"], "{:,}".format(t["objects"]),
               "{:,}".format(t["bytes_allocated"] // 1024),
               100 * t["survivor_rate"], "{:,}".format(t["immortal"]),
               w["n_sites"], pf["k_oracle"], pf["k_star"])
        )
    lines.append("")

    lines.append("## Model quality on the held-out run")
    lines.append("")
    lines.append("`target rate` is the fraction of objects that actually meet the "
                 "learning target (live at least `K*` nurseries). Where it is 100% the "
                 "profiling run chose `K* = 0`, every object is a positive, and AUC is "
                 "undefined -- the model has correctly learned to pretenure everything.")
    lines.append("")
    lines.append("| workload | AUC | precision | recall | target rate | "
                 "distinct contexts scored |")
    lines.append("|" + "---|" * 6)
    for w in r["workloads"]:
        m = w["model"]["test_metrics"]
        learned = next(c for c in w["collectors"] if c["label"] == "gen+learned")
        ctx = learned.get("predictor", {}).get("distinct_contexts", "-")
        lines.append("| `%s` | %s | %.3f | %.3f | %.1f%% | %s |"
                     % (w["name"],
                        "-" if m["auc"] is None else "%.3f" % m["auc"],
                        m["precision"], m["recall"], 100 * m["base_rate"], ctx))
    lines.append("")

    if r.get("transfer"):
        lines.append("## One model across all five programs")
        lines.append("")
        lines.append("Trained on every benchmark at once with the program-specific "
                     "features (site id, function id, caller id) stripped out, leaving "
                     "only allocation kind, size, call depth and per-site allocation "
                     "count.")
        lines.append("")
        lines.append("| workload | GC work vs stock | headroom recovered |")
        lines.append("|---|---|---|")
        for w in r["workloads"]:
            t = r["transfer"]["per_workload"][w["name"]]
            base = next(c for c in w["collectors"] if c["label"] == "generational")["gc_work"]
            rec = t["recovered"]
            rel = 100.0 * t["stats"]["gc_work"] / base
            lines.append(("| `%s` | " + ("%.2f%%" if rel < 1 else "%.0f%%") + " | %s |")
                         % (w["name"], rel,
                            "n/a" if rec is None else "%.0f%%" % (100 * rec)))
        lines.append("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


def main():
    s = build_summary()
    print("wrote", s)
    p = build_report()
    print("wrote", p, "(%.1f MB)" % (os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
