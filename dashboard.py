"""Bay stats dashboard: turnaround time (TTAT), utilisation, visits.

Separate from bay_occupancy.py -- it only READS bay_events.csv (the file
that script appends to) and serves a web page. Run both side by side.

  python dashboard.py [config.json]      # then open http://<host>:8082/

How it works
  * bay_events.csv is imported incrementally into a SQLite file
    (dashboard.db) -- only new rows are read each refresh; if the CSV is
    replaced/rotated the import restarts cleanly. The CSV stays the source
    of truth; delete dashboard.db any time and it rebuilds.
  * Each bay's rows are paired into visits: an "occupied" row opens one,
    the next "empty" row closes it. Visit length = TTAT.
  * A visit that opened on the first row after a service restart (blank
    previous_status) did not necessarily START then -- the truck may have
    been there for hours. Those visits are "partial": counted in
    utilisation but left out of TTAT and visit counts. A visit still open
    right now counts toward utilisation but not TTAT.
  * Utilisation = occupied seconds / observed seconds in the window. The
    window never starts before the first event, so a new install isn't
    penalised for time before it was watching.

Config (optional keys in config.json under "dashboard"):
  {"dashboard": {"host": "0.0.0.0", "port": 8082, "db_path": "dashboard.db",
                 "group_prefix_len": 2}}
group_prefix_len: bays are grouped by the first N characters of their
name (AR-K3 -> "AR"); 0 puts everything in one group.
and top-level "events_csv" (shared with bay_occupancy.py).
"""
import csv
import json
import sqlite3
import statistics
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

DEFAULTS = {"host": "0.0.0.0", "port": 8082, "db_path": "dashboard.db",
            "group_prefix_len": 2}
SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  ts INTEGER NOT NULL, bay TEXT NOT NULL, status TEXT NOT NULL,
  prev_status TEXT, class TEXT, confidence REAL, raw_ts TEXT,
  UNIQUE(raw_ts, bay, status));
CREATE INDEX IF NOT EXISTS ix_events_bay_ts ON events(bay, ts);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""
_lock = threading.Lock()


# --------------------------------------------------------------- import --
def connect(path):
    db = sqlite3.connect(path, check_same_thread=False)
    db.executescript(SCHEMA)
    return db


def import_csv(db, csv_path):
    """Append rows added to the CSV since last time. Tracks a byte offset
    plus the header line, and starts over if the file shrank/changed."""
    p = Path(csv_path)
    if not p.exists():
        return 0
    size = p.stat().st_size
    row = db.execute("SELECT v FROM meta WHERE k='offset'").fetchone()
    offset = int(row[0]) if row else 0
    if size < offset:                    # file replaced or truncated
        offset = 0
    n = 0
    with open(p, "rb") as f:
        header = f.readline()
        start = max(offset, len(header))
        f.seek(start)
        chunk = f.read()
    # Only consume complete lines; a half-written last row waits for the
    # next refresh.
    cut = chunk.rfind(b"\n") + 1
    end = start + cut
    cols = next(csv.reader([header.decode("utf-8-sig")]), [])
    for r in csv.reader(chunk[:cut].decode("utf-8").splitlines()):
        if len(r) < len(cols):
            continue
        d = dict(zip(cols, r))
        try:
            ts = int(datetime.strptime(
                d["timestamp"], "%Y-%m-%dT%H:%M:%S%z").timestamp())
        except (KeyError, ValueError):
            continue
        conf = d.get("confidence") or None
        cur = db.execute(
            "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)",
            (ts, d["bay"], d["status"], d.get("previous_status") or None,
             d.get("class") or None,
             float(conf) if conf else None, d["timestamp"]))
        n += cur.rowcount
    db.execute("INSERT OR REPLACE INTO meta VALUES('offset', ?)", (str(end),))
    db.commit()
    return n


# ---------------------------------------------------------------- stats --
def build_visits(events, now):
    """events: [(ts, bay, status, prev_status[, class])] -> {bay: [visit]}.
    visit = dict(start, end|None, partial, cls). Only valid sequences are
    kept: an occupied row followed by another occupied row (service
    restarted while the bay was empty) leaves the first visit's end
    unknown, so it is dropped rather than guessed."""
    by_bay = {}
    for e in sorted(events, key=lambda e: e[0]):
        by_bay.setdefault(e[1], []).append(e)
    out = {}
    for bay, rows in by_bay.items():
        visits, cur = [], None
        for e in rows:
            ts, status, prev = e[0], e[2], e[3]
            if status == "occupied":
                cur = {"start": ts, "end": None, "partial": prev is None,
                       "cls": e[4] if len(e) > 4 else None}
            elif status == "empty" and cur:
                cur["end"] = ts
                visits.append(cur)
                cur = None
        if cur:
            visits.append(cur)           # still open
        out[bay] = visits
    return out


def _pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]


def group_of(bay, prefix_len=2):
    """Bay group = first `prefix_len` characters of the bay name
    (AR-K3 -> AR)."""
    return bay[:prefix_len].upper() if prefix_len else "ALL"


def _hour_slices(start, end):
    """Split [start, end) at local clock-hour boundaries ->
    [(hour_of_day, seconds)]."""
    out, t = [], start
    while t < end:
        dt = datetime.fromtimestamp(t)
        nxt = (dt.replace(minute=0, second=0, microsecond=0)
               + timedelta(hours=1)).timestamp()
        seg_end = min(end, nxt)
        out.append((dt.hour, seg_end - t))
        t = seg_end
    return out


def _stats_for(visits, w_start, now):
    """Occupied seconds in the window plus the TTAT list / last departure."""
    occ_sec, ttats, last_end = 0.0, [], None
    for v in visits:
        e = v["end"] if v["end"] is not None else now
        cs, ce = max(v["start"], w_start), min(e, now)
        if ce > cs:
            occ_sec += ce - cs
        if v["end"] is None or v["partial"] or v["end"] < w_start:
            continue
        ttats.append(v["end"] - v["start"])
        last_end = v["end"]
    return occ_sec, ttats, last_end


def _ttat_summary(ttats):
    return {
        "ttat_avg": round(statistics.mean(ttats)) if ttats else None,
        "ttat_median": round(statistics.median(ttats)) if ttats else None,
        "ttat_p90": round(_pct(ttats, 90)) if ttats else None,
        "ttat_min": min(ttats) if ttats else None,
        "ttat_max": max(ttats) if ttats else None,
    }


def _window(events, now, days):
    first = min(e[0] for e in events)
    w_start = first if days is None else max(first, now - days * 86400)
    return w_start, max(1.0, now - w_start)


def _counted_visits(visits, w_start):
    """Completed, non-partial visits that ended inside the window."""
    return [v for v in visits if v["end"] is not None and not v["partial"]
            and v["end"] >= w_start]


def build_stats(events, now, days=None, group=None, prefix_len=2):
    """Overview for the last `days` days (None = all); `group` restricts
    bays/charts/cards to one bay group but `groups` always lists all."""
    if not events:
        return {"generated": int(now), "bays": [], "groups": [],
                "fleet": {}, "by_hour": [0] * 24, "by_day": [],
                "window_sec": 0, "group": group}
    w_start, window = _window(events, now, days)
    visits = build_visits(events, now)
    last_status = {}
    for e in sorted(events, key=lambda e: e[0]):
        last_status[e[1]] = (e[2], e[0])

    bays, grp = [], {}
    for bay in sorted(visits):
        occ_sec, ttats, last_end = _stats_for(visits[bay], w_start, now)
        g = group_of(bay, prefix_len)
        st = last_status[bay]
        row = {"bay": bay, "group": g, "status": st[0], "since": st[1],
               "visits": len(ttats),
               "utilisation_pct": round(100 * occ_sec / window, 1),
               "last_visit_end": last_end, **_ttat_summary(ttats)}
        bays.append(row)
        a = grp.setdefault(g, {"bays": 0, "occupied_now": 0,
                               "ttats": [], "utils": []})
        a["bays"] += 1
        a["occupied_now"] += st[0] == "occupied"
        a["ttats"] += ttats
        a["utils"].append(row["utilisation_pct"])
    groups = [{"group": g, "bays": a["bays"],
               "occupied_now": a["occupied_now"], "visits": len(a["ttats"]),
               "utilisation_pct": round(statistics.mean(a["utils"]), 1),
               **_ttat_summary(a["ttats"])}
              for g, a in sorted(grp.items())]

    shown = [b for b in bays if group in (None, "", b["group"])]
    by_hour, by_day, all_ttat = [0] * 24, {}, []
    for b in shown:
        for v in _counted_visits(visits[b["bay"]], w_start):
            all_ttat.append(v["end"] - v["start"])
            if v["start"] >= w_start:
                dt = datetime.fromtimestamp(v["start"])
                by_hour[dt.hour] += 1
                d = dt.strftime("%Y-%m-%d")
                by_day[d] = by_day.get(d, 0) + 1
    return {
        "generated": int(now), "window_sec": int(window), "days": days,
        "group": group or None, "bays": shown, "groups": groups,
        "fleet": {
            "bays": len(shown),
            "occupied_now": sum(b["status"] == "occupied" for b in shown),
            "visits": len(all_ttat),
            "ttat_avg": round(statistics.mean(all_ttat)) if all_ttat else None,
            "ttat_median": (round(statistics.median(all_ttat))
                            if all_ttat else None),
            "utilisation_pct": (round(statistics.mean(
                b["utilisation_pct"] for b in shown), 1) if shown else 0),
        },
        "by_hour": by_hour,
        "by_day": [{"day": d, "visits": by_day[d]} for d in sorted(by_day)],
    }


def build_bay_detail(events, bay, now, days=None, prefix_len=2, limit=300):
    """Drill-down for one bay: summary, arrivals and occupancy by hour of
    day, and one entry per truck visit (for the per-truck chart/table).
    Returns None for a bay with no events."""
    if not any(e[1] == bay for e in events):
        return None
    w_start, window = _window(events, now, days)
    visits = build_visits(events, now).get(bay, [])
    occ_sec, ttats, last_end = _stats_for(visits, w_start, now)
    st = max((e for e in events if e[1] == bay), key=lambda e: e[0])

    arrivals = [0] * 24
    occ_by_hour = [0.0] * 24
    for v in visits:
        e = v["end"] if v["end"] is not None else now
        cs, ce = max(v["start"], w_start), min(e, now)
        for h, sec in (_hour_slices(cs, ce) if ce > cs else []):
            occ_by_hour[h] += sec
        if v["start"] >= w_start and not v["partial"]:
            arrivals[datetime.fromtimestamp(v["start"]).hour] += 1
    seen = [0.0] * 24
    for h, sec in _hour_slices(w_start, now):
        seen[h] += sec
    occ_pct = [round(100 * o / s, 1) if s else 0 for o, s in
               zip(occ_by_hour, seen)]

    listed = [v for v in visits
              if v["end"] is None or v["end"] >= w_start][-limit:]
    return {
        "generated": int(now), "window_sec": int(window), "days": days,
        "bay": bay, "group": group_of(bay, prefix_len),
        "status": st[2], "since": st[0],
        "visits": len(ttats),
        "utilisation_pct": round(100 * occ_sec / window, 1),
        "last_visit_end": last_end, **_ttat_summary(ttats),
        "arrivals_by_hour": arrivals, "occupancy_pct_by_hour": occ_pct,
        "trucks": [{"start": v["start"], "end": v["end"],
                    "ttat": (v["end"] - v["start"]
                             if v["end"] is not None else None),
                    "partial": v["partial"], "class": v["cls"]}
                   for v in listed],
    }


def load_events(db):
    return db.execute(
        "SELECT ts, bay, status, prev_status, class FROM events").fetchall()


# ------------------------------------------------------------------ web --
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bay Stats</title><style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2230;--mut:#6b7385;--line:#e3e6ec;
--acc:#2f6fed;--acc2:#1f9d9d;--ok:#1f9d55;--bad:#d9822b}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;
--fg:#e6e9f0;--mut:#8b93a6;--line:#2a3140;--acc:#6b9bff;--acc2:#4fc3c3;--ok:#3ec47a;--bad:#e8a04c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
h1{font-size:20px;margin:0}h1 a{color:inherit;text-decoration:none}
select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 0}
.chip{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:99px;
padding:4px 12px;cursor:pointer;font:inherit}.chip.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:12px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px;margin-bottom:14px;overflow-x:auto}h2{font-size:14px;margin:0 0 8px}
table{border-collapse:collapse;width:100%}th,td{padding:6px 8px;text-align:right;
border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);
font-weight:600;cursor:pointer}th:first-child,td:first-child{text-align:left}
tr.click{cursor:pointer}tr.click:hover td{background:var(--bg)}
.pill{padding:2px 8px;border-radius:99px;font-size:12px;color:#fff}
.occupied{background:var(--bad)}.empty{background:var(--ok)}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--acc);
vertical-align:middle;margin-right:6px}
.chart{display:flex;align-items:flex-end;gap:3px;height:110px}
.chart div{flex:1;background:var(--acc);border-radius:3px 3px 0 0;min-height:1px}
.chart.b div{background:var(--acc2)}.chart div.part{opacity:.4}
.axis{display:flex;gap:3px;color:var(--mut);font-size:10px}.axis span{flex:1;text-align:center;overflow:hidden}
small{color:var(--mut)}a.back{color:var(--acc);cursor:pointer;text-decoration:none}
</style></head><body><main>
<header><h1><a href="#/">Bay stats</a></h1><span><select id="days">
<option value="1">Last 24 h</option><option value="7" selected>Last 7 days</option>
<option value="30">Last 30 days</option><option value="">All time</option></select>
<small id="gen"></small></span></header>
<div id="view"></div>
</main><script>
const $=id=>document.getElementById(id);
let data=null,sort={k:'bay',d:1},group='',route=location.hash||'#/';
const dur=s=>s==null?'–':s>=3600?(s/3600).toFixed(1)+' h':s>=60?Math.round(s/60)+' min':s+' s';
const ago=t=>t?dur(Math.max(0,data.generated-t))+' ago':'–';
const fmt=t=>t?new Date(t*1000).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'–';
const card=(l,v)=>`<div class=card><b>${v}</b><span>${l}</span></div>`;
const bars=(vals,labels,cls='',tip=(i,v)=>v,parts=[])=>{const m=Math.max(1,...vals);
return `<div class="chart ${cls}">`+vals.map((v,i)=>`<div class="${parts[i]?'part':''}" style="height:${100*v/m}%" title="${labels[i]}: ${tip(i,v)}"></div>`).join('')+
`</div><div class=axis>`+labels.map((l,i)=>`<span>${vals.length>14&&i%Math.ceil(vals.length/14)?'':l}</span>`).join('')+`</div>`};
const HRS=[...Array(24).keys()].map(h=>String(h).padStart(2,'0'));
const COLS=[['bay','Bay'],['status','Now'],['visits','Visits'],['utilisation_pct','Utilisation'],
['ttat_avg','Avg TTAT'],['ttat_median','Median'],['ttat_p90','P90'],['ttat_min','Min'],
['ttat_max','Max'],['last_visit_end','Last departure']];
const util=u=>`<span class=bar style="width:${Math.min(100,u)*1.2}px"></span>${u}%`;
const pill=(s,t)=>`<span class="pill ${s}">${s}</span> <small>${t}</small>`;

function renderOverview(){const f=data.fleet||{};
const gs=data.groups||[];
let h=`<div class=chips><button class="chip ${group?'':'on'}" data-g="">All groups</button>`+
gs.map(g=>`<button class="chip ${group==g.group?'on':''}" data-g="${g.group}">${g.group} <small>(${g.bays})</small></button>`).join('')+`</div>`;
h+=`<div class=cards>`+card('Occupied now',`${f.occupied_now??0} / ${f.bays??0}`)+card('Visits',f.visits??0)+
card('Avg TTAT',dur(f.ttat_avg))+card('Median TTAT',dur(f.ttat_median))+card('Avg utilisation',(f.utilisation_pct??0)+'%')+`</div>`;
if(gs.length>1)h+=`<section><h2>By group <small>(first characters of the bay name · click to filter)</small></h2><table><tr>
<th>Group</th><th>Bays</th><th>Occupied now</th><th>Visits</th><th>Utilisation</th><th>Avg TTAT</th><th>Median</th><th>P90</th></tr>`+
gs.map(g=>`<tr class=click data-g="${g.group}"><td>${g.group}</td><td>${g.bays}</td><td>${g.occupied_now}</td><td>${g.visits}</td>
<td>${util(g.utilisation_pct)}</td><td>${dur(g.ttat_avg)}</td><td>${dur(g.ttat_median)}</td><td>${dur(g.ttat_p90)}</td></tr>`).join('')+`</table></section>`;
const rows=[...data.bays].sort((a,b)=>{const x=a[sort.k],y=b[sort.k];
return ((x==null)-(y==null))||(x>y?1:x<y?-1:0)*sort.d});
h+=`<section><h2>Per bay <small>(click a heading to sort, a bay for detail · TTAT = occupied → empty)</small></h2><table><tr>`+
COLS.map(([k,l])=>`<th data-k=${k}>${l}${sort.k==k?(sort.d>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>'+
(rows.map(b=>`<tr class=click data-bay="${b.bay}"><td>${b.bay}</td><td>${pill(b.status,ago(b.since))}</td>
<td>${b.visits}</td><td>${util(b.utilisation_pct)}</td><td>${dur(b.ttat_avg)}</td><td>${dur(b.ttat_median)}</td><td>${dur(b.ttat_p90)}</td>
<td>${dur(b.ttat_min)}</td><td>${dur(b.ttat_max)}</td><td>${ago(b.last_visit_end)}</td></tr>`).join('')||'<tr><td>No data yet</td></tr>')+`</table></section>`;
h+=`<section><h2>Arrivals by hour of day</h2>${bars(data.by_hour,HRS)}</section>`;
h+=`<section><h2>Arrivals per day</h2>${bars(data.by_day.map(d=>d.visits),data.by_day.map(d=>d.day.slice(5)))}</section>`;
$('view').innerHTML=h;
document.querySelectorAll('[data-g]').forEach(e=>e.onclick=()=>{group=e.dataset.g;load()});
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
sort=sort.k==k?{k,d:-sort.d}:{k,d:1};renderOverview()});
document.querySelectorAll('[data-bay]').forEach(e=>e.onclick=()=>{location.hash='#/bay/'+encodeURIComponent(e.dataset.bay)});}

function renderBay(){const d=data;
let h=`<p><a class=back href="#/">← All bays</a></p><h2 style="font-size:18px">${d.bay} <small>group ${d.group}</small> ${pill(d.status,ago(d.since))}</h2>`;
h+=`<div class=cards>`+card('Visits',d.visits)+card('Utilisation',d.utilisation_pct+'%')+card('Avg TTAT',dur(d.ttat_avg))+
card('Median TTAT',dur(d.ttat_median))+card('P90 TTAT',dur(d.ttat_p90))+card('Longest',dur(d.ttat_max))+card('Last departure',ago(d.last_visit_end))+`</div>`;
h+=`<section><h2>Arrivals by hour of day</h2>${bars(d.arrivals_by_hour,HRS)}</section>`;
h+=`<section><h2>Occupancy by hour of day <small>(% of each hour the bay was occupied)</small></h2>${bars(d.occupancy_pct_by_hour,HRS,'b',(i,v)=>v+'%')}</section>`;
const t=d.trucks,done=t.filter(x=>x.ttat!=null);
h+=`<section><h2>Each truck entry <small>(bar = time in bay, oldest → newest; faded = already there when monitoring started)</small></h2>`+
(done.length?bars(done.map(x=>x.ttat),done.map(x=>new Date(x.start*1000).toLocaleDateString([],{month:'numeric',day:'numeric'})),'',(i,v)=>dur(v)+' from '+fmt(done[i].start),done.map(x=>x.partial)):'<small>No completed visits in this range.</small>')+`</section>`;
h+=`<section><h2>Entries <small>(newest first)</small></h2><table><tr><th>Arrived</th><th>Left</th><th>TTAT</th><th>Detected as</th></tr>`+
[...t].reverse().map(x=>`<tr><td>${fmt(x.start)}${x.partial?' <small>(already there at startup)</small>':''}</td>
<td>${x.end?fmt(x.end):'<small>in bay now</small>'}</td><td>${dur(x.ttat)}</td><td>${x.class||'–'}</td></tr>`).join('')+`</table></section>`;
$('view').innerHTML=h}

async function load(){route=location.hash||'#/';const d=$('days').value,q=[];
if(d)q.push('days='+d);
const m=route.match(/^#\/bay\/(.+)$/);
let url;if(m){url='/api/bay/'+m[1]+(q.length?'?'+q.join('&'):'')}
else{if(group)q.push('group='+encodeURIComponent(group));url='/api/stats'+(q.length?'?'+q.join('&'):'')}
const r=await fetch(url);
if(!r.ok){$('view').innerHTML='<p><a class=back href="#/">← All bays</a></p><p>Not found.</p>';return}
data=await r.json();m?renderBay():renderOverview();
$('gen').textContent=' updated '+new Date(data.generated*1000).toLocaleTimeString()}
$('days').onchange=load;window.onhashchange=load;load();setInterval(load,60000);
</script></body></html>"""


def make_handler(db, csv_path, prefix_len=2):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/healthz":
                return self._json(200, {"ok": True})
            try:
                dq = q.get("days", [""])[0]
                days = float(dq) if dq else None
            except ValueError:
                return self._json(400, {"error": "bad days"})
            if u.path == "/api/stats":
                with _lock:
                    import_csv(db, csv_path)
                    stats = build_stats(load_events(db), time.time(), days,
                                        q.get("group", [None])[0], prefix_len)
                return self._json(200, stats)
            if u.path.startswith("/api/bay/"):
                bay = unquote(u.path[len("/api/bay/"):])
                with _lock:
                    import_csv(db, csv_path)
                    det = build_bay_detail(load_events(db), bay, time.time(),
                                           days, prefix_len)
                if det is None:
                    return self._json(404, {"error": "unknown bay"})
                return self._json(200, det)
            self._json(404, {"error": "not found"})

        def log_message(self, *a):
            pass
    return H


def main():
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    raw = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    d = {**DEFAULTS, **(raw.get("dashboard") or {})}
    csv_path = raw.get("events_csv") or "bay_events.csv"
    db = connect(d["db_path"])
    n = import_csv(db, csv_path)
    print(f"imported {n} events from {csv_path}; "
          f"serving on http://{d['host']}:{d['port']}/")
    ThreadingHTTPServer((d["host"], int(d["port"])), make_handler(
        db, csv_path, int(d["group_prefix_len"]))).serve_forever()


if __name__ == "__main__":
    main()
