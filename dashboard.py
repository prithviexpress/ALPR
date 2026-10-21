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
  {"dashboard": {"host": "0.0.0.0", "port": 8082, "db_path": "dashboard.db"}}
and top-level "events_csv" (shared with bay_occupancy.py).
"""
import csv
import json
import sqlite3
import statistics
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULTS = {"host": "0.0.0.0", "port": 8082, "db_path": "dashboard.db"}
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
    """events: [(ts, bay, status, prev_status)] -> {bay: [visit,...]}.
    visit = dict(start, end|None, partial). Only valid sequences are kept:
    an occupied row followed by another occupied row (service restarted
    while the bay was empty) leaves the first visit's end unknown, so it
    is dropped rather than guessed."""
    by_bay = {}
    for ts, bay, status, prev in sorted(events):
        by_bay.setdefault(bay, []).append((ts, status, prev))
    out = {}
    for bay, rows in by_bay.items():
        visits, cur = [], None
        for ts, status, prev in rows:
            if status == "occupied":
                cur = {"start": ts, "end": None, "partial": prev is None}
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


def build_stats(events, now, days=None):
    """Everything the page shows, for the last `days` days (None = all)."""
    if not events:
        return {"generated": int(now), "bays": [], "fleet": {},
                "by_hour": [0] * 24, "by_day": [], "window_sec": 0}
    first = min(e[0] for e in events)
    w_start = first if days is None else max(first, now - days * 86400)
    window = max(1.0, now - w_start)
    visits = build_visits(events, now)
    last_status = {}
    for ts, bay, status, _ in sorted(events):
        last_status[bay] = (status, ts)

    bays, all_ttat = [], []
    by_hour, by_day = [0] * 24, {}
    for bay in sorted(visits):
        occ_sec, ttats, n_visits, last_visit = 0.0, [], 0, None
        for v in visits[bay]:
            s, e = v["start"], v["end"] if v["end"] is not None else now
            cs, ce = max(s, w_start), min(e, now)
            if ce > cs:
                occ_sec += ce - cs
            if v["end"] is None or v["partial"] or v["end"] < w_start:
                continue
            n_visits += 1
            ttats.append(v["end"] - v["start"])
            last_visit = v["end"]
            if v["start"] >= w_start:
                by_hour[datetime.fromtimestamp(v["start"]).hour] += 1
                day = datetime.fromtimestamp(v["start"]).strftime("%Y-%m-%d")
                by_day[day] = by_day.get(day, 0) + 1
        all_ttat += ttats
        st = last_status[bay]
        bays.append({
            "bay": bay, "status": st[0], "since": st[1],
            "visits": n_visits,
            "utilisation_pct": round(100 * occ_sec / window, 1),
            "ttat_avg": round(statistics.mean(ttats)) if ttats else None,
            "ttat_median": round(statistics.median(ttats)) if ttats else None,
            "ttat_p90": round(_pct(ttats, 90)) if ttats else None,
            "ttat_min": min(ttats) if ttats else None,
            "ttat_max": max(ttats) if ttats else None,
            "last_visit_end": last_visit,
        })
    n_occ = sum(1 for b in bays if b["status"] == "occupied")
    return {
        "generated": int(now), "window_sec": int(window), "days": days,
        "bays": bays,
        "fleet": {
            "bays": len(bays), "occupied_now": n_occ,
            "visits": sum(b["visits"] for b in bays),
            "ttat_avg": round(statistics.mean(all_ttat)) if all_ttat else None,
            "ttat_median": (round(statistics.median(all_ttat))
                            if all_ttat else None),
            "utilisation_pct": (round(statistics.mean(
                b["utilisation_pct"] for b in bays), 1) if bays else 0),
        },
        "by_hour": by_hour,
        "by_day": [{"day": d, "visits": by_day[d]} for d in sorted(by_day)],
    }


def load_events(db):
    return db.execute(
        "SELECT ts, bay, status, prev_status FROM events").fetchall()


# ------------------------------------------------------------------ web --
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bay Stats</title><style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2230;--mut:#6b7385;--line:#e3e6ec;
--acc:#2f6fed;--ok:#1f9d55;--bad:#d9822b}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;
--fg:#e6e9f0;--mut:#8b93a6;--line:#2a3140;--acc:#6b9bff;--ok:#3ec47a;--bad:#e8a04c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
h1{font-size:20px;margin:0}select{background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:12px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px;margin-bottom:14px;overflow-x:auto}h2{font-size:14px;margin:0 0 8px}
table{border-collapse:collapse;width:100%}th,td{padding:6px 8px;text-align:right;
border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);
font-weight:600;cursor:pointer}th:first-child,td:first-child{text-align:left}
.pill{padding:2px 8px;border-radius:99px;font-size:12px;color:#fff}
.occupied{background:var(--bad)}.empty{background:var(--ok)}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--acc);
vertical-align:middle;margin-right:6px}
.chart{display:flex;align-items:flex-end;gap:3px;height:110px}
.chart div{flex:1;background:var(--acc);border-radius:3px 3px 0 0;min-height:1px;position:relative}
.axis{display:flex;gap:3px;color:var(--mut);font-size:10px}.axis span{flex:1;text-align:center}
small{color:var(--mut)}
</style></head><body><main>
<header><h1>Bay stats</h1><span><select id="days">
<option value="1">Last 24 h</option><option value="7" selected>Last 7 days</option>
<option value="30">Last 30 days</option><option value="">All time</option></select>
<small id="gen"></small></span></header>
<div class="cards" id="cards"></div>
<section><h2>Per bay <small>(click a heading to sort · TTAT = occupied → empty)</small></h2>
<table id="tbl"></table></section>
<section><h2>Arrivals by hour of day</h2><div class="chart" id="hours"></div>
<div class="axis" id="haxis"></div></section>
<section><h2>Arrivals per day</h2><div class="chart" id="days_chart"></div>
<div class="axis" id="daxis"></div></section>
</main><script>
const $=id=>document.getElementById(id);let data=null,sort={k:'bay',d:1};
const dur=s=>s==null?'–':s>=3600?(s/3600).toFixed(1)+' h':s>=60?Math.round(s/60)+' min':s+' s';
const ago=t=>t?dur(Math.max(0,data.generated-t))+' ago':'–';
const COLS=[['bay','Bay'],['status','Now'],['visits','Visits'],['utilisation_pct','Utilisation'],
['ttat_avg','Avg TTAT'],['ttat_median','Median'],['ttat_p90','P90'],['ttat_min','Min'],
['ttat_max','Max'],['last_visit_end','Last departure']];
function render(){const f=data.fleet||{};
$('cards').innerHTML=[['Occupied now',`${f.occupied_now??0} / ${f.bays??0}`],
['Visits',f.visits??0],['Avg TTAT',dur(f.ttat_avg)],['Median TTAT',dur(f.ttat_median)],
['Avg utilisation',(f.utilisation_pct??0)+'%']].map(([l,v])=>`<div class=card><b>${v}</b><span>${l}</span></div>`).join('');
const rows=[...data.bays].sort((a,b)=>{const x=a[sort.k],y=b[sort.k];
return ((x==null)-(y==null))||(x>y?1:x<y?-1:0)*sort.d});
$('tbl').innerHTML='<tr>'+COLS.map(([k,l])=>`<th data-k=${k}>${l}${sort.k==k?(sort.d>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>'+
rows.map(b=>`<tr><td>${b.bay}</td><td><span class="pill ${b.status}">${b.status}</span> <small>${ago(b.since)}</small></td>
<td>${b.visits}</td><td><span class=bar style="width:${Math.min(100,b.utilisation_pct)*1.2}px"></span>${b.utilisation_pct}%</td>
<td>${dur(b.ttat_avg)}</td><td>${dur(b.ttat_median)}</td><td>${dur(b.ttat_p90)}</td><td>${dur(b.ttat_min)}</td>
<td>${dur(b.ttat_max)}</td><td>${ago(b.last_visit_end)}</td></tr>`).join('')||'<tr><td>No data yet</td></tr>';
document.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
sort=sort.k==k?{k,d:-sort.d}:{k,d:1};render()});
const bars=(el,ax,vals,labels)=>{const m=Math.max(1,...vals);
$(el).innerHTML=vals.map((v,i)=>`<div style="height:${100*v/m}%" title="${labels[i]}: ${v}"></div>`).join('');
$(ax).innerHTML=labels.map((l,i)=>`<span>${vals.length>14&&i%2?'':l}</span>`).join('')};
bars('hours','haxis',data.by_hour,[...Array(24).keys()].map(h=>String(h).padStart(2,'0')));
bars('days_chart','daxis',data.by_day.map(d=>d.visits),data.by_day.map(d=>d.day.slice(5)));
$('gen').textContent=' updated '+new Date(data.generated*1000).toLocaleTimeString()}
async function load(){const d=$('days').value;
data=await (await fetch('/api/stats'+(d?'?days='+d:''))).json();render()}
$('days').onchange=load;load();setInterval(load,60000);
</script></body></html>"""


def make_handler(db, csv_path):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/healthz":
                return self._send(200, b'{"ok":true}', "application/json")
            if u.path == "/api/stats":
                q = parse_qs(u.query).get("days", [""])[0]
                try:
                    days = float(q) if q else None
                except ValueError:
                    return self._send(400, b'{"error":"bad days"}',
                                      "application/json")
                with _lock:
                    import_csv(db, csv_path)
                    stats = build_stats(load_events(db), time.time(), days)
                return self._send(200, json.dumps(stats).encode(),
                                  "application/json")
            self._send(404, b'{"error":"not found"}', "application/json")

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
    ThreadingHTTPServer((d["host"], int(d["port"])),
                        make_handler(db, csv_path)).serve_forever()


if __name__ == "__main__":
    main()
