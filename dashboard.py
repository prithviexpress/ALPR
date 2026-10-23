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
overstay_hours: a bay occupied longer than this is flagged (0 = off).
group_prefix_len: bays are grouped by the first N characters of their
name (AR-K3 -> "AR"); 0 puts everything in one group.
and top-level "events_csv" (shared with bay_occupancy.py).
"""
import base64
import csv
import io
import json
import logging
import re
import sqlite3
import statistics
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request
from urllib.parse import parse_qs, quote, unquote, urlparse

DEFAULTS = {"host": "0.0.0.0", "port": 8082, "db_path": "dashboard.db",
            "group_prefix_len": 2, "overstay_hours": 4,
            "bay_webhook_url": "http://127.0.0.1:8081"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  ts INTEGER NOT NULL, bay TEXT NOT NULL, status TEXT NOT NULL,
  prev_status TEXT, class TEXT, confidence REAL, raw_ts TEXT, image TEXT,
  source TEXT, UNIQUE(raw_ts, bay, status));
CREATE INDEX IF NOT EXISTS ix_events_bay_ts ON events(bay, ts);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""
_lock = threading.Lock()
IMG_RE = re.compile(r"^[A-Za-z0-9._-]+\.jpg$")


# --------------------------------------------------------------- import --
def connect(path):
    db = sqlite3.connect(path, check_same_thread=False)
    db.executescript(SCHEMA)
    have = {r[1] for r in db.execute("PRAGMA table_info(events)")}
    for col in ("image", "source"):          # columns added after v1 of the db
        if col not in have:
            db.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
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
    hdr = db.execute("SELECT v FROM meta WHERE k='header'").fetchone()
    with open(p, "rb") as f:
        header_now = f.readline().decode("utf-8-sig").strip()
    if size < offset or (hdr and hdr[0] != header_now):
        offset = 0           # file replaced/truncated or columns changed
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
            "INSERT INTO events(ts,bay,status,prev_status,class,confidence,"
            "raw_ts,image,source) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(raw_ts,bay,status) DO UPDATE SET "
            "image=COALESCE(excluded.image, image), "
            "source=COALESCE(excluded.source, source)",
            (ts, d["bay"], d["status"], d.get("previous_status") or None,
             d.get("class") or None,
             float(conf) if conf else None, d["timestamp"],
             d.get("image") or None, d.get("source") or None))
        n += cur.rowcount
    db.execute("INSERT OR REPLACE INTO meta VALUES('offset', ?)", (str(end),))
    db.execute("INSERT OR REPLACE INTO meta VALUES('header', ?)",
               (header_now,))
    db.commit()
    return n


# ------------------------------------------------------- live snapshot --
log = logging.getLogger("dashboard")


def api_headers(api_key):
    """external.api_key may be the key value (sent as X-API-Key) or a
    key/value pair: {"key": "X-API-Key", "value": "..."}."""
    if isinstance(api_key, dict):
        return {str(api_key.get("key") or "X-API-Key"):
                str(api_key.get("value") or "")}
    return {"X-API-Key": str(api_key or "")}


def api_key_value(api_key):
    return api_key.get("value") if isinstance(api_key, dict) else api_key


def norm_bay(b):
    return str(b or "").strip().upper()


class PlateSource:
    """Truck numbers (plate_text) from the site's detections API,
    GET <base>/api/detections?date=YYYY-MM-DD, cached per day. Only the
    plate text is used; images are not fetched."""

    def __init__(self, base_url, api_key, window_sec=900, cache_sec=60,
                 timeout=10):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.window = int(window_sec)
        self.cache_sec = cache_sec
        self.timeout = timeout
        self._cache = {}
        self._lock = threading.Lock()

    def _get(self, path):
        req = urllib.request.Request(self.base + path,
                                     headers=api_headers(self.key))
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def day(self, date_str):
        now = time.time()
        with self._lock:
            hit = self._cache.get(date_str)
            if hit and now - hit[0] < self.cache_sec:
                return hit[1]
        reads = []
        try:
            payload = json.loads(self._get(f"/api/detections?date={date_str}"))
            items = payload.get("items") if isinstance(payload, dict) else []
            for it in items or []:
                try:
                    ts = datetime.strptime(
                        f"{it.get('date') or date_str} {it['time']}",
                        "%Y-%m-%d %H:%M:%S").timestamp()
                except (KeyError, TypeError, ValueError):
                    continue
                if it.get("plate_text"):
                    reads.append({"ts": int(ts),
                                  "bay": norm_bay(it.get("bay_number")),
                                  "plate": str(it["plate_text"])})
        except (OSError, ValueError) as e:
            log.warning(f"detections for {date_str} unavailable: {e}")
        with self._lock:
            self._cache[date_str] = (now, reads)
        return reads

    def for_visit(self, bay, start, end):
        """Truck number(s) read on this bay while the truck was in it. If
        nothing was read inside the visit, the nearest read within the
        window of its arrival is used and marked as such. None if neither."""
        end = end if end is not None else time.time()
        b = norm_bay(bay)
        inside, near = [], None
        t = start - self.window
        days = set()
        while t <= end + self.window:
            days.add(datetime.fromtimestamp(t).strftime("%Y-%m-%d"))
            t += 86400
        days.add(datetime.fromtimestamp(end + self.window).strftime("%Y-%m-%d"))
        for day in sorted(days):
            for r in self.day(day):
                if r["bay"] != b:
                    continue
                if start <= r["ts"] <= end:
                    if r["plate"] not in inside:
                        inside.append(r["plate"])
                else:
                    dt = abs(r["ts"] - start)
                    if dt <= self.window and (near is None or dt < near[0]):
                        near = (dt, r["plate"])
        if inside:
            return ", ".join(inside)
        if near:
            return f"{near[1]} (read near arrival)"
        return None



def fetch_live_snapshot(webhook_url, bay, timeout=5):
    """The live frame bay_occupancy.py serves at GET /bay/<bay> (its
    snapshot_base64 JPEG). Raises OSError when the bay or service is
    unavailable."""
    url = webhook_url.rstrip("/") + "/bay/" + quote(bay, safe="")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        j = json.loads(r.read())
    if not j.get("snapshot_base64"):
        raise OSError("no snapshot yet")
    return base64.b64decode(j["snapshot_base64"])


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
                       "cls": e[4] if len(e) > 4 else None,
                       "conf": e[5] if len(e) > 5 else None,
                       "img_in": e[6] if len(e) > 6 else None,
                       "src": e[7] if len(e) > 7 else None,
                       "img_out": None}
            elif status == "empty" and cur:
                cur["end"] = ts
                cur["img_out"] = e[6] if len(e) > 6 else None
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


# Windows up to this long get an hourly timeline, longer ones a daily one.
HOURLY_MAX_SEC = 72 * 3600


def _floor(t, unit):
    dt = datetime.fromtimestamp(t).replace(minute=0, second=0, microsecond=0)
    return (dt.replace(hour=0) if unit == "day" else dt).timestamp()


def _next(t, unit):
    step = timedelta(days=1) if unit == "day" else timedelta(hours=1)
    return (datetime.fromtimestamp(t) + step).timestamp()


def _slices(start, end, unit):
    """Split [start, end) at LOCAL clock boundaries (hour or midnight) ->
    [(bucket_start_epoch, seconds)]."""
    out, t = [], start
    while t < end:
        b = _floor(t, unit)
        seg_end = min(end, _next(b, unit))
        out.append((b, seg_end - t))
        t = seg_end
    return out


def _resolve_window(events, now, days, start, end):
    """-> (d_start, d_end, w_start, w_end, window_sec).
    d_* is the DISPLAYED range (a whole local day for 'today', so the
    timeline always runs 00:00-24:00); w_* is where data can exist: it
    never starts before the first event nor ends after now."""
    first = min(e[0] for e in events)
    if start is not None:
        d_start, d_end = start, (end if end is not None else now)
    elif days is None:
        d_start, d_end = first, now
    else:
        d_start, d_end = now - days * 86400, now
    w_start, w_end = max(first, d_start), min(d_end, now)
    return d_start, d_end, w_start, w_end, max(1.0, w_end - w_start)


def _stats_for(visits, w_start, w_end):
    """Occupied seconds in the window plus TTAT list / last departure.
    Only completed, non-partial visits that ENDED in the window count
    toward TTAT."""
    occ_sec, ttats, last_end = 0.0, [], None
    for v in visits:
        e = v["end"] if v["end"] is not None else w_end
        cs, ce = max(v["start"], w_start), min(e, w_end)
        if ce > cs:
            occ_sec += ce - cs
        if (v["end"] is None or v["partial"]
                or not w_start <= v["end"] <= w_end):
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


def _timeline(bay_visits, d_start, d_end, w_start, w_end, unit, now):
    """Actual-clock buckets across the displayed range: arrivals and
    occupancy %. bay_visits = list of per-bay visit lists. Buckets in the
    future (or before the first event) carry occ_pct None."""
    n = max(1, len(bay_visits))
    arr, occ = {}, {}
    for visits in bay_visits:
        for v in visits:
            if not v["partial"] and w_start <= v["start"] <= w_end:
                b = _floor(v["start"], unit)
                arr[b] = arr.get(b, 0) + 1
            e = v["end"] if v["end"] is not None else w_end
            cs, ce = max(v["start"], w_start), min(e, w_end)
            if ce > cs:
                for b, sec in _slices(cs, ce, unit):
                    occ[b] = occ.get(b, 0.0) + sec
    seen = dict(_slices(w_start, w_end, unit)) if w_end > w_start else {}
    out, b = [], _floor(d_start, unit)
    while b < d_end:
        s = seen.get(b, 0.0)
        out.append({"t": int(b), "arrivals": arr.get(b, 0),
                    "occ_pct": (round(100 * occ.get(b, 0.0) / (s * n), 1)
                                if s else None)})
        b = _next(b, unit)
    return out


def _segments(visits, d_start, d_end, plates=None, bay=None, now=None):
    out = []
    for v in visits:
        if v["start"] < d_end and (v["end"] is None or v["end"] > d_start):
            pl = plates.for_visit(bay, v["start"], v["end"] or now) if plates else None
            out.append([v["start"], v["end"], v["partial"], pl])
    return out


TTAT_EDGES = [1800, 3600, 7200, 14400]
TTAT_LABELS = ["<30 min", "30-60 min", "1-2 h", "2-4 h", "4 h+"]


def _ttat_hist(ttats):
    out = [0] * len(TTAT_LABELS)
    for t in ttats:
        out[sum(t >= e for e in TTAT_EDGES)] += 1
    return out


def _concurrency(bay_visits, w_start, w_end):
    """Peak number of bays occupied at once in the window -> (count, t)."""
    pts = []
    for visits in bay_visits:
        for v in visits:
            e = v["end"] if v["end"] is not None else w_end
            cs, ce = max(v["start"], w_start), min(e, w_end)
            if ce > cs:
                pts += [(cs, 1), (ce, -1)]
    best, cur, at = 0, 0, None
    for t, d in sorted(pts, key=lambda p: (p[0], p[1])):   # leaves first
        cur += d
        if cur > best:
            best, at = cur, t
    return best, at


def _period_numbers(visits, bays, w_start, w_end):
    """visits/ttat/utilisation for a window -- used for the previous
    period, so cards can show change."""
    window = max(1.0, w_end - w_start)
    ttats, utils = [], []
    for b in bays:
        occ, tt, _ = _stats_for(visits.get(b, []), w_start, w_end)
        ttats += tt
        utils.append(100 * occ / window)
    return {"visits": len(ttats),
            "ttat_avg": round(statistics.mean(ttats)) if ttats else None,
            "utilisation_pct": (round(statistics.mean(utils), 1)
                                if utils else 0)}


def _previous(visits, bays, events, now, days, d_start, d_end, w_end):
    """Same-length window just before the displayed one. For a day still
    in progress (today) it covers only the same elapsed time, so the
    comparison is like-for-like. None when there is no earlier data."""
    first = min(e[0] for e in events)
    if days is None and d_end == now and d_start == first:
        return None                                  # "all time"
    length = d_end - d_start
    p_start = d_start - length
    p_end = p_start + (w_end - d_start)
    if p_end <= first or p_end <= p_start:
        return None
    p_start = max(p_start, first)
    out = _period_numbers(visits, bays, p_start, p_end)
    out["label"] = ("same time yesterday" if length == 86400 and d_end > now
                    else "previous period")
    return out


def _highlights(visits, last_status, shown, w_start, w_end, now, timeline,
                overstay_sec):
    hl = {}
    peak = max(timeline, key=lambda r: r["arrivals"], default=None)
    if peak and peak["arrivals"]:
        hl["busiest_slot"] = {"t": peak["t"], "arrivals": peak["arrivals"]}
    n, at = _concurrency([visits[b["bay"]] for b in shown], w_start, w_end)
    if n:
        hl["peak_occupied"] = {"count": n, "t": int(at)}
    longest = None
    for b in shown:
        for v in visits[b["bay"]]:
            if (v["end"] is not None and not v["partial"]
                    and w_start <= v["end"] <= w_end):
                d = v["end"] - v["start"]
                if longest is None or d > longest["ttat"]:
                    longest = {"bay": b["bay"], "ttat": d, "t": v["start"]}
    if longest:
        hl["longest_stay"] = longest
    occ = [b for b in shown if b["status"] == "occupied"]
    if occ:
        o = min(occ, key=lambda b: b["since"])
        hl["longest_in_bay_now"] = {"bay": o["bay"], "since": o["since"]}
    idle = [b for b in shown if b["status"] == "empty"]
    if idle:
        i = min(idle, key=lambda b: b["since"])
        hl["longest_idle_now"] = {"bay": i["bay"], "since": i["since"]}
    hl["overstay_sec"] = overstay_sec
    hl["overstay_bays"] = [b["bay"] for b in occ
                           if overstay_sec and now - b["since"] > overstay_sec]
    return hl


def build_stats(events, now, days=None, group=None, prefix_len=2,
                start=None, end=None, overstay_sec=4 * 3600, plates=None):
    """Overview for the last `days` days, or the explicit [start, end)
    range in epoch seconds (e.g. today 00:00 -> 24:00). `group` restricts
    bays/charts/cards to one bay group; `groups` always lists all."""
    if not events:
        return {"generated": int(now), "bays": [], "groups": [],
                "fleet": {}, "by_hour": [0] * 24, "by_day": [],
                "timeline": [], "unit": "hour", "window_sec": 0,
                "group": group, "range_start": int(now),
                "range_end": int(now), "highlights": {}, "prev": None,
                "ttat_hist": {"labels": TTAT_LABELS,
                              "counts": [0] * len(TTAT_LABELS)}}
    d_start, d_end, w_start, w_end, window = _resolve_window(
        events, now, days, start, end)
    unit = "hour" if d_end - d_start <= HOURLY_MAX_SEC else "day"
    visits = build_visits(events, now)
    last_status = {}
    for e in sorted(events, key=lambda e: e[0]):
        last_status[e[1]] = (e[2], e[0])

    bays, grp = [], {}
    for bay in sorted(visits):
        occ_sec, ttats, last_end = _stats_for(visits[bay], w_start, w_end)
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
        if unit == "hour":
            b["segments"] = _segments(visits[b["bay"]], d_start, d_end,
                                      plates, b["bay"], now)
        for v in visits[b["bay"]]:
            if (v["end"] is None or v["partial"]
                    or not w_start <= v["end"] <= w_end):
                continue
            all_ttat.append(v["end"] - v["start"])
            if v["start"] >= w_start:
                dt = datetime.fromtimestamp(v["start"])
                by_hour[dt.hour] += 1
                d = dt.strftime("%Y-%m-%d")
                by_day[d] = by_day.get(d, 0) + 1
    timeline = _timeline([visits[b["bay"]] for b in shown], d_start,
                         d_end, w_start, w_end, unit, now)
    for b in shown:
        b["overstay"] = bool(overstay_sec and b["status"] == "occupied"
                             and now - b["since"] > overstay_sec)
    return {
        "generated": int(now), "window_sec": int(window), "days": days,
        "range_start": int(d_start), "range_end": int(d_end), "unit": unit,
        "group": group or None, "bays": shown, "groups": groups,
        "highlights": _highlights(visits, last_status, shown, w_start,
                                  w_end, now, timeline, overstay_sec),
        "prev": _previous(visits, [b["bay"] for b in shown], events, now,
                          days, d_start, d_end, w_end),
        "ttat_hist": {"labels": TTAT_LABELS, "counts": _ttat_hist(all_ttat)},
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
        "timeline": timeline,
        "by_hour": by_hour,
        "by_day": [{"day": d, "visits": by_day[d]} for d in sorted(by_day)],
    }


def build_bay_detail(events, bay, now, days=None, prefix_len=2, limit=300,
                     start=None, end=None, plates=None):
    """Drill-down for one bay: summary, actual-clock timeline, hour-of-day
    pattern, and one entry per truck visit. None for a bay with no events."""
    if not any(e[1] == bay for e in events):
        return None
    d_start, d_end, w_start, w_end, window = _resolve_window(
        events, now, days, start, end)
    unit = "hour" if d_end - d_start <= HOURLY_MAX_SEC else "day"
    visits = build_visits(events, now).get(bay, [])
    occ_sec, ttats, last_end = _stats_for(visits, w_start, w_end)
    st = max((e for e in events if e[1] == bay), key=lambda e: e[0])

    arrivals = [0] * 24
    occ_by_hour = [0.0] * 24
    for v in visits:
        e = v["end"] if v["end"] is not None else w_end
        cs, ce = max(v["start"], w_start), min(e, w_end)
        for b, sec in (_slices(cs, ce, "hour") if ce > cs else []):
            occ_by_hour[datetime.fromtimestamp(b).hour] += sec
        if w_start <= v["start"] <= w_end and not v["partial"]:
            arrivals[datetime.fromtimestamp(v["start"]).hour] += 1
    seen = [0.0] * 24
    if w_end > w_start:
        for b, sec in _slices(w_start, w_end, "hour"):
            seen[datetime.fromtimestamp(b).hour] += sec
    occ_pct = [round(100 * o / s, 1) if s else 0 for o, s in
               zip(occ_by_hour, seen)]

    listed = [v for v in visits if v["start"] < d_end
              and (v["end"] is None or v["end"] >= d_start)][-limit:]
    return {
        "generated": int(now), "window_sec": int(window), "days": days,
        "range_start": int(d_start), "range_end": int(d_end), "unit": unit,
        "bay": bay, "group": group_of(bay, prefix_len),
        "status": st[2], "since": st[0],
        "visits": len(ttats),
        "utilisation_pct": round(100 * occ_sec / window, 1),
        "last_visit_end": last_end, **_ttat_summary(ttats),
        "timeline": _timeline([visits], d_start, d_end, w_start, w_end,
                              unit, now),
        "arrivals_by_hour": arrivals, "occupancy_pct_by_hour": occ_pct,
        "ttat_hist": {"labels": TTAT_LABELS, "counts": _ttat_hist(ttats)},
        "trucks": [{"start": v["start"], "end": v["end"],
                    "ttat": (v["end"] - v["start"]
                             if v["end"] is not None else None),
                    "partial": v["partial"], "class": v["cls"],
                    "plate": (plates.for_visit(bay, v["start"], v["end"] or now)
                              if plates else None)}
                   for v in listed],
    }


def build_entry(events, bay, start, now, prefix_len=2):
    """One truck entry (identified by bay + arrival epoch) with context:
    gap since the previous truck, how its stay compares with the bay's
    usual one, how many other bays were busy at arrival, and the
    previous/next entry for navigation. None if there is no such visit."""
    allv = build_visits(events, now)
    vs = allv.get(bay, [])
    idx = next((i for i, v in enumerate(vs) if v["start"] == int(start)), None)
    if idx is None:
        return None
    v = vs[idx]
    done = [x["end"] - x["start"] for x in vs
            if x["end"] is not None and not x["partial"]]
    ttat = v["end"] - v["start"] if v["end"] is not None else None
    prev_v = vs[idx - 1] if idx else None
    avg = statistics.mean(done) if done else None
    others = sum(1 for b, bv in allv.items() if b != bay and any(
        o["start"] <= v["start"] and (o["end"] is None or o["end"] > v["start"])
        for o in bv))
    rank = None
    if ttat is not None and not v["partial"] and done:
        rank = 1 + sum(1 for x in done if x > ttat)
    return {
        "generated": int(now), "bay": bay, "group": group_of(bay, prefix_len),
        "start": v["start"], "end": v["end"], "ttat": ttat,
        "source": v.get("src"),
        "in_bay_for": int(now - v["start"]) if v["end"] is None else None,
        "partial": v["partial"], "class": v["cls"],
        "confidence": v["conf"],
        "arrival_image": v["img_in"], "departure_image": v["img_out"],
        "idle_before": (v["start"] - prev_v["end"]
                        if prev_v and prev_v["end"] is not None else None),
        "bay_avg_ttat": round(avg) if avg is not None else None,
        "vs_avg_pct": (round(100 * (ttat - avg) / avg)
                       if ttat is not None and avg else None),
        "rank_longest": rank, "of_visits": len(done),
        "other_bays_occupied_at_arrival": others,
        "prev_start": prev_v["start"] if prev_v else None,
        "next_start": vs[idx + 1]["start"] if idx + 1 < len(vs) else None,
    }


def visits_csv(events, now, days=None, group=None, bay=None, prefix_len=2,
               start=None, end=None):
    """CSV text: one row per visit overlapping the range."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["bay", "group", "arrived", "left", "ttat_sec", "status",
                "already_there_at_startup", "class", "arrival_image",
                "departure_image", "source"])
    if events:
        d_start, d_end, *_ = _resolve_window(events, now, days, start, end)
        iso = lambda t: (datetime.fromtimestamp(t).astimezone()
                         .isoformat(timespec="seconds"))
        for b, vs in sorted(build_visits(events, now).items()):
            if bay and b != bay or group and group_of(b, prefix_len) != group:
                continue
            for v in vs:
                if v["start"] >= d_end or (v["end"] is not None
                                           and v["end"] < d_start):
                    continue
                w.writerow([b, group_of(b, prefix_len), iso(v["start"]),
                            iso(v["end"]) if v["end"] else "",
                            v["end"] - v["start"] if v["end"] else "",
                            "completed" if v["end"] else "in_bay",
                            int(v["partial"]), v["cls"] or "",
                            v["img_in"] or "", v["img_out"] or "",
                            v.get("src") or ""])
    return out.getvalue()


def load_events(db):
    return db.execute(
        "SELECT ts, bay, status, prev_status, class, confidence, image, source "
        "FROM events").fetchall()


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
.chart{display:flex;align-items:flex-end;gap:3px;height:128px;padding-top:16px}
.chart .c{flex:1;min-width:0;height:100%;display:flex;flex-direction:column;justify-content:flex-end}
.chart .c>*{flex:none}.chart .v{font-size:10px;line-height:13px;text-align:center;color:var(--mut);font-weight:600;white-space:nowrap;overflow:visible}
.chart i{display:block;background:var(--acc);border-radius:3px 3px 0 0;min-height:1px}
.chart.b i{background:var(--acc2)}.chart i.part{opacity:.4}
.card small{display:block;margin-top:2px}.up{color:var(--ok)}.dn{color:var(--bad)}
.chart .c[data-id],.gantt .seg,tr[data-entry]{cursor:pointer}.chart .c[data-id]:hover i{filter:brightness(1.15)}
.gantt .seg:hover{filter:brightness(1.15);outline:1px solid var(--fg)}
.shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.shots figure{margin:0}
.shots img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}.shots figcaption{color:var(--mut);font-size:12px;margin-top:4px}
.nav{display:flex;justify-content:space-between;margin:0 0 10px}.noimg{color:var(--mut);padding:24px;border:1px dashed var(--line);border-radius:8px;text-align:center}
.flag{color:#d33;font-weight:600}tr.over td:first-child{color:#d33;font-weight:600}
.axis{display:flex;gap:3px;color:var(--mut);font-size:10px}.axis span{flex:1;text-align:center;overflow:hidden}
.tile{padding:8px 8px 6px;cursor:zoom-in;display:flex;flex-direction:column;gap:4px}
.tile small{font-size:11px}.tile img{width:100%;height:auto;border-radius:6px;display:block;background:var(--bg)}
.tile.big{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.92);border:0;border-radius:0;align-items:center;justify-content:center;padding:24px;cursor:zoom-out}
.tile.big small{position:absolute;top:14px;left:18px;color:#ddd;font-size:13px}
.tile.big img{width:min(92vw,1400px);max-height:92vh;height:auto;object-fit:contain}
small{color:var(--mut)}.gantt{position:relative;display:grid;grid-template-columns:90px 1fr;gap:3px 8px;align-items:center}
.gantt .lbl{font-size:12px;cursor:pointer}.gantt .track{position:relative;height:16px;background:var(--bg);border-radius:3px}
.gantt .seg{position:absolute;top:0;bottom:0;background:var(--bad);border-radius:2px;min-width:2px}.gantt .seg.part{opacity:.45}
.gantt .now{position:absolute;top:0;bottom:0;width:2px;background:var(--acc)}
.gantt .grid{position:absolute;top:0;bottom:0;width:1px;background:var(--line)}
.ticks{position:relative;height:16px;font-size:10px;color:var(--mut)}.ticks span{position:absolute;transform:translateX(-50%)}
a.back{color:var(--acc);cursor:pointer;text-decoration:none}
</style></head><body><main>
<header><h1><a href="#/">Bay stats</a></h1><span><select id="days">
<option value="today" selected>Today</option><option value="yesterday">Yesterday</option>
<option value="1">Last 24 h</option><option value="7">Last 7 days</option>
<option value="30">Last 30 days</option><option value="all">All time</option>
<option value="date">Pick a day…</option></select>
<input type="date" id="pick" style="display:none;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px">
<small id="gen"></small> <a id="csv" class="back" href="#" title="Visits in this range as CSV">⬇ CSV</a></span></header>
<div id="view"></div>
</main><script>
const $=id=>document.getElementById(id);
let data=null,sort={k:'bay',d:1},group='',route=location.hash||'#/';
const dur=s=>s==null?'–':s>=3600?(s/3600).toFixed(1)+' h':s>=60?Math.round(s/60)+' min':s+' s';
const ago=t=>t?dur(Math.max(0,data.generated-t))+' ago':'–';
const fmt=t=>t?new Date(t*1000).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'–';
const card=(l,v,sub='')=>`<div class=card><b>${v}</b><span>${l}</span>${sub?`<small>${sub}</small>`:''}</div>`;
const delta=(cur,prev,fmtf,label,lowerBetter)=>{if(prev==null||cur==null)return '';const d=cur-prev;if(!d)return `= ${label}`;
const good=lowerBetter?d<0:d>0;return `<span class="${good?'up':'dn'}">${d>0?'▲':'▼'} ${fmtf(Math.abs(d))}</span> vs ${label}`};
const hhmm=t=>new Date(t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
const bars=(vals,labels,cls='',tip=(i,v)=>v,parts=[],lab=(i,v)=>v?String(v):'',ids=[])=>{const m=Math.max(1,...vals),show=vals.length<=40;
return `<div class="chart ${cls}">`+vals.map((v,i)=>`<div class=c ${ids[i]?`data-id="${ids[i]}"`:''} title="${labels[i]}: ${tip(i,v)}${ids[i]?' · click for detail':''}"><b class=v>${show?lab(i,v):''}</b><i class="${parts[i]?'part':''}" style="height:${100*v/m}%"></i></div>`).join('')+
`</div><div class=axis>`+labels.map((l,i)=>`<span>${vals.length>14&&i%Math.ceil(vals.length/14)?'':l}</span>`).join('')+`</div>`};
const pad=n=>String(n).padStart(2,'0');
const tlabel=(t,unit)=>{const d=new Date(t*1000);return unit=='day'?`${d.getMonth()+1}-${pad(d.getDate())}`:(d.getHours()==0?`${d.getMonth()+1}/${d.getDate()}`:pad(d.getHours()))};
const tfull=(t,unit)=>{const d=new Date(t*1000);return d.toLocaleString([],unit=='day'?{month:'short',day:'numeric'}:{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})};
const timeline=(tl,unit,key,cls,suffix)=>bars(tl.map(r=>r[key]??0),tl.map(r=>tlabel(r.t,unit)),cls,
(i,v)=>tl[i][key]==null?'no data':v+suffix,[],(i,v)=>v?(suffix=='%'?Math.round(v)+'%':v):'');
function gantt(rows,d){const W0=d.range_start,W1=d.range_end,span=W1-W0,p=t=>100*(Math.min(Math.max(t,W0),W1)-W0)/span;
const step=span<=26*3600?2*3600:span<=50*3600?4*3600:6*3600;let ticks='',grid='';
for(let t=Math.ceil(W0/step)*step;t<W1;t+=step){ticks+=`<span style="left:${p(t)}%">${tlabel(t,'hour')}</span>`;grid+=`<i class=grid style="left:${p(t)}%"></i>`}
const nowL=d.generated>W0&&d.generated<W1?`<i class=now style="left:${p(d.generated)}%"></i>`:'';
return `<div class=gantt><span></span><div class=ticks>${ticks}</div>`+rows.map(r=>`<span class=lbl data-bay="${r.bay}">${r.bay}</span><div class=track>${grid}`+
(r.segments||[]).map(([s,e,pt,pl])=>`<i class="seg ${pt?'part':''}" data-bay="${r.bay}" data-start="${s}" style="left:${p(s)}%;width:${Math.max(0.2,p(e??d.generated)-p(s))}%" title="${fmt(s)} → ${e?fmt(e):'now'} · ${pl?'Truck '+pl:'no truck number read'}"></i>`).join('')+nowL+`</div>`).join('')+`</div>`}
const rangeNote=d=>`<small>${tfull(d.range_start,'hour')} → ${tfull(d.range_end,'hour')}</small>`;
const HRS=[...Array(24).keys()].map(h=>String(h).padStart(2,'0'));
const COLS=[['bay','Bay'],['status','Now'],['visits','Visits'],['utilisation_pct','Utilisation'],
['ttat_avg','Avg TTAT'],['ttat_median','Median'],['ttat_p90','P90'],['ttat_min','Min'],
['ttat_max','Max'],['last_visit_end','Last departure']];
const util=u=>`<span class=bar style="width:${Math.min(100,u)*1.2}px"></span>${u}%`;
const pill=(s,t)=>`<span class="pill ${s}">${s}</span> <small>${t}</small>`;

function highlights(hl,d){const c=[],unit=d.unit=='day'?'day':'hour';
if(hl.busiest_slot)c.push(card(`Busiest ${unit}`,(d.unit=='hour'?pad(new Date(hl.busiest_slot.t*1000).getHours())+':00':tlabel(hl.busiest_slot.t,'day')),hl.busiest_slot.arrivals+' arrivals'));
if(hl.peak_occupied)c.push(card('Peak occupancy',hl.peak_occupied.count+' bays',`at ${hhmm(hl.peak_occupied.t)}`));
if(hl.longest_stay)c.push(card('Longest stay',dur(hl.longest_stay.ttat),`${hl.longest_stay.bay} · arrived ${fmt(hl.longest_stay.t)}`));
if(hl.longest_in_bay_now)c.push(card('In bay longest (now)',dur(d.generated-hl.longest_in_bay_now.since),hl.longest_in_bay_now.bay));
if(hl.longest_idle_now)c.push(card('Idle longest (now)',dur(d.generated-hl.longest_idle_now.since),hl.longest_idle_now.bay+' empty'));
if(hl.overstay_sec)c.push(card(`Over ${dur(hl.overstay_sec)} in bay`,`<span class="${hl.overstay_bays.length?'flag':''}">${hl.overstay_bays.length}</span>`,hl.overstay_bays.join(', ')||'none'));
return c.length?`<div class=cards>${c.join('')}</div>`:''}
const histo=(hs,cls)=>bars(hs.counts,hs.labels,cls||'b',(i,v)=>v+' trucks');
function renderOverview(){const f=data.fleet||{};
const gs=data.groups||[];
let h=`<div class=chips><button class="chip ${group?'':'on'}" data-g="">All groups</button>`+
gs.map(g=>`<button class="chip ${group==g.group?'on':''}" data-g="${g.group}">${g.group} <small>(${g.bays})</small></button>`).join('')+`</div>`;
const pv=data.prev,pl=pv?pv.label:'';
h+=`<div class=cards>`+card('Occupied now',`${f.occupied_now??0} / ${f.bays??0}`)+
card('Visits',f.visits??0,pv?delta(f.visits,pv.visits,x=>x,pl):'')+
card('Avg TTAT',dur(f.ttat_avg),pv?delta(f.ttat_avg,pv.ttat_avg,dur,pl,true):'')+card('Median TTAT',dur(f.ttat_median))+
card('Avg utilisation',(f.utilisation_pct??0)+'%',pv?delta(f.utilisation_pct,pv.utilisation_pct,x=>x.toFixed(1)+' pts',pl):'')+`</div>`;
h+=highlights(data.highlights||{},data);
if(gs.length>1)h+=`<section><h2>By group <small>(first characters of the bay name · click to filter)</small></h2><table><tr>
<th>Group</th><th>Bays</th><th>Occupied now</th><th>Visits</th><th>Utilisation</th><th>Avg TTAT</th><th>Median</th><th>P90</th></tr>`+
gs.map(g=>`<tr class=click data-g="${g.group}"><td>${g.group}</td><td>${g.bays}</td><td>${g.occupied_now}</td><td>${g.visits}</td>
<td>${util(g.utilisation_pct)}</td><td>${dur(g.ttat_avg)}</td><td>${dur(g.ttat_median)}</td><td>${dur(g.ttat_p90)}</td></tr>`).join('')+`</table></section>`;
const rows=[...data.bays].sort((a,b)=>{const x=a[sort.k],y=b[sort.k];
return ((x==null)-(y==null))||(x>y?1:x<y?-1:0)*sort.d});
h+=`<section><h2>Per bay <small>(click a heading to sort, a bay for detail · TTAT = occupied → empty)</small></h2><table><tr>`+
COLS.map(([k,l])=>`<th data-k=${k}>${l}${sort.k==k?(sort.d>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>'+
(rows.map(b=>`<tr class="click ${b.overstay?'over':''}" data-bay="${b.bay}"><td>${b.bay}${b.overstay?' ⚠':''}</td><td>${pill(b.status,ago(b.since))}</td>
<td>${b.visits}</td><td>${util(b.utilisation_pct)}</td><td>${dur(b.ttat_avg)}</td><td>${dur(b.ttat_median)}</td><td>${dur(b.ttat_p90)}</td>
<td>${dur(b.ttat_min)}</td><td>${dur(b.ttat_max)}</td><td>${ago(b.last_visit_end)}</td></tr>`).join('')||'<tr><td>No data yet</td></tr>')+`</table></section>`;
const u=data.unit,per=u=='day'?'day':'hour';
if(u=='hour'&&data.bays.length)h+=`<section><h2>Bay timeline <small>(orange = truck in bay · click a bay · ${rangeNote(data).replace(/<\/?small>/g,'')})</small></h2>${gantt(data.bays,data)}</section>`;
h+=`<section><h2>Arrivals per ${per} <small>(actual clock time)</small></h2>${timeline(data.timeline,u,'arrivals','','')}</section>`;
h+=`<section><h2>Occupancy per ${per} <small>(% of bays occupied)</small></h2>${timeline(data.timeline,u,'occ_pct','b','%')}</section>`;
h+=`<section><h2>How long trucks stay <small>(TTAT distribution · completed visits)</small></h2>${histo(data.ttat_hist)}</section>`;
if(u=='day')h+=`<section><h2>Arrivals by hour of day <small>(pattern across the range)</small></h2>${bars(data.by_hour,HRS)}</section>`;
$('view').innerHTML=h;
document.querySelectorAll('.lbl[data-bay]').forEach(e=>e.onclick=()=>{location.hash='#/bay/'+encodeURIComponent(e.dataset.bay)});
document.querySelectorAll('[data-g]').forEach(e=>e.onclick=()=>{group=e.dataset.g;load()});
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
sort=sort.k==k?{k,d:-sort.d}:{k,d:1};renderOverview()});
document.querySelectorAll('[data-bay]').forEach(e=>e.onclick=()=>{location.hash='#/bay/'+encodeURIComponent(e.dataset.bay)});}

function renderBay(){const d=data;
let h=`<p><a class=back href="#/">← All bays</a></p><h2 style="font-size:18px">${d.bay} <small>group ${d.group}</small> ${pill(d.status,ago(d.since))}</h2>`;
h+=`<div class=cards>`+card('Visits',d.visits)+card('Utilisation',d.utilisation_pct+'%')+card('Avg TTAT',dur(d.ttat_avg))+
card('Median TTAT',dur(d.ttat_median))+card('P90 TTAT',dur(d.ttat_p90))+card('Longest',dur(d.ttat_max))+card('Last departure',ago(d.last_visit_end))+
`<div class="card tile" id=tilebox></div></div>`;
const u=d.unit,per=u=='day'?'day':'hour';
if(u=='hour')h+=`<section><h2>Timeline <small>(orange = truck in bay)</small></h2>${gantt([{bay:d.bay,segments:d.trucks.map(x=>[x.start,x.end,x.partial,x.plate])}],d)}</section>`;
h+=`<section><h2>Arrivals per ${per} <small>(actual clock time)</small></h2>${timeline(d.timeline,u,'arrivals','','')}</section>`;
h+=`<section><h2>Occupancy per ${per} <small>(% of each ${per} the bay was occupied)</small></h2>${timeline(d.timeline,u,'occ_pct','b','%')}</section>`;
h+=`<section><h2>How long trucks stay here <small>(TTAT distribution)</small></h2>${histo(d.ttat_hist)}</section>`;
if(u=='day'){h+=`<section><h2>Arrivals by hour of day <small>(pattern across the range)</small></h2>${bars(d.arrivals_by_hour,HRS)}</section>`;
h+=`<section><h2>Occupancy by hour of day <small>(% of each hour the bay was occupied)</small></h2>${bars(d.occupancy_pct_by_hour,HRS,'b',(i,v)=>v+'%',[],(i,v)=>v?Math.round(v)+'%':'')}</section>`}
const t=d.trucks,done=t.filter(x=>x.ttat!=null);
h+=`<section><h2>Each truck entry <small>(bar = time in bay, oldest → newest; faded = already there when monitoring started · click a bar)</small></h2>`+
(done.length?bars(done.map(x=>x.ttat),done.map(x=>new Date(x.start*1000).toLocaleDateString([],{month:'numeric',day:'numeric'})),'',(i,v)=>dur(v)+' from '+fmt(done[i].start),done.map(x=>x.partial),(i,v)=>dur(v),done.map(x=>x.start)):'<small>No completed visits in this range.</small>')+`</section>`;
h+=`<section><h2>Entries <small>(newest first · click one for its detail)</small></h2><table><tr><th>Arrived</th><th>Left</th><th>TTAT</th><th>Truck</th><th>Detected as</th></tr>`+
[...t].reverse().map(x=>`<tr data-entry="${d.bay}|${x.start}"><td>${fmt(x.start)}${x.partial?' <small>(already there at startup)</small>':''}</td>
<td>${x.end?fmt(x.end):'<small>in bay now</small>'}</td><td>${dur(x.ttat)}</td><td>${x.plate||'<small>–</small>'}</td><td>${x.class||'–'}</td></tr>`).join('')+`</table></section>`;
$('view').innerHTML=h;wireEntries();placeTile(d.bay)}
// One snapshot per opening of the bay page: the same <img> is moved into
// the tile on every stats refresh, so it is never fetched again. Clicking
// the tile enlarges it in place (no second request).
let liveImg=null,liveBay=null;
function placeTile(bay){const box=$('tilebox');if(!box)return;
if(!liveImg||liveBay!==bay){liveImg=new Image();liveBay=bay;liveImg.alt=bay+' live view';
liveImg.onerror=()=>{liveImg.alt='live view unavailable'};liveImg.src='/snapshot/'+encodeURIComponent(bay)}
box.innerHTML='<small>Live view · click to enlarge</small>';box.appendChild(liveImg);
box.onclick=()=>box.classList.toggle('big')}
document.addEventListener('keydown',e=>{if(e.key=='Escape')document.querySelectorAll('.tile.big').forEach(t=>t.classList.remove('big'))});
const goEntry=(bay,start)=>{location.hash='#/entry/'+encodeURIComponent(bay)+'/'+start};
function wireEntries(){document.querySelectorAll('tr[data-entry]').forEach(e=>e.onclick=()=>{const [b,s]=e.dataset.entry.split('|');goEntry(b,s)});
document.querySelectorAll('.c[data-id]').forEach(e=>e.onclick=()=>goEntry(data.bay,e.dataset.id));
document.querySelectorAll('.seg[data-start]').forEach(e=>e.onclick=()=>goEntry(e.dataset.bay,e.dataset.start))}
const shot=(f,cap)=>f?`<figure><a href="/img/${encodeURIComponent(f)}" target="_blank"><img loading=lazy src="/img/${encodeURIComponent(f)}" alt="${cap}"></a><figcaption>${cap}</figcaption></figure>`:
`<div class=noimg>No snapshot for ${cap.toLowerCase()}.<br><small>Snapshots are saved only while <code>event_images_dir</code> is enabled in bay_occupancy.</small></div>`;
function renderEntry(){const e=data,bay=encodeURIComponent(e.bay);
const nav=`<div class=nav><span>${e.prev_start?`<a class=back href="#/entry/${bay}/${e.prev_start}">← Previous truck</a>`:''}</span>
<span><a class=back href="#/bay/${bay}">${e.bay} · all entries</a></span><span>${e.next_start?`<a class=back href="#/entry/${bay}/${e.next_start}">Next truck →</a>`:''}</span></div>`;
const inBay=e.end==null;
let h=`${nav}<h2 style="font-size:18px">${e.bay} <small>group ${e.group}</small> · ${fmt(e.start)} ${inBay?pill('occupied','in bay now'):pill('empty','completed')}</h2>`;
const vs=e.vs_avg_pct==null?'':`<span class="${e.vs_avg_pct>0?'dn':'up'}">${e.vs_avg_pct>0?'▲':'▼'} ${Math.abs(e.vs_avg_pct)}%</span> vs this bay's average (${dur(e.bay_avg_ttat)})`;
h+=`<div class=cards>`+card('Arrived',fmt(e.start),e.partial?'already there when monitoring started':'')+
card('Left',inBay?'in bay':fmt(e.end))+card(inBay?'In bay for':'Time in bay (TTAT)',dur(inBay?e.in_bay_for:e.ttat),vs)+
(e.rank_longest?card('Ranking',`#${e.rank_longest} longest`,`of ${e.of_visits} visits in this bay`):'')+
card('Bay was empty for',dur(e.idle_before),'before this truck')+card('Other bays busy',e.other_bays_occupied_at_arrival,'at arrival')+
card('Detected as',e.class||'–',e.confidence!=null?'confidence '+e.confidence:'')+
card('Bay status from',e.source=='external'?'site bay API':e.source=='camera'?'camera':'–','')+`</div>`;
h+=`<section><h2>What the camera saw</h2><div class=shots>${shot(e.arrival_image,'Arrival · '+fmt(e.start))}${shot(e.departure_image,inBay?'Departure · not yet':'Departure · '+fmt(e.end))}</div></section>`;
$('view').innerHTML=h}

function rangeQuery(){const v=$('days').value,q=[];
const mid=(off)=>{const d=new Date();d.setHours(0,0,0,0);d.setDate(d.getDate()+off);return d};
const span=(a,b)=>{q.push('start='+Math.floor(a/1000),'end='+Math.floor(b/1000))};
$('pick').style.display=v=='date'?'':'none';
if(v=='today')span(mid(0),mid(1));
else if(v=='yesterday')span(mid(-1),mid(0));
else if(v=='date'){const p=$('pick').value;if(p){const a=new Date(p+'T00:00:00'),b=new Date(a);b.setDate(b.getDate()+1);span(a,b)}else span(mid(0),mid(1))}
else if(v!='all')q.push('days='+v);
return q}
async function load(){route=location.hash||'#/';const q=rangeQuery();
const m=route.match(/^#\/bay\/(.+)$/),en=route.match(/^#\/entry\/(.+)\/(\d+)$/);
let url;if(en){url='/api/entry/'+en[1]+'/'+en[2]}
else if(m){url='/api/bay/'+m[1]+(q.length?'?'+q.join('&'):'')}
else{if(group)q.push('group='+encodeURIComponent(group));url='/api/stats'+(q.length?'?'+q.join('&'):'')}
const cq=q.filter(x=>!x.startsWith('group=')).concat(m?['bay='+m[1]]:(group?['group='+encodeURIComponent(group)]:[]));
$('csv').href='/api/visits.csv'+(cq.length?'?'+cq.join('&'):'');
const r=await fetch(url);
if(!r.ok){$('view').innerHTML='<p><a class=back href="#/">← All bays</a></p><p>Not found.</p>';return}
data=await r.json();en?renderEntry():m?renderBay():renderOverview();
if(!m)liveImg=null;   // leaving the bay page: next open fetches a fresh frame
if(!en&&!m)wireEntries();
$('gen').textContent=' updated '+new Date(data.generated*1000).toLocaleTimeString()}
$('days').onchange=load;$('pick').onchange=load;window.onhashchange=load;load();setInterval(load,60000);
</script></body></html>"""


def make_handler(db, csv_path, prefix_len=2, overstay_sec=4 * 3600,
                 image_dir=None, webhook_url=None, plates=None):
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
                def num(k):
                    v = q.get(k, [""])[0]
                    return float(v) if v else None
                days, start, end = num("days"), num("start"), num("end")
            except ValueError:
                return self._json(400, {"error": "bad days/start/end"})
            if u.path == "/api/stats":
                with _lock:
                    import_csv(db, csv_path)
                    stats = build_stats(load_events(db), time.time(), days,
                                        q.get("group", [None])[0], prefix_len,
                                        start, end, overstay_sec,
                                        plates=plates)
                return self._json(200, stats)
            if u.path.startswith("/snapshot/"):
                bay = unquote(u.path[len("/snapshot/"):])
                try:
                    data = fetch_live_snapshot(webhook_url, bay)
                except (OSError, ValueError) as e:
                    log.info(f"live snapshot for {bay} unavailable: {e}")
                    return self._json(503, {"error": "live snapshot unavailable"})
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if u.path.startswith("/img/"):
                name = unquote(u.path[5:])
                f = (Path(image_dir) / name) if image_dir else None
                if (not f or not IMG_RE.match(name) or not f.is_file()):
                    return self._json(404, {"error": "no such image"})
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
                return
            if u.path.startswith("/api/entry/"):
                parts = unquote(u.path[len("/api/entry/"):]).rsplit("/", 1)
                try:
                    bay, st = parts[0], int(parts[1])
                except (IndexError, ValueError):
                    return self._json(400, {"error": "use /api/entry/<bay>/<start>"})
                with _lock:
                    import_csv(db, csv_path)
                    ent = build_entry(load_events(db), bay, st, time.time(),
                                      prefix_len)
                if ent is None:
                    return self._json(404, {"error": "unknown entry"})
                return self._json(200, ent)
            if u.path == "/api/visits.csv":
                with _lock:
                    import_csv(db, csv_path)
                    body = visits_csv(load_events(db), time.time(), days,
                                      q.get("group", [None])[0],
                                      q.get("bay", [None])[0], prefix_len,
                                      start, end)
                return self._send(200, body.encode(), "text/csv")
            if u.path.startswith("/api/bay/"):
                bay = unquote(u.path[len("/api/bay/"):])
                with _lock:
                    import_csv(db, csv_path)
                    det = build_bay_detail(load_events(db), bay, time.time(),
                                           days, prefix_len, start=start,
                                           end=end, plates=plates)
                if det is None:
                    return self._json(404, {"error": "unknown bay"})
                return self._json(200, det)
            self._json(404, {"error": "not found"})

        def log_message(self, *a):
            pass
    return H


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    raw = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    d = {**DEFAULTS, **(raw.get("dashboard") or {})}
    csv_path = raw.get("events_csv") or "bay_events.csv"
    db = connect(d["db_path"])
    n = import_csv(db, csv_path)
    print(f"imported {n} events from {csv_path}; "
          f"serving on http://{d['host']}:{d['port']}/")
    ext = raw.get("external") or {}
    plates = None
    if api_key_value(ext.get("api_key")) not in (None, "", "CHANGE_ME"):
        plates = PlateSource(ext.get("base_url", "http://10.69.12.41:8502"),
                             ext["api_key"], ext.get("plate_match_window_sec", 900))
        print("truck numbers from site detections API: on")
    ThreadingHTTPServer((d["host"], int(d["port"])), make_handler(
        db, csv_path, int(d["group_prefix_len"]),
        int(float(d["overstay_hours"]) * 3600),
        raw.get("event_images_dir", "event_images"),
        d["bay_webhook_url"], plates)).serve_forever()


if __name__ == "__main__":
    main()
