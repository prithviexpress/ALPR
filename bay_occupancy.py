"""Simple bay occupancy monitor: is each bay occupied or empty?

Fully standalone: one file, one config.json, one cameras.json.

How it decides
  1. Each round, every enabled camera's snapshot is fetched in parallel.
  2. The truck model (Truck_model.pt) runs on the frame. A frame is
     "occupied" if ANY truck class (Enter/Docked, open/closed -- not
     Number_Plate) is detected with conf >= min_conf AND its box covers
     at least min_box_area_frac of the frame. The area filter removes
     small false hits (a distant truck passing the yard, a cab edge).
  3. A bay only changes state after N consecutive frames agree
     (occupied_confirm_count / empty_confirm_count). Empty needs more
     because a truck briefly missed by the model must not flap the bay.
  4. A failed fetch never counts as "empty" -- the state is simply held.
  5. MQTT is published ONLY on a confirmed change (plus the first
     confirmed state after startup), retained, so a late subscriber
     still sees the current state of every bay.

Query: GET http://<host>:8081/bay/<bay>  ->  {bay, status, timestamp,
       status_since, class, confidence, snapshot_base64}. Add ?image=0 to
       skip the image; GET /bays lists every bay without images.

Run:  python bay_occupancy.py [config.json] [cameras.json]
"""
import base64
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import logging

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

logging.basicConfig(level=logging.INFO, format=(
    "%(asctime)s [%(levelname)s] %(message)s"))
log = logging.getLogger("occupancy")


class SnapshotError(RuntimeError):
    pass


def fetch_snapshot(session, url, auth, connect_ms, read_ms):
    """One JPEG from the camera's snapshot URL -> decoded BGR frame."""
    try:
        resp = session.get(url, auth=auth,
                           timeout=(connect_ms / 1000, read_ms / 1000))
    except requests.RequestException as e:
        raise SnapshotError(f"request failed: {e}") from e
    if resp.status_code != 200:
        raise SnapshotError(f"HTTP {resp.status_code}")
    frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8),
                         cv2.IMREAD_COLOR)
    if frame is None:
        raise SnapshotError("could not decode JPEG")
    return frame


def load_cameras(path: Path) -> dict:
    """cameras.json: {"C9": {"ip": "10.0.0.9"}, ...}. Keys starting with
    "_" are comments; "enabled": false skips a camera."""
    raw = json.loads(path.read_text())
    cams = {b: c for b, c in raw.items()
            if not b.startswith("_") and c.get("enabled", True)}
    for b, c in cams.items():
        if not c.get("ip"):
            sys.exit(f"camera '{b}' has no 'ip' in {path}")
    return cams

DEFAULTS = {
    "model_path": "Truck_model.pt",
    "plate_class": "Number_Plate",
    "min_conf": 0.4,
    "imgsz": 640,
    "min_box_area_frac": 0.05,
    "occupied_confirm_count": 2,
    "empty_confirm_count": 4,
    "round_interval_sec": 3.0,
    "fetch_workers": 8,
    # Pixels trimmed from each edge of the snapshot BEFORE detection (and
    # before saving). A camera in cameras.json may carry its own "crop".
    "crop": {"top": 0, "bottom": 0, "left": 0, "right": 0},
    # Keep only the newest image per bay: <bay>_<status>_<YYYYmmdd_HHMMSS>.jpg
    "save_images": True,
    "save_dir": "latest",
    "save_interval_sec": 10,      # also saved immediately on a status change
    # One row per confirmed status change, all bays. "" disables.
    "events_csv": "bay_events.csv",
    # Query webhook: GET /bay/<bay> -> status + timestamp + base64 JPEG.
    "webhook": {"enabled": True, "host": "0.0.0.0", "port": 8081,
                "max_dimension": 640, "jpeg_quality": 80},
    "snapshot": {"url_template": "http://{ip}:{port}/snap.jpg", "port": 80,
                 "username": None, "password": None,
                 "connect_timeout_ms": 3000, "read_timeout_ms": 3000},
    "mqtt": {"host": "localhost", "port": 1883, "username": None,
             "password": None, "topic_prefix": "site/alpr/bay_occupancy",
             "retain": True},
}


def load_config(path: Path) -> dict:
    raw = json.loads(path.read_text())
    cfg = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in DEFAULTS.items()}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if isinstance(cfg.get(k), dict) and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def frame_is_occupied(boxes, frame_w, frame_h, min_conf, min_area_frac,
                      plate_class):
    """boxes: iterable of (class_name, conf, x1, y1, x2, y2).
    Returns (occupied, best) where best is the strongest qualifying box
    (class_name, conf, area_frac) or None."""
    best = None
    frame_area = float(frame_w * frame_h)
    for name, conf, x1, y1, x2, y2 in boxes:
        if name == plate_class or conf < min_conf:
            continue
        frac = max(0, x2 - x1) * max(0, y2 - y1) / frame_area
        if frac < min_area_frac:
            continue
        if best is None or conf > best[1]:
            best = (name, conf, frac)
    return best is not None, best


class BayDebouncer:
    """Holds a bay's confirmed state and flips it only after enough
    consecutive agreeing frames. State starts as None (unknown), so the
    first confirmed reading is reported as a change."""

    def __init__(self, occupied_n, empty_n):
        self.need = {True: occupied_n, False: empty_n}
        self.state = None          # True = occupied, False = empty
        self.streak_value = None
        self.streak = 0

    def update(self, occupied: bool):
        """Returns True if the confirmed state just changed."""
        if occupied == self.streak_value:
            self.streak += 1
        else:
            self.streak_value, self.streak = occupied, 1
        if occupied != self.state and self.streak >= self.need[occupied]:
            self.state = occupied
            return True
        return False


class Monitor:
    def __init__(self, cfg, cameras, model=None, mqtt_client=None):
        self.cfg = cfg
        self.cameras = cameras
        self.deb = {b: BayDebouncer(cfg["occupied_confirm_count"],
                                    cfg["empty_confirm_count"])
                    for b in self.cameras}
        self.model = model
        self.mqtt = mqtt_client
        self.latest = {}          # bay -> dict served by the webhook
        self.latest_lock = threading.Lock()
        self.last_change = {}     # bay -> (status, epoch) for the CSV
        self.last_saved = {}      # bay -> (path, time)
        if cfg["save_images"]:
            Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
            for b in self.cameras:
                self._remove_old_images(b)
        self.stop = threading.Event()
        self._tls = threading.local()   # one HTTP session per fetch thread

    # -- capture / detect -------------------------------------------------
    def _fetch(self, bay):
        if not hasattr(self._tls, "session"):
            self._tls.session = requests.Session()
            self._tls.auth = HTTPDigestAuth(self.cfg["snapshot"]["username"],
                                            self.cfg["snapshot"]["password"])
        s = self.cfg["snapshot"]
        url = s["url_template"].format(ip=self.cameras[bay]["ip"],
                                       port=s["port"])
        try:
            frame = fetch_snapshot(self._tls.session, url, self._tls.auth,
                                   s["connect_timeout_ms"],
                                   s["read_timeout_ms"])
            return bay, frame
        except SnapshotError as e:
            log.warning(f"{bay}: snapshot failed ({e}) -- state held")
            return bay, None

    def _crop(self, bay, frame):
        c = dict(self.cfg["crop"])
        c.update(self.cameras[bay].get("crop") or {})
        h, w = frame.shape[:2]
        t, b = int(c["top"]), int(c["bottom"])
        l, r = int(c["left"]), int(c["right"])
        if t + b >= h or l + r >= w:
            log.warning(f"{bay}: crop {c} leaves nothing of {w}x{h} "
                        f"-- using the full frame")
            return frame
        return frame[t:h - b, l:w - r]

    def _remove_old_images(self, bay):
        pat = re.compile(rf"^{re.escape(bay)}_(occupied|empty)_"
                         r"\d{8}_\d{6}\.jpg$")
        for f in Path(self.cfg["save_dir"]).iterdir():
            if pat.match(f.name):
                f.unlink(missing_ok=True)

    def save_image(self, bay, frame, status, force=False):
        """Newest image per bay only: write the new file, drop the old."""
        now = time.time()
        prev = self.last_saved.get(bay)
        if (not force and prev
                and now - prev[1] < self.cfg["save_interval_sec"]):
            return
        path = Path(self.cfg["save_dir"]) / (
            f"{bay}_{status}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(path)
        if prev and prev[0] != path:
            prev[0].unlink(missing_ok=True)
        self.last_saved[bay] = (path, now)

    def _detect(self, frame):
        res = self.model(frame, conf=self.cfg["min_conf"],
                         imgsz=self.cfg["imgsz"], verbose=False)
        names = self.model.names
        boxes = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                boxes.append((names[int(b.cls[0])], float(b.conf[0]),
                              x1, y1, x2, y2))
        h, w = frame.shape[:2]
        return frame_is_occupied(boxes, w, h, self.cfg["min_conf"],
                                 self.cfg["min_box_area_frac"],
                                 self.cfg["plate_class"])

    # -- publish ----------------------------------------------------------
    def publish(self, bay, occupied, best):
        payload = {
            "bay": bay,
            "status": "occupied" if occupied else "empty",
            "class": best[0] if best else None,
            "confidence": round(best[1], 3) if best else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        m = self.cfg["mqtt"]
        topic = f"{m['topic_prefix']}/{bay}"
        log.info(f"{bay}: -> {payload['status']} "
                 f"({payload['class']} {payload['confidence']})")
        self.record_event(payload)
        if self.mqtt:
            self.mqtt.publish(topic, json.dumps(payload), qos=1,
                              retain=bool(m.get("retain", True)))

    CSV_COLUMNS = ["timestamp", "bay", "status", "previous_status",
                   "previous_duration_sec", "class", "confidence"]

    def record_event(self, payload):
        """Append one row per confirmed change. previous_* are blank for a
        bay's first confirmed state after startup."""
        path = self.cfg.get("events_csv")
        if not path:
            return
        bay, now = payload["bay"], time.time()
        prev = self.last_change.get(bay)
        self.last_change[bay] = (payload["status"], now)
        row = [payload["timestamp"], bay, payload["status"],
               prev[0] if prev else "",
               round(now - prev[1]) if prev else "",
               payload["class"] or "", payload["confidence"] or ""]
        try:
            new = not Path(path).exists() or Path(path).stat().st_size == 0
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(self.CSV_COLUMNS)
                w.writerow(row)
        except OSError as e:
            log.warning(f"could not write {path}: {e}")

    # -- loop -------------------------------------------------------------
    def run_round(self, pool):
        for bay, frame in pool.map(self._fetch, list(self.cameras)):
            if frame is None:
                continue
            frame = self._crop(bay, frame)
            occupied, best = self._detect(frame)
            changed = self.deb[bay].update(occupied)
            if changed:
                self.publish(bay, occupied, best)
            if self.cfg["webhook"]["enabled"]:
                self._update_latest(bay, frame, occupied, best, changed)
            if self.cfg["save_images"]:
                # Confirmed state if there is one, else this frame's reading.
                state = self.deb[bay].state
                state = occupied if state is None else state
                self.save_image(bay, frame,
                                "occupied" if state else "empty",
                                force=changed)

    # -- query webhook ----------------------------------------------------
    def _update_latest(self, bay, frame, occupied, best, changed):
        """Keep what the webhook serves: a small JPEG per bay, encoded once
        per round (full-size frames for every bay would be gigabytes)."""
        w = self.cfg["webhook"]
        h, wd = frame.shape[:2]
        scale = min(1.0, w["max_dimension"] / max(h, wd))
        small = frame if scale == 1.0 else cv2.resize(
            frame, (int(wd * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY,
                                               int(w["jpeg_quality"])])
        state = self.deb[bay].state
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self.latest_lock:
            prev = self.latest.get(bay, {})
            self.latest[bay] = {
                "bay": bay,
                # Confirmed state only; "unknown" until debounce confirms.
                "status": ("unknown" if state is None
                           else "occupied" if state else "empty"),
                "timestamp": now,
                "status_since": (now if changed or "status_since" not in prev
                                 else prev["status_since"]),
                "class": best[0] if best else None,
                "confidence": round(best[1], 3) if best else None,
                "jpeg": buf.tobytes() if ok else None,
            }

    def query(self, bay, include_image=True):
        """Dict for one bay, or None if the bay is unknown/not scanned yet."""
        with self.latest_lock:
            d = self.latest.get(bay)
        if d is None:
            return None
        out = {k: v for k, v in d.items() if k != "jpeg"}
        if include_image:
            out["snapshot_base64"] = (base64.b64encode(d["jpeg"]).decode()
                                      if d["jpeg"] else None)
        return out

    def start_webhook(self):
        w = self.cfg["webhook"]
        mon = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                parts = [p for p in u.path.split("/") if p]
                q = parse_qs(u.query)
                if parts == ["healthz"]:
                    return self._send(200, {"ok": True,
                                            "bays": len(mon.cameras)})
                if parts == ["bays"]:
                    with mon.latest_lock:
                        bays = list(mon.latest)
                    return self._send(200, {"bays": [
                        mon.query(b, include_image=False) for b in bays]})
                bay = (parts[1] if len(parts) == 2 and parts[0] == "bay"
                       else (q.get("bay") or [None])[0]
                       if parts == ["bay"] else None)
                if bay is None:
                    return self._send(404, {"error": "use /bay/<bay>"})
                img = (q.get("image") or ["1"])[0] not in ("0", "false")
                d = mon.query(bay, include_image=img)
                if d is None:
                    known = bay in mon.cameras
                    return self._send(404 if not known else 503, {
                        "error": ("no reading yet for bay" if known
                                  else "unknown bay"), "bay": bay})
                self._send(200, d)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer((w["host"], int(w["port"])), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log.info(f"webhook on http://{w['host']}:{w['port']}/bay/<bay>")
        return srv

    def run(self):
        interval = self.cfg["round_interval_sec"]
        with ThreadPoolExecutor(self.cfg["fetch_workers"]) as pool:
            while not self.stop.is_set():
                t0 = time.time()
                try:
                    self.run_round(pool)
                except Exception:
                    log.exception("round failed")
                # Sleep once per ROUND, not per bay.
                self.stop.wait(max(0.0, interval - (time.time() - t0)))


def main():
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    cam_path = Path(sys.argv[2] if len(sys.argv) > 2 else "cameras.json")
    cfg = load_config(cfg_path)
    cameras = load_cameras(cam_path)

    from ultralytics import YOLO
    import paho.mqtt.client as mqtt
    if not Path(cfg["model_path"]).exists():
        sys.exit(f"model not found: {cfg['model_path']}")
    model = YOLO(cfg["model_path"])
    log.info(f"model classes: {sorted(model.names.values())}")

    m = cfg["mqtt"]
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if m.get("username"):
        client.username_pw_set(m["username"], m.get("password"))
    client.connect(m["host"], m["port"])
    client.loop_start()

    mon = Monitor(cfg, cameras, model, client)
    log.info(f"monitoring {len(mon.cameras)} bays")
    if cfg["webhook"]["enabled"]:
        mon.start_webhook()
    try:
        mon.run()
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
