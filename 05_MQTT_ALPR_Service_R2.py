# 05_MQTT_ALPR_Service_R2.py -- R1 + observability & hardening.
# Changes vs R1 (transcribed original):
#   1. Structured stage logging via debug_log() everywhere (MQTT/QUEUE/RTSP/
#      COLLECT/OCR/VOTE/RESULT/SETUP) with timings and rejection reasons.
#   2. Debug images per event (gated by DEBUG_SAVE_IMAGES):
#      audit/<bay>/<ts>/debug/00_first_frame.jpg   -- proves RTSP + scene
#      audit/<bay>/<ts>/debug/01_annotated_roi.jpg -- ROI with accepted boxes
#      audit/<bay>/<ts>/debug/prep_NN.jpg          -- exact image fed to OCR
#      audit/<bay>/<ts>/crop_NN.jpg                -- selected plate crops (as R1)
#   3. RTSP hardening: URL-encoded credentials, cap.isOpened() check with
#      fail-fast log, stream-open timing.
#   4. Audit retention: folders older than AUDIT_RETENTION_DAYS pruned at
#      startup (default 14 -- override via config alpr.audit_retention_days).
#   5. result.json now includes per-read details + collection stats.
#   6. Single COOLDOWN_SEC assignment (config with fallback 90) -- R1 assigned
#      it twice.
#   7. datetime.now(timezone.utc) instead of deprecated utcnow().

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2, re, time, json, queue, shutil, threading, traceback
import numpy as np
from pathlib import Path
from urllib.parse import quote
from collections import defaultdict, Counter
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ------------------------- Service config -------------------------
CONFIG = json.loads(Path("config.json").read_text())

MQTT_HOST = CONFIG["mqtt"]["host"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_USER = CONFIG["mqtt"].get("username")
MQTT_PASS = CONFIG["mqtt"].get("password")
TRIGGER_TOPIC = CONFIG["mqtt"]["subscribe_topic"]
RESULT_TOPIC_PREFIX = CONFIG["mqtt"]["result_topic_prefix"]
CAMERAS_FILE = "cameras.json"
MODEL_PATH = "best.pt"
AUDIT_DIR = Path("audit")

NUM_WORKERS = 3     # ~= max concurrent dockings; ~2GB RAM each
QUEUE_MAX = 50

COLLECTION_TIMEOUT = CONFIG["alpr"]["collection_timeout"]
FRAME_SKIP = CONFIG["alpr"]["frame_skip"]
MAX_RAW_SAMPLES = CONFIG["alpr"]["max_raw_samples"]
BEST_SAMPLES = CONFIG["alpr"]["best_samples"]
MIN_PLATE_WIDTH = CONFIG["alpr"]["min_plate_width"]
MIN_PLATE_HEIGHT = CONFIG["alpr"]["min_plate_height"]
CENTER_DISTANCE_LIMIT = CONFIG["alpr"]["center_distance_limit"]
COOLDOWN_SEC = CONFIG["alpr"].get("cooldown_sec", 90)
AUDIT_RETENTION_DAYS = CONFIG["alpr"].get("audit_retention_days", 14)

TARGET_H = 220
PAD = 24
MIN_CONF = 0.35

PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$'),
    re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$'),
]

BH_PATTERN = PLATE_PATTERNS[1]

TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2',
            'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'J': '3'}

TO_ALPHA = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S',
            '6': 'G', '7': 'T', '8': 'B'}

# ------------------------- Debug support -------------------------
DEBUG_SAVE_IMAGES = True

def debug_log(stage, message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{stage}] {message}", flush=True)

def save_debug_image(folder, name, image):
    if image is None:
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / name), image)
    except Exception as e:
        debug_log("IMAGE", f"Failed to save {name}: {e}")

def prune_audit(days=AUDIT_RETENTION_DAYS):
    """Delete audit run folders older than `days`. Keeps disk bounded."""
    if not AUDIT_DIR.exists() or days <= 0:
        return
    cutoff = time.time() - days * 86400
    removed = 0
    for cam_dir in AUDIT_DIR.iterdir():
        if not cam_dir.is_dir():
            continue
        for run_dir in cam_dir.iterdir():
            try:
                if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(run_dir)
                    removed += 1
            except Exception as e:
                debug_log("SETUP", f"prune failed for {run_dir}: {e}")
    if removed:
        debug_log("SETUP", f"pruned {removed} audit folders older than {days}d")

# ------------------------- Event / RTSP / cameras -------------------------

# Map incoming MQTT message to a camera (bay)
def extract_event(topic, payload):
    parts = topic.split('/')
    if len(parts) < 6:
        return None
    if parts[4] != 'LineDetector':
        return None
    if parts[5] != 'Crossed':
        return None
    bay = parts[1]
    try:
        data = json.loads(payload.decode())
        event_time = data.get("UtcTime")
    except Exception:
        event_time = None
    return {"bay": bay, "event_time": event_time}


def build_rtsp(camera_cfg, config):
    rtsp = config["rtsp"]
    # URL-encode credentials so '@', ':', '/', '#' in the password
    # can't corrupt the URL (R1 interpolated them raw).
    user = quote(str(rtsp['username']), safe='')
    pw = quote(str(rtsp['password']), safe='')
    return (
        f"rtsp://{user}:{pw}@"
        f"{rtsp['server']}:{rtsp['port']}/"
        f"{camera_cfg['guid']}/live"
    )


# Camera registry
def load_cameras() -> dict:
    p = Path(CAMERAS_FILE)
    if not p.exists():
        template = {
            "AR-FS": {
                "guid": "01000000001babe00c81f7ff9",
                "ip": "10.69.10.100",
                "roi": [850, 250, 1750, 1800],
                "enabled": True
            }
        }
        p.write_text(json.dumps(template, indent=2))
        debug_log("SETUP", f"Wrote template {CAMERAS_FILE} - fill it in and rerun")
        raise SystemExit(1)
    return json.loads(p.read_text())

# ------------------- Plate text handling (Indian plates) -------------------

def normalize(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_valid(text):
    return any(p.fullmatch(text) for p in PLATE_PATTERNS)

def fix_indian_plate(text):
    text = normalize(text)
    if BH_PATTERN.fullmatch(text):
        return text
    if len(text) < 8:
        return text
    c = list(text)
    c[0] = TO_ALPHA.get(c[0], c[0]); c[1] = TO_ALPHA.get(c[1], c[1])
    if len(c) >= 4:
        c[2] = TO_DIGIT.get(c[2], c[2]); c[3] = TO_DIGIT.get(c[3], c[3])
    for i in range(max(0, len(c) - 4), len(c)):
        c[i] = TO_DIGIT.get(c[i], c[i])
    for i in range(4, max(4, len(c) - 4)):
        c[i] = TO_ALPHA.get(c[i], c[i])
    fixed = ''.join(c)
    if is_valid(text) and not is_valid(fixed):
        return text
    return fixed


def prep(img):
    scale = TARGET_H / img.shape[0]
    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    edge = np.concatenate([img[0].reshape(-1, 3), img[-1].reshape(-1, 3),
                           img[:, 0].reshape(-1, 3), img[:, -1].reshape(-1, 3)])
    fill = tuple(int(v) for v in np.median(edge, axis=0))
    return cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD,
                              borderType=cv2.BORDER_CONSTANT, value=fill)


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                         cv2.CV_64F).var()

def duplicate(a, b):
    try:
        a = cv2.resize(a, (300, 100)); b = cv2.resize(b, (300, 100))
        return cv2.absdiff(a, b).mean() < 5
    except Exception:
        return False

def weighted_vote(results):
    valid = [r for r in results if r['valid']]
    pool = valid if valid else results
    mode_len = Counter(len(r['plate']) for r in pool).most_common(1)[0][0]
    pool = [r for r in pool if len(r['plate']) == mode_len]
    votes = defaultdict(dict)
    for r in pool:
        for i, ch in enumerate(r['plate']):
            votes[i][ch] = votes[i].get(ch, 0) + r['conf']
    out = ''
    for i in range(mode_len):
        if i in votes:
            out += max(votes[i], key=votes[i].get)
    if not is_valid(out) and valid:
        out = max(valid, key=lambda r: r['conf'])['plate']
    return out

# ------------------- Worker class (owns models) -------------------

class Worker(threading.Thread):
    def __init__(self, wid, jobs: "queue.Queue", cameras: dict, publish_fn):
        super().__init__(daemon=True, name=f"worker-{wid}")
        self.wid = wid
        self.jobs = jobs
        self.cameras = cameras
        self.publish = publish_fn
        self.model = None
        self.ocr = None

    def log(self, stage, msg):
        debug_log(stage, f"[w{self.wid}] {msg}")

    def run(self):
        self.log("SETUP", "loading models...")
        t = time.time()
        self.model = YOLO(MODEL_PATH)
        self.ocr = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)
        self.log("SETUP", f"ready ({round(time.time() - t, 1)}s)")

        while True:
            job = self.jobs.get()
            bay = job['bay']
            try:
                self.handle(job)
            except Exception:
                self.log("ERROR", f"({bay}) job failed:\n{traceback.format_exc()}")
            finally:
                ACTIVE.discard(bay)
                self.jobs.task_done()

    # --------- per-job pipeline ---------
    def handle(self, job):
        bay = job['bay']
        event_time = job['event_time']
        cam = self.cameras[bay]
        if not cam.get('enabled', True):
            self.log("QUEUE", f"({bay}) camera disabled, skipping")
            return

        t0 = time.time()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = AUDIT_DIR / bay / ts
        debug_dir = folder / "debug"
        folder.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(exist_ok=True)
        self.log("JOB", f"({bay}) started, audit -> {folder}")

        samples, cstats = self.collect(cam, debug_dir)
        self.log("COLLECT",
                 f"({bay}) frames={cstats['frames_read']} "
                 f"evaluated={cstats['frames_evaluated']} "
                 f"raw_cands={cstats['raw_candidates']} "
                 f"kept={len(samples)} "
                 f"rejected={cstats['rejected']} "
                 f"in {cstats['collect_sec']}s "
                 f"(rtsp open {cstats['rtsp_open_ms']}ms)")

        reads = []
        for i, s in enumerate(samples, 1):
            cv2.imwrite(str(folder / f"crop_{i:02d}.jpg"), s['crop'])
            prep_path = (debug_dir / f"prep_{i:02d}.jpg"
                         if DEBUG_SAVE_IMAGES else None)
            plate, conf, valid, raw = self.ocr_image(s['crop'], prep_path)
            self.log("OCR",
                     f"({bay}) sample {i}: raw='{raw}' fixed='{plate}' "
                     f"conf={conf:.2f} valid={valid} score={s['score']:.2f}")
            if plate:
                reads.append({'plate': plate, 'conf': round(conf, 3),
                              'valid': valid, 'raw': raw, 'sample': i})

        final = weighted_vote(reads) if reads else None
        confidence = 0.0
        supporting_reads = 0
        if final:
            matches = [r['conf'] for r in reads if r['plate'] == final]
            if matches:
                confidence = round(sum(matches) / len(matches), 3)
                supporting_reads = len(matches)
        self.log("VOTE", f"({bay}) {len(reads)} reads -> "
                         f"'{final}' (support {supporting_reads}, "
                         f"conf {confidence})")
        elapsed = round(time.time() - t0, 1)

        result = {
            'bay': bay,
            'truck_number': final,
            'confidence': confidence,
            'supporting_reads': supporting_reads,
            'total_reads': len(reads),
            'valid': bool(final and is_valid(final)),
            'event_time': event_time,
            'ocr_time': datetime.now(timezone.utc).isoformat(),
            'processing_time_sec': elapsed,
            'status': 'SUCCESS' if final else 'NO_VALID_PLATE',
            'reads': reads,
            'collection': cstats,
            'audit_folder': str(folder),
        }

        (folder / 'result.json').write_text(
            json.dumps(result, indent=2, default=str))
        (folder / 'event.json').write_text(
            json.dumps(job, indent=2, default=str))
        topic = f"{RESULT_TOPIC_PREFIX}/{bay}"
        self.publish(topic, json.dumps(result, default=str))
        self.log("RESULT", f"({bay}) {final or 'NO PLATE'} "
                           f"published to {topic} "
                           f"({elapsed}s total, {len(reads)} reads)")

    def collect(self, cam: dict, debug_dir=None):
        x1r, y1r, x2r, y2r = cam['roi']
        rtsp_url = build_rtsp(cam, CONFIG)

        t_open = time.time()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        open_ms = round((time.time() - t_open) * 1000)
        stats = {'rtsp_open_ms': open_ms, 'frames_read': 0,
                 'frames_evaluated': 0, 'raw_candidates': 0,
                 'rejected': {}, 'collect_sec': 0.0}

        if not cap.isOpened():
            self.log("RTSP", f"FAILED to open stream "
                             f"({cam.get('ip', '?')}) after {open_ms}ms")
            stats['error'] = 'rtsp_open_failed'
            return [], stats
        self.log("RTSP", f"stream open in {open_ms}ms ({cam.get('ip', '?')})")

        cands = []
        rejected = Counter()
        frame_no = 0
        first_saved = False
        annotated = None
        start = time.time()

        while time.time() - start < COLLECTION_TIMEOUT:
            ok, frame = cap.read()
            if not ok:
                rejected['read_fail'] += 1
                continue
            stats['frames_read'] += 1
            frame_no += 1
            if frame_no % FRAME_SKIP:
                continue
            stats['frames_evaluated'] += 1

            if DEBUG_SAVE_IMAGES and debug_dir and not first_saved:
                save_debug_image(debug_dir, "00_first_frame.jpg", frame)
                first_saved = True

            roi = frame[y1r:y2r, x1r:x2r]
            if roi.size == 0:
                rejected['empty_roi'] += 1
                continue
            center = roi.shape[1] / 2
            results = self.model(roi, verbose=False)
            for r in results:
                for b in r.boxes:
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                    w = bx2 - bx1; h = by2 - by1
                    if w < MIN_PLATE_WIDTH or h < MIN_PLATE_HEIGHT:
                        rejected['too_small'] += 1
                        continue
                    if ((by1 + by2) / 2) < roi.shape[0] * 0.45:
                        rejected['upper_half'] += 1
                        continue
                    dist = abs((bx1 + bx2) / 2 - center)
                    if dist > CENTER_DISTANCE_LIMIT:
                        rejected['off_center'] += 1
                        continue
                    crop = roi[by1:by2, bx1:bx2]
                    if any(duplicate(crop, c['crop']) for c in cands):
                        rejected['duplicate'] += 1
                        continue
                    area = min(crop.shape[0] * crop.shape[1] / 25000, 1.0)
                    shp = min(sharpness(crop) / 600, 1.0)
                    sc = (float(b.conf[0]) * 0.4 + area * 0.25 +
                          shp * 0.2 + (1 - dist / center) * 0.15)
                    cands.append({'crop': crop, 'score': sc})
                    if DEBUG_SAVE_IMAGES and debug_dir:
                        if annotated is None:
                            annotated = roi.copy()
                        cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                                      (0, 255, 0), 2)
                        cv2.putText(annotated, f"{sc:.2f}", (bx1, by1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                    if len(cands) >= MAX_RAW_SAMPLES:
                        break
                if len(cands) >= MAX_RAW_SAMPLES:
                    break
            if len(cands) >= MAX_RAW_SAMPLES:
                break

        cap.release()
        if DEBUG_SAVE_IMAGES and debug_dir and annotated is not None:
            save_debug_image(debug_dir, "01_annotated_roi.jpg", annotated)

        stats['raw_candidates'] = len(cands)
        stats['rejected'] = dict(rejected)
        stats['collect_sec'] = round(time.time() - start, 1)
        cands.sort(key=lambda x: x['score'], reverse=True)
        return cands[:BEST_SAMPLES], stats

    def ocr_image(self, img, debug_path=None):
        pimg = prep(img)
        if debug_path is not None:
            save_debug_image(debug_path.parent, debug_path.name, pimg)
        result = self.ocr.ocr(pimg, cls=False)
        if not result or result[0] is None:
            return '', 0.0, False, ''
        page = result[0]
        parts = []; confs = []
        for item in page:
            try:
                box, (txt, conf) = item
            except (TypeError, ValueError):
                continue
            if conf < MIN_CONF:
                continue
            parts.append(txt); confs.append(conf)
        if not parts:
            return '', 0.0, False, ''
        raw = ''.join(parts)
        txt = fix_indian_plate(raw)
        return txt, float(np.mean(confs)), is_valid(txt), raw

# ------------------- MQTT plumbing + debounce -------------------

JOBS: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAX)
ACTIVE: set = set()      # camera_ids queued/in-progress
LAST_FIRED: dict = {}    # camera_id -> last accepted time
_state_lock = threading.Lock()

def try_enqueue(event: dict) -> bool:
    bay = event['bay']; now = time.time()
    with _state_lock:
        if bay in ACTIVE:
            debug_log("QUEUE", f"({bay}) rejected: job already active")
            return False
        remaining = COOLDOWN_SEC - (now - LAST_FIRED.get(bay, 0))
        if remaining > 0:
            debug_log("QUEUE",
                      f"({bay}) rejected: cooldown, {remaining:.0f}s left")
            return False
        try:
            JOBS.put_nowait(event)
        except queue.Full:
            debug_log("QUEUE",
                      f"({bay}) rejected: job queue FULL "
                      f"({JOBS.qsize()}/{QUEUE_MAX}), dropping event")
            return False
        ACTIVE.add(bay); LAST_FIRED[bay] = now
        debug_log("QUEUE",
                  f"({bay}) accepted (queue depth {JOBS.qsize()}, "
                  f"active {sorted(ACTIVE)})")
        return True

def build_mqtt(cameras: dict) -> mqtt.Client:
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):
        client = mqtt.Client()

    def on_connect(client, userdata, flags, rc, *args):
        debug_log("MQTT", f"connected rc={rc}, subscribing {TRIGGER_TOPIC}")
        client.subscribe(TRIGGER_TOPIC, qos=1)

    def on_disconnect(client, userdata, *args):
        debug_log("MQTT", "disconnected -- paho will auto-reconnect")

    def on_message(client, userdata, msg):
        event = extract_event(msg.topic, msg.payload)
        if event is None:
            return
        bay = event["bay"]
        if bay not in cameras:
            debug_log("MQTT", f"[warn] event for unknown bay "
                              f"'{bay}' (topic {msg.topic})")
            return
        if not cameras[bay].get("enabled", True):
            return
        if try_enqueue(event):
            debug_log("MQTT", f"({bay}) event queued for processing")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    return client

# ------------------------- Main -------------------------

def main():
    debug_log("SETUP", "=" * 60)
    debug_log("SETUP", "05_MQTT_ALPR_Service_R2 starting")
    debug_log("SETUP", f"mqtt={MQTT_HOST}:{MQTT_PORT} "
                       f"trigger='{TRIGGER_TOPIC}' "
                       f"results='{RESULT_TOPIC_PREFIX}/<bay>'")
    debug_log("SETUP", f"workers={NUM_WORKERS} queue_max={QUEUE_MAX} "
                       f"cooldown={COOLDOWN_SEC}s "
                       f"collect_timeout={COLLECTION_TIMEOUT}s "
                       f"frame_skip={FRAME_SKIP}")
    debug_log("SETUP", f"samples: raw<={MAX_RAW_SAMPLES} best={BEST_SAMPLES} "
                       f"min_plate={MIN_PLATE_WIDTH}x{MIN_PLATE_HEIGHT} "
                       f"center_limit={CENTER_DISTANCE_LIMIT}")
    debug_log("SETUP", f"debug_images={DEBUG_SAVE_IMAGES} "
                       f"audit_retention={AUDIT_RETENTION_DAYS}d")

    cameras = load_cameras()
    enabled = [b for b, c in cameras.items() if c.get('enabled', True)]
    debug_log("SETUP", f"Loaded {len(cameras)} cameras from {CAMERAS_FILE} "
                       f"({len(enabled)} enabled: {enabled})")

    AUDIT_DIR.mkdir(exist_ok=True)
    prune_audit()

    client = build_mqtt(cameras)

    def publish(topic, payload):
        client.publish(topic, payload, qos=1)

    workers = [
        Worker(i + 1, JOBS, cameras, publish)
        for i in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()

    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    debug_log("SETUP", "Service running. Ctrl+C to stop.")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        debug_log("SETUP", "Ctrl+C received, shutting down")


if __name__ == "__main__":
    main()
