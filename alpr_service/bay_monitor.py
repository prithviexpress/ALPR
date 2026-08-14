"""Continuous, trigger-independent bay-activity monitor.

Separate concern from the enter/leave ALPR pipeline (worker.py / JobBus /
MQTT-or-HTTP triggers) -- this never touches any of that. It only reuses
cameras.json (via the same cameras dict service.py already loaded) and the
HTTP snapshot fetch mechanism (snapshot.py) that ALPR's own collect() uses.

How it works: round-robins every enabled camera, one frame at a time,
"by turns". A bay with nothing detected is scanned at the baseline cadence
(bay_monitor.baseline_scan_interval_ms between each bay checked). Presence
is a cheap signal -- reusing the existing YOLO plate model exactly as
loaded for ALPR, but with no filtering at all (unlike worker.py's
size/position/duplicate checks): here ANY detected box, however small,
weak or off-center, counts as "something's there", since the point isn't
to find a clean plate -- it's just "is this bay worth paying closer
attention to".

Once a bay has any detection it's "zoomed in": every
bay_monitor.classify_interval_sec (default 60s -- "one frame per minute
per truck") a fresh frame is sent to a local Ollama-hosted vision model
(bay_monitor.ollama_model) with a classification prompt, and the reply is
published to mqtt.bay_status_topic_prefix + "/<bay>". Other bays are not
paused while one is zoomed in -- the round-robin keeps visiting all of
them each pass; a zoomed-in bay just also gets the extra classification
call layered on top when its interval is due, and is skipped without a
frame fetch on passes where it isn't. After bay_monitor.empty_debounce_count
consecutive "empty" classifications the bay reverts to baseline YOLO-only
scanning, since an LLM call every round for a bay with nothing happening
is wasted latency and load.
"""
import base64
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests

from .logging_setup import get_logger
from .snapshot import build_snapshot_url, build_auth, fetch_snapshot, SnapshotError


class BayState:
    def __init__(self):
        self.zoomed_in = False
        self.last_classify_time = 0.0
        self.consecutive_empty = 0


def load_reference_images(bm_cfg: dict, config_dir: Path, log) -> list:
    """Loads and JPEG-encodes each configured exemplar image once at
    startup (not per-classification-call -- these are static, re-reading
    and re-encoding them on every scan would be pure waste). A missing or
    unreadable file is logged and skipped rather than failing startup --
    a typo'd reference path shouldn't take down the whole monitor, it
    just means classification runs without that one example's help."""
    refs = []
    for entry in bm_cfg.get("reference_images", []):
        path = config_dir / entry["path"]
        img = cv2.imread(str(path))
        if img is None:
            log.warning(f"reference image not found/unreadable: {path} "
                        f"(label={entry.get('label')!r}), skipping")
            continue
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            log.warning(f"failed to encode reference image {path}, skipping")
            continue
        refs.append((entry["label"], base64.b64encode(buf.tobytes()).decode("ascii")))
    if refs:
        log.info(f"loaded {len(refs)} reference image(s) as few-shot "
                  f"classification context: {[l for l, _ in refs]}")
    return refs


def classify_frame(frame, bm_cfg: dict, log, bay: str, reference_images: list = None):
    """POSTs one JPEG-encoded frame -- plus, if configured, a set of
    labeled exemplar images sent alongside it as few-shot context -- to a
    local Ollama vision model, and returns whichever of
    bm_cfg['status_values'] appears in its reply. Returns None if the
    request/response didn't produce a usable answer (logged, never
    raised -- a flaky local LLM call shouldn't take the scanner thread
    down).

    Few-shot exemplars matter most for exactly the ambiguous cases a bare
    prompt gets wrong -- e.g. a truck with its bay/cargo door open, which
    can look enough like "empty" or confuse plate detection that showing
    the model a labeled example of that specific situation calibrates it
    far better than describing it in words alone."""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        log.warning(f"({bay}) failed to JPEG-encode frame for classification")
        return None
    target_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    reference_images = reference_images or []
    images = [b64 for _, b64 in reference_images] + [target_b64]

    if reference_images:
        example_lines = "\n".join(
            f"Image {i + 1} is a labeled example of: {label}."
            for i, (label, _) in enumerate(reference_images))
        prompt = (
            f"{bm_cfg['classification_prompt']}\n\n"
            f"For calibration, here are labeled example images:\n"
            f"{example_lines}\n"
            f"Image {len(reference_images) + 1} (the LAST image) is the "
            f"one you must classify now. The earlier images are only "
            f"reference examples, not the subject of your answer."
        )
    else:
        prompt = bm_cfg["classification_prompt"]

    url = f"{bm_cfg['ollama_host'].rstrip('/')}/api/generate"
    payload = {
        "model": bm_cfg["ollama_model"],
        "prompt": prompt,
        "images": images,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=bm_cfg["ollama_timeout_sec"])
        resp.raise_for_status()
        text = resp.json().get("response", "").strip().lower()
    except (requests.RequestException, ValueError) as e:
        log.warning(f"({bay}) ollama classification call failed: {e}")
        return None

    for status in bm_cfg["status_values"]:
        if status in text:
            return status
    log.warning(f"({bay}) ollama response didn't match any of "
                f"{bm_cfg['status_values']}: {text!r}")
    return None


def run_bay_monitor(cameras: dict, config: dict, publish_fn, stop_event: threading.Event):
    log = get_logger("BAY_MONITOR")
    bm_cfg = config["bay_monitor"]
    snap_cfg = config["snapshot"]
    bay_status_topic_prefix = config["mqtt"]["bay_status_topic_prefix"]

    from ultralytics import YOLO
    config_dir = Path(config["_config_dir"])
    model_path = config_dir / config["model_path"]
    model = YOLO(str(model_path))
    reference_images = load_reference_images(bm_cfg, config_dir, log)
    log.info(f"bay monitor started: model={model_path} "
              f"baseline_interval={bm_cfg['baseline_scan_interval_ms']}ms "
              f"classify_interval={bm_cfg['classify_interval_sec']}s "
              f"ollama={bm_cfg['ollama_host']} model={bm_cfg['ollama_model']}")

    session = requests.Session()
    auth = build_auth(config)
    states = {}

    while not stop_event.is_set():
        for bay, cam in cameras.items():
            if stop_event.is_set():
                break
            if not cam.get("enabled", True):
                continue
            state = states.setdefault(bay, BayState())

            if state.zoomed_in and (time.time() - state.last_classify_time
                                     < bm_cfg["classify_interval_sec"]):
                # Not due yet -- skip without fetching a frame, but still
                # pace the loop. Without this wait, a round consisting
                # entirely of zoomed-in-but-not-due bays would busy-spin
                # the thread at ~100% CPU checking timestamps in a tight
                # loop instead of idling until there's real work to do.
                if stop_event.wait(bm_cfg["baseline_scan_interval_ms"] / 1000):
                    break
                continue

            try:
                url = build_snapshot_url(cam, config)
                frame, fetch_ms, size_bytes = fetch_snapshot(
                    session, url, auth,
                    snap_cfg["connect_timeout_ms"], snap_cfg["read_timeout_ms"])
            except SnapshotError as e:
                log.debug(f"({bay}) snapshot fetch failed: {e}")
                frame = None

            if frame is not None:
                if not state.zoomed_in:
                    x1, y1, x2, y2 = cam["roi"]
                    roi = frame[y1:y2, x1:x2]
                    results = model(roi, verbose=False)
                    if any(len(r.boxes) > 0 for r in results):
                        state.zoomed_in = True
                        state.consecutive_empty = 0
                        log.info(f"({bay}) presence detected, zooming in")

                if state.zoomed_in:
                    status = classify_frame(frame, bm_cfg, log, bay, reference_images)
                    state.last_classify_time = time.time()
                    if status is not None:
                        payload = {
                            "bay": bay,
                            "status": status,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        topic = f"{bay_status_topic_prefix}/{bay}"
                        publish_fn(topic, json.dumps(payload))
                        log.info(f"({bay}) status={status} -> {topic}")

                        if status == "empty":
                            state.consecutive_empty += 1
                            if state.consecutive_empty >= bm_cfg["empty_debounce_count"]:
                                state.zoomed_in = False
                                log.info(f"({bay}) {state.consecutive_empty} "
                                          f"consecutive 'empty' reads, reverting "
                                          f"to baseline scan")
                        else:
                            state.consecutive_empty = 0

            # Pace between bays regardless of what happened above -- keeps
            # the round-robin from hammering every camera back-to-back
            # with no gap. wait() returns True the moment stop_event is
            # set, letting shutdown interrupt the pause immediately
            # instead of waiting out the full interval.
            if stop_event.wait(bm_cfg["baseline_scan_interval_ms"] / 1000):
                break

    log.info("bay monitor stopped")


def start_bay_monitor(cameras: dict, config: dict, publish_fn):
    """Starts the round-robin scanner in a background thread. Returns
    (thread, stop_event) -- signal stop_event to ask the loop to exit at
    its next check point (it won't interrupt a request already in
    flight)."""
    stop_event = threading.Event()
    t = threading.Thread(target=run_bay_monitor,
                          args=(cameras, config, publish_fn, stop_event),
                          daemon=True, name="bay-monitor")
    t.start()
    return t, stop_event
