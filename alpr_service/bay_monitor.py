"""Continuous, trigger-independent bay-activity monitor.

Separate concern from the enter/leave ALPR pipeline (worker.py / JobBus /
MQTT-or-HTTP triggers) -- this never touches any of that. It only reuses
cameras.json (via the same cameras dict service.py already loaded) and the
HTTP snapshot fetch mechanism (snapshot.py) that ALPR's own
collect_and_read() uses.

How it works: round-robins every enabled camera, one frame at a time,
"by turns". A bay with nothing detected is scanned at the baseline cadence
(bay_monitor.baseline_scan_interval_ms between each bay checked). Presence
is a cheap signal -- a plain thumbnail diff of the ROI against the last
time this bay was looked at while empty, with no model involved at all.
Deliberately NOT the ALPR plate model: a bay is occupied or not regardless
of whether a plate happens to be visible in frame (a reversed-in trailer's
plate usually faces away from the dock camera entirely), so gating on
plate detection left the monitor blind to exactly the trucks it exists to
track. Any meaningful visual change -- a truck, a forklift, a person, an
open door -- counts as "something's there", since the point isn't to find
a clean plate, or even identify what changed -- it's just "is this bay
worth paying closer attention to".

Once a bay's scene has changed it's "zoomed in": a frame is sent to a
local Ollama-hosted vision model (bay_monitor.ollama_model) with a
classification prompt, and the reply is published to
mqtt.bay_status_topic_prefix + "/<bay>". Two independent things can
trigger that call while zoomed in -- whichever comes first: the
bay_monitor.classify_interval_sec timer (default 60s -- "at least one
frame per minute per truck"), or the same kind of thumbnail diff used
for presence detection noticing the scene changed since the last
classify (bay_monitor.classify_diff_*) -- e.g. a truck going from idle
to actively being unloaded gets picked up immediately rather than
waiting out the rest of the interval. A zoomed-in bay is fetched every
round either way (same cadence as baseline scanning), just not always
classified. Other bays are not paused while one is zoomed in -- the
round-robin keeps visiting all of them each pass. After
bay_monitor.empty_debounce_count consecutive "empty" classifications the
bay reverts to baseline diff-only scanning, since an LLM call every
round for a bay with nothing happening is wasted latency and load.
"""
import base64
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests

from .image_ops import thumbnail, duplicate_thumbs
from .logging_setup import get_logger
from .snapshot import build_snapshot_url, build_auth, fetch_snapshot, SnapshotError


class BayState:
    def __init__(self):
        self.zoomed_in = False
        self.last_classify_time = 0.0
        self.consecutive_empty = 0
        # Baseline presence check (see BayMonitor._detect_presence): a
        # thumbnail of this ROI the last time it was looked at while
        # empty, so the next look can tell whether the scene changed.
        self.presence_thumb = None
        # Same idea while already zoomed in (see BayMonitor._detect_
        # change): a thumbnail of this ROI as of the last classified (or
        # looked-at) frame, so an in-progress visit's activity change
        # (e.g. idle -> unloading) can trigger an early reclassify
        # instead of waiting out classify_interval_sec.
        self.classify_thumb = None
        # Whether this bay has EVER been successfully classified since
        # the process started. The presence-diff check alone can never
        # trigger the very first classify for a bay: there is nothing
        # yet to diff the current frame against, so a truck already
        # parked when the service starts would otherwise stay invisible
        # -- no Ollama call, no bay_state_engine session, nothing --
        # until some LATER visual change happens to trip the diff. See
        # BayMonitor._scan_bay's "first look" handling, which forces
        # exactly one classify per bay regardless of the diff, to
        # establish real ground truth on startup.
        self.classified_once = False
        # Last time this bay's periodic image heartbeat (bay_monitor.
        # snapshot_publish_interval_sec) was published -- independent of
        # classification entirely, so it keeps firing even for a bay
        # that's sitting at baseline with nothing happening.
        self.last_snapshot_publish_time = 0.0
        # This bay's most recent successful classification word (e.g.
        # "loading", "empty") -- so the periodic snapshot heartbeat can
        # report current occupancy/activity alongside the image, without
        # forcing a fresh classify call just to answer "what do we think
        # is happening right now". None until the first classify ever
        # succeeds.
        self.last_status = None


def downscale(img, max_dimension: int):
    """Shrink so the longest side is at most max_dimension, preserving
    aspect. A vision model's prefill cost scales with pixel count, and a
    raw 5MP snapshot is ~6x the pixels of a 1024-wide view -- on a
    CPU-only box that difference routinely exceeds ollama_timeout_sec,
    and a timeout throws away the fetch, the encode and the partial
    inference alike, leaving the bay with no status at all. Returns the
    image untouched if it's already small enough or if max_dimension is
    falsy (downscaling disabled)."""
    if not max_dimension:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return img
    scale = max_dimension / longest
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)


def encode_image(img, max_dimension: int, jpeg_quality: int):
    """Downscale -> JPEG -> base64, the form Ollama wants."""
    ok, buf = cv2.imencode(".jpg", downscale(img, max_dimension),
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def load_reference_images(bm_cfg: dict, config_dir: Path, log) -> list:
    """Loads, downscales and JPEG-encodes each configured exemplar image
    once at startup (not per-classification-call -- these are static, so
    re-reading and re-encoding them on every scan would be pure waste,
    and at full source resolution each one would multiply every
    classification's prefill bill). A missing or unreadable file is
    logged and skipped rather than failing startup -- a typo'd reference
    path shouldn't take down the whole monitor, it just means
    classification runs without that one example's help."""
    refs = []
    max_dim = bm_cfg["classify_max_dimension"]
    quality = bm_cfg["classify_jpeg_quality"]
    for entry in bm_cfg["reference_images"]:
        # A malformed entry (not a dict, or missing "path"/"label") is
        # skipped with a warning like a missing file is -- a config typo
        # here must not raise out of startup and kill the monitor thread.
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("label"):
            log.warning(f"reference_images entry is not a "
                        f"{{'path': ..., 'label': ...}} object: {entry!r}, skipping")
            continue
        path = config_dir / entry["path"]
        img = cv2.imread(str(path))
        if img is None:
            log.warning(f"reference image not found/unreadable: {path} "
                        f"(label={entry['label']!r}), skipping")
            continue
        b64 = encode_image(img, max_dim, quality)
        if b64 is None:
            log.warning(f"failed to encode reference image {path}, skipping")
            continue
        refs.append((entry["label"], b64))
    if refs:
        log.info(f"loaded {len(refs)} reference image(s) as few-shot "
                  f"classification context: {[l for l, _ in refs]}")
    return refs


def classify_frame(frame, bm_cfg: dict, log, bay: str, reference_images: list = None,
                    session: requests.Session = None):
    """POSTs one JPEG-encoded frame -- plus, if configured, a set of
    labeled exemplar images sent alongside it as few-shot context -- to a
    local Ollama vision model, and returns (status, comment, image_b64):
    status is whichever of bm_cfg['status_values'] appears in the reply
    (or None if the reply didn't produce a usable answer -- logged, never
    raised, since a flaky local LLM call shouldn't take the scanner
    thread down), comment is the model's free-text description of what
    it sees (see split_status_comment -- falls back to the whole reply
    if the model didn't follow the STATUS:/COMMENT: format), and
    image_b64 is the exact JPEG-base64 this call sent to Ollama, handed
    back so a caller that wants to publish/forward the classified frame
    doesn't have to re-encode it.

    Few-shot exemplars matter most for exactly the ambiguous cases a bare
    prompt gets wrong -- e.g. a truck with its bay/cargo door open, which
    can look enough like "empty" or confuse plate detection that showing
    the model a labeled example of that specific situation calibrates it
    far better than describing it in words alone."""
    target_b64 = encode_image(frame, bm_cfg["classify_max_dimension"],
                               bm_cfg["classify_jpeg_quality"])
    if target_b64 is None:
        log.warning(f"({bay}) failed to JPEG-encode frame for classification")
        return None, None, None

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
        "think": bm_cfg["ollama_think"],
    }
    requester = session if session is not None else requests
    try:
        resp = requester.post(url, json=payload, timeout=bm_cfg["ollama_timeout_sec"])
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except (requests.RequestException, ValueError) as e:
        log.warning(f"({bay}) ollama classification call failed: {e}")
        return None, None, None

    status_text, comment = split_status_comment(text)
    status = parse_status(status_text, bm_cfg["status_values"], log, bay)
    return status, comment, target_b64


def split_status_comment(text: str):
    """Splits a 'STATUS: <word>\\nCOMMENT: <text>' reply (the format
    bay_monitor.classification_prompt's default asks for) into the two
    parts, so a comment mentioning a DIFFERENT status word than the one
    actually chosen (e.g. "STATUS: idle / COMMENT: looks mostly empty of
    cargo") can't make parse_status() see two candidate words in the same
    blob of text and call the whole reply ambiguous.

    Falls back to using the full reply as both parts if either marker is
    missing -- keeps this working with a model that ignores the
    requested format, or a custom classification_prompt that doesn't ask
    for one at all (the pre-existing single-word-only behavior)."""
    status_part = text
    comment_part = text
    m = re.search(r'status\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        status_part = m.group(1).strip()
    m2 = re.search(r'comment\s*:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if m2:
        comment_part = m2.group(1).strip()
    return status_part, comment_part


def parse_status(text: str, status_values, log, bay: str):
    """Resolve a model's free-text reply to exactly one of status_values.

    A plain substring scan in list order is wrong here: a reply like
    "Not empty - a truck is unloading." contains "empty" before it
    contains "unloading", so it would resolve to the exact opposite of
    what the model actually said -- and three of those in a row would
    fire a premature departure. So:
      1. An exact match (the prompt asks for one bare word) wins outright.
      2. Otherwise match only on whole words, and require exactly one
         distinct status to appear. Two or more means the reply is
         genuinely ambiguous to us, so we return None rather than guess;
         None is already handled everywhere as "no state change this
         round", which is the safe outcome.
    """
    text = (text or "").strip().lower()
    if not text:
        return None
    for status in status_values:
        if text == status.lower():
            return status
    matched = [s for s in status_values
               if re.search(rf"\b{re.escape(s.lower())}\b", text)]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        log.warning(f"({bay}) ollama response matched multiple statuses "
                    f"{matched}, too ambiguous to act on: {text!r}")
        return None
    log.warning(f"({bay}) ollama response didn't match any of "
                f"{list(status_values)}: {text!r}")
    return None


class BayMonitor:
    """The round-robin scanner.

    A class rather than functions threading a dozen loop-invariant
    arguments through each call: the model, HTTP session, auth, encoded
    reference images, logger and config slices are all fixed for the
    thread's lifetime, and only (bay, cam, state) vary per turn.
    """

    def __init__(self, cameras: dict, config: dict, publish_fn, on_status=None,
                 audit_dir: Path = None):
        self.log = get_logger("BAY_MONITOR")
        self.cameras = cameras
        self.config = config
        self.publish = publish_fn
        self.on_status = on_status
        self.cfg = config["bay_monitor"]
        self.snap_cfg = config["snapshot"]
        self.status_topic_prefix = config["mqtt"]["bay_status_topic_prefix"]
        self.snapshot_topic_prefix = config["mqtt"]["bay_snapshot_topic_prefix"]
        self.snapshot_publish_interval_sec = self.cfg["snapshot_publish_interval_sec"]
        self.empty_status = self.cfg["empty_status"]
        self.classify_region = self.cfg["classify_region"]
        self.scan_interval_sec = self.cfg["baseline_scan_interval_ms"] / 1000
        self.presence_diff_resize = (self.cfg["presence_diff_resize_width"],
                                      self.cfg["presence_diff_resize_height"])
        # Where every bay's most-recently-fetched frame is dropped (see
        # _save_latest_frame) -- None (no audit_dir passed, or
        # save_latest_frame turned off) just skips that write.
        self.audit_dir = audit_dir
        self.save_latest_frame = (audit_dir is not None
                                   and self.cfg["save_latest_frame"])
        self.states = {}

        config_dir = Path(config["_config_dir"])
        self.references = load_reference_images(self.cfg, config_dir, self.log)
        self.session = requests.Session()
        # Cameras and the Ollama host are always on the local/internal
        # network. `requests` honors HTTP_PROXY/HTTPS_PROXY env vars and
        # Windows system proxy settings by default (trust_env=True), which
        # on corporate machines can route -- or block -- these internal
        # requests through a proxy that has no route to them, while a
        # plain `curl` in the same shell session bypasses it. Disable
        # proxy use entirely so these calls always go direct.
        self.session.trust_env = False
        self.auth = build_auth(config)
        self.log.info(
            f"bay monitor started: "
            f"baseline_interval={self.cfg['baseline_scan_interval_ms']}ms "
            f"classify_interval={self.cfg['classify_interval_sec']}s "
            f"classify_region={self.classify_region} "
            f"classify_max_dim={self.cfg['classify_max_dimension']} "
            f"presence_diff_enabled={self.cfg['presence_diff_enabled']} "
            f"ollama={self.cfg['ollama_host']} model={self.cfg['ollama_model']}")

    def run(self, stop_event: threading.Event):
        while not stop_event.is_set():
            for bay, cam in self.cameras.items():
                if stop_event.is_set():
                    break
                if not cam.get("enabled", True):
                    continue
                state = self.states.setdefault(bay, BayState())

                # A zoomed-in bay is still fetched every round (not just
                # when classify_interval_sec is due) -- see _scan_bay's
                # "already zoomed in" branch, which reclassifies early on
                # a meaningful scene change (e.g. idle -> unloading)
                # without waiting for the timer, while the timer itself
                # still fires as a periodic backstop.

                # Every bay's turn is wrapped: this loop is the ONLY thing
                # driving bay_state_engine, so an unhandled exception here
                # would kill the thread and silently stop all ALPR while
                # the process still looks healthy. One bad bay must cost
                # one round, not the whole monitor.
                try:
                    self._scan_bay(bay, cam, state)
                except Exception:
                    self.log.error(f"({bay}) scan round failed, continuing "
                                   f"with the next bay", exc_info=True)

                # Pace between bays regardless of what happened above --
                # keeps the round-robin from hammering every camera
                # back-to-back. wait() returns True the moment stop_event
                # is set, so shutdown interrupts the pause immediately.
                if stop_event.wait(self.scan_interval_sec):
                    break

        self.log.info("bay monitor stopped")

    def _fetch(self, bay: str, cam: dict):
        try:
            url = build_snapshot_url(cam, self.config)
            frame, _, _ = fetch_snapshot(
                self.session, url, self.auth,
                self.snap_cfg["connect_timeout_ms"],
                self.snap_cfg["read_timeout_ms"])
            if self.save_latest_frame:
                self._save_latest_frame(bay, frame)
            return frame
        except SnapshotError as e:
            self.log.debug(f"({bay}) snapshot fetch failed: {e}")
            return None

    def _save_latest_frame(self, bay: str, frame):
        """Overwrites audit/<bay>/latest_frame.jpg on every successful
        fetch -- a live "what does this camera currently see" view for
        remote troubleshooting, without wading through diagnostics_mode's
        much larger per-event dumps. One file per bay, always the most
        recent frame, never accumulates."""
        try:
            bay_dir = self.audit_dir / bay
            bay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(bay_dir / "latest_frame.jpg"), frame)
        except Exception:
            # Best-effort diagnostics -- must never take the scan loop
            # down over a disk/permission problem.
            self.log.warning(f"({bay}) failed to save latest_frame.jpg",
                             exc_info=True)

    def _roi(self, bay: str, cam: dict, frame):
        """The configured ROI slice, or None if it falls outside the frame
        (the same guard worker.py's collect() applies -- an empty slice is
        what YOLO raises on)."""
        x1, y1, x2, y2 = cam["roi"]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.log.warning(f"({bay}) roi {cam['roi']} is empty against a "
                             f"{frame.shape[1]}x{frame.shape[0]} frame -- "
                             f"check the roi in cameras.json")
            return None
        return roi

    def _detect_presence(self, bay: str, roi, state: BayState) -> bool:
        """Is the ROI's scene different from the last time this bay was
        looked at while empty -- a truck, forklift, person, or open door
        changing the pixels, whatever it is. Deliberately NOT plate
        detection: this used to reuse the ALPR YOLO model (any box =
        presence), but a plate is frequently not visible at all from a
        dock camera's angle on a reversed-in trailer, which made the
        monitor blind to exactly the trucks it exists to track. Presence
        here has nothing to do with whether a plate can be read -- that's
        the ALPR pipeline's job, entirely separate from "what's happening
        at this bay right now."

        A resize+diff (~100us) is the whole test, run every baseline
        round -- there's no expensive model pass to gate here, so unlike
        the old YOLO-backed version there's nothing to skip or
        periodically re-verify against. presence_diff_enabled=False
        bypasses the check and reports presence unconditionally (useful
        to force classification on every round while tuning
        classify_interval_sec/the prompt)."""
        if not self.cfg["presence_diff_enabled"]:
            return True

        thumb = thumbnail(roi, self.presence_diff_resize)
        if state.presence_thumb is None:
            # No history yet for this bay -- establish the baseline
            # rather than call a single frame "presence" with nothing to
            # compare it against.
            state.presence_thumb = thumb
            return False

        changed = not duplicate_thumbs(thumb, state.presence_thumb,
                                        self.cfg["presence_diff_threshold"])
        state.presence_thumb = thumb
        return changed

    def _detect_change(self, bay: str, roi, state: BayState) -> bool:
        """While already zoomed in: is the ROI meaningfully different
        from the last frame this bay was looked at -- e.g. a truck going
        from idle to actively being unloaded. This runs alongside
        classify_interval_sec's periodic reclassify, not instead of it:
        a change big enough to matter gets picked up immediately rather
        than waiting out the rest of the interval, while a change too
        subtle to move enough pixels still gets caught on the next
        scheduled tick. Uses a separate threshold from the baseline
        presence check (classify_diff_threshold, not
        presence_diff_threshold) since "is anything here at all" and
        "did an established truck's activity change" are different
        questions that don't need to share a sensitivity."""
        if not self.cfg["classify_diff_enabled"]:
            return False

        thumb = thumbnail(roi, self.presence_diff_resize)
        if state.classify_thumb is None:
            state.classify_thumb = thumb
            return False

        changed = not duplicate_thumbs(thumb, state.classify_thumb,
                                        self.cfg["classify_diff_threshold"])
        state.classify_thumb = thumb
        return changed

    def _publish_snapshot(self, bay: str, frame, state: BayState):
        """A base64 image heartbeat on its own timer (bay_monitor.
        snapshot_publish_interval_sec, default 300s), independent of
        whether a classify happens to run this round -- so a downstream
        consumer has a periodic "here is what this bay looks like, and
        what we currently believe is happening" without needing to wait
        for -- or trigger -- a status change. occupancy_status/activity
        report the last classification this bay actually got (state.
        last_status, possibly from several rounds ago if nothing's
        changed since -- not a fresh classify call, which would defeat
        the point of this being a CHEAP heartbeat), or None if this bay
        has never been classified yet. 0 disables it."""
        if not self.snapshot_publish_interval_sec:
            return
        if (time.time() - state.last_snapshot_publish_time
                < self.snapshot_publish_interval_sec):
            return
        image_b64 = encode_image(frame, self.cfg["classify_max_dimension"],
                                  self.cfg["classify_jpeg_quality"])
        if image_b64 is None:
            return
        state.last_snapshot_publish_time = time.time()
        occupancy_status = None
        if state.last_status is not None:
            occupancy_status = ("empty" if state.last_status == self.empty_status
                                 else "occupied")
        topic = f"{self.snapshot_topic_prefix}/{bay}"
        self.publish(topic, json.dumps({
            "bay": bay,
            "occupancy_status": occupancy_status,
            "activity": state.last_status,
            "image_base64": image_b64,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        self.log.debug(f"({bay}) periodic snapshot published -> {topic}")

    def _scan_bay(self, bay: str, cam: dict, state: BayState):
        """One bay's turn: fetch a frame, then either check for presence
        (baseline) or classify activity (zoomed in)."""
        frame = self._fetch(bay, cam)
        if frame is None:
            return
        self._publish_snapshot(bay, frame, state)
        roi = self._roi(bay, cam, frame)

        if not state.zoomed_in:
            if roi is None:
                return
            presence = self._detect_presence(bay, roi, state)
            # The diff alone can never trigger a bay's very FIRST-ever
            # classify: there's nothing yet to diff the current frame
            # against, so a truck already parked when the service starts
            # would otherwise never get looked at until some LATER visual
            # change happens to trip the diff -- no Ollama call, no
            # bay_state_engine session, nothing, indefinitely. Force
            # exactly one classify per bay regardless of presence to
            # establish real ground truth, whatever it turns out to be.
            first_look = not state.classified_once
            if not (presence or first_look):
                return
            self.log.info(
                f"({bay}) presence detected, zooming in" if presence else
                f"({bay}) first look at this bay -- classifying once to "
                f"establish its current state")
        else:
            due = (time.time() - state.last_classify_time
                   >= self.cfg["classify_interval_sec"])
            changed = roi is not None and self._detect_change(bay, roi, state)
            if not (due or changed):
                return
            if changed and not due:
                self.log.info(f"({bay}) scene changed, reclassifying early")

        # Which pixels the vision model judges. "full_frame" (default)
        # shows it the whole bay -- cargo, forklifts, doors -- which is
        # what the activity vocabulary is about, but a truck in an
        # ADJACENT bay that's in shot can then drive this bay's status.
        # "roi" removes that cross-talk at the cost of a narrower view.
        classify_img = roi if (self.classify_region == "roi" and roi is not None) else frame
        status, comment, image_b64 = classify_frame(
            classify_img, self.cfg, self.log, bay,
            self.references, session=self.session)
        state.last_classify_time = time.time()
        if status is None:
            return
        state.classified_once = True
        state.last_status = status

        timestamp = datetime.now(timezone.utc).isoformat()
        topic = f"{self.status_topic_prefix}/{bay}"
        self.publish(topic, json.dumps(
            {"bay": bay, "status": status, "timestamp": timestamp}))
        self.log.info(f"({bay}) status={status} -> {topic}")

        # Whether this classify counts as an arrival (or a departure) is
        # decided HERE, from the actual answer, rather than earlier from
        # the diff that triggered the call -- a presence-diff false
        # alarm (e.g. a lighting change) that classify then reads as
        # "empty" must not park the bay in zoomed-in mode waiting out a
        # debounce it never really entered.
        occupied = status != self.empty_status
        departed = False
        if occupied:
            if not state.zoomed_in:
                state.zoomed_in = True
                # The cached thumbnails describe the pre-arrival scene
                # and are meaningless once this bay reverts to baseline
                # again (whether "unchanged" then means "still that
                # truck" or "back to empty" is exactly what a stale
                # cache can't tell) -- clear them so the first baseline
                # check after this visit establishes a fresh empty
                # baseline rather than diffing against a frame from
                # before the truck showed up.
                state.presence_thumb = None
                state.classify_thumb = None
            state.consecutive_empty = 0
        elif state.zoomed_in:
            state.consecutive_empty += 1
            if state.consecutive_empty >= self.cfg["empty_debounce_count"]:
                state.zoomed_in = False
                state.presence_thumb = None
                state.classify_thumb = None
                departed = True
                self.log.info(f"({bay}) {state.consecutive_empty} consecutive "
                              f"'{self.empty_status}' reads, reverting to "
                              f"baseline scan")
        # else: a first-look (or otherwise not-yet-zoomed) result came
        # back empty -- already at baseline, nothing to revert.

        self._notify(bay, status, timestamp, occupied, departed, comment, image_b64)

    def _notify(self, bay, status, timestamp, occupied, departed,
                comment=None, image_b64=None):
        """Hand the reading to a consumer as an already-interpreted event.

        `occupied` and `departed` are decided here rather than shipping
        raw internal state, because this module owns both the vocabulary
        (empty_status) and the debounce (empty_debounce_count) they
        derive from. A consumer re-deriving either would hold a second
        copy of a threshold that can drift out of sync with this one.

        `comment` and `image_b64` are the LLM's free-text description and
        the exact frame it was shown, passed through so a consumer (e.g.
        bay_state_engine) can build a richer notification without
        re-fetching or re-classifying anything itself.
        """
        if self.on_status is None:
            return
        try:
            self.on_status(bay, status, timestamp, occupied, departed,
                            comment, image_b64)
        except Exception:
            self.log.error(f"({bay}) on_status hook raised", exc_info=True)


def start_bay_monitor(cameras: dict, config: dict, publish_fn, on_status=None,
                       audit_dir: Path = None):
    """Starts the round-robin scanner in a background thread. Returns
    (thread, stop_event) -- set stop_event to ask the loop to exit at its
    next check point (it won't interrupt a request already in flight).

    on_status, if given, is called
    (bay, status, timestamp, occupied, departed) after every successful
    classification -- see bay_state.py's BayStateEngine, the main
    consumer, for what those last two mean and why they're computed here.

    audit_dir, if given, enables bay_monitor.save_latest_frame -- see
    BayMonitor._save_latest_frame.
    """
    monitor = BayMonitor(cameras, config, publish_fn, on_status, audit_dir)
    stop_event = threading.Event()
    t = threading.Thread(target=monitor.run, args=(stop_event,),
                          daemon=True, name="bay-monitor")
    t.start()
    return t, stop_event
