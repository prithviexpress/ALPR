"""Side-by-side effectiveness probe for the two detection models.

A measuring instrument, not part of the service. It polls every camera
in turn, runs BOTH models on each frame, records what each one saw, and
saves an annotated image whenever either finds something -- so "which
model actually works, where, and how often" is answered from evidence
rather than from spot-checking logs.

Runs standalone (06_Model_Probe.py) against the same config.json and
cameras.json the service uses, and touches nothing the service owns: no
bay state, no sessions, no ALPR reads, no enter/leave results. Safe to
run alongside a live service, or on its own.

Two things it deliberately does NOT do, both because they would corrupt
the measurement:

  * No geometry filters. The ALPR pipeline drops boxes for too_small /
    upper_half / off_center before OCR, and in the field that discarded
    ALL 11 boxes on a bay where the plate model was working perfectly.
    A probe that applied the same filters would report zero and hide
    exactly what it exists to find. Raw model output only.
  * No ROI cropping by default. The ROI is itself a suspect -- a
    mis-placed one is indistinguishable from a blind model when you only
    see the filtered result -- so the default is the whole frame, with
    model_probe.use_roi to compare against.

Every frame is appended to detections.jsonl (one JSON object per line,
the analysis source of truth) and periodically summarised to the log
with the agreement stats that actually settle the question: how often
each model sees a truck the other misses, and how the truck model's
Number_Plate class compares against the dedicated plate model.
"""
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests

from .image_ops import thumbnail, duplicate_thumbs
from .logging_setup import get_logger
from .results import build_reply, result_topic
from .snapshot import build_snapshot_url, build_auth, fetch_snapshot, SnapshotError

# BGR, drawn on the saved frames.
PLATE_COLOR = (0, 255, 0)        # green  -- the dedicated plate model
TRUCK_COLOR = (255, 128, 0)      # blue   -- the truck model's truck classes
TRUCK_PLATE_COLOR = (0, 255, 255)  # yellow -- the truck model's Number_Plate



class ProbeStats:
    """Running tallies across the whole run, per bay and overall.

    The agreement counters are the point of the exercise: "plate model
    found one, truck model didn't" and its mirror are what decide
    whether the two models are redundant, complementary, or whether one
    can replace the other."""

    def __init__(self):
        self.frames = 0
        self.fetch_failures = 0
        self.plate_frames = 0        # frames where the plate model found >=1
        self.plate_boxes = 0
        self.truck_frames = 0        # frames where the truck model found a truck
        self.truck_boxes = 0
        self.truck_plate_frames = 0  # frames where the truck model saw a plate
        self.class_counts = {}
        # Agreement on "is a plate visible", the direct A/B between the
        # dedicated plate model and the truck model's Number_Plate class.
        self.both_plate = 0
        self.plate_only = 0
        self.truck_plate_only = 0
        self.neither_plate = 0
        self.plate_ms = 0
        self.truck_ms = 0

    def record(self, plate_res: dict, truck_res: dict):
        self.frames += 1
        p = plate_res["count"] > 0
        self.plate_boxes += plate_res["count"]
        self.plate_ms += plate_res["inference_ms"]
        if p:
            self.plate_frames += 1

        tp = truck_res["plate_count"] > 0
        self.truck_boxes += truck_res["truck_count"]
        self.truck_plate_frames += 1 if tp else 0
        self.truck_ms += truck_res["inference_ms"]
        if truck_res["truck_count"]:
            self.truck_frames += 1
        for name, n in truck_res["class_counts"].items():
            self.class_counts[name] = self.class_counts.get(name, 0) + n

        if p and tp:
            self.both_plate += 1
        elif p:
            self.plate_only += 1
        elif tp:
            self.truck_plate_only += 1
        else:
            self.neither_plate += 1

    def as_dict(self) -> dict:
        def pct(n):
            return round(100.0 * n / self.frames, 1) if self.frames else 0.0
        return {
            "frames": self.frames,
            "fetch_failures": self.fetch_failures,
            "plate_model": {
                "frames_with_detection": self.plate_frames,
                "pct_of_frames": pct(self.plate_frames),
                "total_boxes": self.plate_boxes,
                "avg_inference_ms": (round(self.plate_ms / self.frames, 1)
                                     if self.frames else 0.0),
            },
            "truck_model": {
                "frames_with_truck": self.truck_frames,
                "pct_of_frames": pct(self.truck_frames),
                "total_truck_boxes": self.truck_boxes,
                "frames_with_plate": self.truck_plate_frames,
                "class_counts": dict(sorted(self.class_counts.items(),
                                             key=lambda kv: -kv[1])),
                "avg_inference_ms": (round(self.truck_ms / self.frames, 1)
                                     if self.frames else 0.0),
            },
            "plate_agreement": {
                "both_found": self.both_plate,
                "plate_model_only": self.plate_only,
                "truck_model_only": self.truck_plate_only,
                "neither": self.neither_plate,
            },
        }


class ModelProbe:
    def __init__(self, cameras: dict, config: dict, publish_fn=None,
                 audit_dir: Path = None):
        self.log = get_logger("MODEL_PROBE")
        self.cameras = cameras
        self.config = config
        self.cfg = config["model_probe"]
        self.snap_cfg = config["snapshot"]
        self.publish = publish_fn
        self.interval_sec = self.cfg["poll_interval_ms"] / 1000
        self.use_roi = self.cfg["use_roi"]
        self.save_mode = self.cfg["save_images"]
        # Lowercased once here rather than per frame -- _should_save runs
        # on every frame of every bay.
        self.save_classes = {c.strip().lower()
                             for c in (self.cfg["save_classes"] or [])}
        # A valid entry's box must not reach further down the frame than
        # this -- see _is_valid_enter. The pixel form wins when set,
        # since that's how the line is actually measured off a frame;
        # the fraction is the resolution-independent fallback.
        self.enter_max_bottom_px = self.cfg["enter_max_bottom_px"]
        self.enter_max_bottom_frac = self.cfg["enter_max_bottom_frac"]
        self.save_annotated = self.cfg["annotate_saved_images"]
        self.max_saved_per_bay = self.cfg["max_saved_images_per_bay"]
        self.summary_every = self.cfg["summary_every_frames"]
        self.topic_prefix = self.cfg["topic_prefix"]

        self.out_dir = (audit_dir / self.cfg["output_subdir"]
                        if audit_dir else None)
        # Every image from every bay in ONE flat folder, named
        # "<bay>_<timestamp>...": a per-bay folder tree meant opening one
        # directory per camera to review a run, which is exactly the
        # friction this tool exists to remove. Named bay-first so a
        # single sorted listing still groups each camera's frames
        # together, chronologically within the group. They sit in an
        # images/ subfolder purely so detections.jsonl and summary.json
        # aren't buried among thousands of jpgs.
        self.images_dir = self.out_dir / "images" if self.out_dir else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = (self.out_dir / "detections.jsonl"
                           if self.out_dir else None)

        self.stats = ProbeStats()
        self.per_bay = {bay: ProbeStats() for bay in cameras}
        self.saved_counts = {bay: 0 for bay in cameras}

        config_dir = Path(config["_config_dir"])
        from .truck_detector import TruckDetector, PlateAssistDetector
        plate_path = config_dir / (self.cfg["plate_model_path"]
                                   or config["model_path"])
        truck_path = config_dir / (self.cfg["truck_model_path"]
                                   or config["bay_monitor"]["truck_model_path"])
        for label, path in (("plate", plate_path), ("truck", truck_path)):
            if not path.exists():
                raise FileNotFoundError(
                    f"model_probe needs the {label} model at {path} -- set "
                    f"model_probe.{label}_model_path (relative to "
                    f"config.json)")
        # Reuses the same two detector classes the service runs, on
        # purpose: a probe measuring something subtly different from what
        # production does would answer the wrong question.
        self.plate_model = PlateAssistDetector(plate_path, self._plate_cfg(),
                                                self.log)
        self.truck_model = TruckDetector(truck_path, self._truck_cfg(),
                                          self.log)

        # Plate reading. Loaded only when enabled, so a probe run that
        # only wants detection counts doesn't pay for PaddleOCR.
        self.reader = None
        self.sessions = {}
        # Bays whose reading session has finished -- read or given up --
        # and must not immediately open another for the SAME truck.
        # Without this, a truck with an unreadable plate exhausts its
        # budget, closes, and the very next frame opens a fresh session
        # with an empty dedupe list that re-reads the identical crops,
        # forever. Cleared when the bay is next seen with no truck and no
        # plate at all, i.e. that truck has left, which is the honest
        # definition of "a new truck may now be read".
        self.finished_bays = set()
        self.ocr_enabled = self.cfg["ocr_enabled"]
        self.ocr_trigger_classes = {c.strip().lower()
                                    for c in (self.cfg["ocr_trigger_classes"] or [])}
        self.ocr_trigger_on_plate = self.cfg["ocr_trigger_on_plate"]
        self.ocr_max_attempts = self.cfg["ocr_max_attempts"]
        self.ocr_session_timeout = self.cfg["ocr_session_timeout_sec"]
        self.ocr_crop_padding_pct = self.cfg["ocr_crop_padding_pct"]
        self.ocr_topic_prefix = self.cfg["ocr_topic_prefix"]
        self.ocr_save_crops = self.cfg["ocr_save_crops"]
        self.ocr_publish_failures = self.cfg["ocr_publish_failures"]
        self.plates_read = 0
        self.sessions_opened = 0
        self.crops_dir = self.out_dir / "plates" if self.out_dir else None
        if self.ocr_enabled:
            from .probe_ocr import PlateReader
            self.reader = PlateReader(config, self.log)
            if self.crops_dir is not None:
                self.crops_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(
                f"plate reading on: triggers="
                f"{sorted(self.ocr_trigger_classes) or 'none'}"
                + (" + any detected plate" if self.ocr_trigger_on_plate else "")
                + f", max {self.ocr_max_attempts} attempt(s) or "
                  f"{self.ocr_session_timeout}s per session -> "
                  f"{self.ocr_topic_prefix}/<bay>")

        self.session = requests.Session()
        self.session.trust_env = False   # cameras are on the local network
        self.auth = build_auth(config)
        self.log.info(
            f"model probe ready: {len(cameras)} camera(s), "
            f"poll_interval={self.cfg['poll_interval_ms']}ms "
            f"region={'roi' if self.use_roi else 'full_frame'} "
            f"save_images={self.save_mode}"
            + (f" ({', '.join(sorted(self.save_classes))}, box bottom <= "
               + (f"{self.enter_max_bottom_px}px"
                  if self.enter_max_bottom_px
                  else f"{self.enter_max_bottom_frac} of frame height")
               + ")" if self.save_mode == "classes" else "")
            + f" output={self.out_dir}")

    def _plate_cfg(self):
        return {"plate_assist_conf_threshold": self.cfg["plate_conf_threshold"],
                "plate_assist_imgsz": self.cfg["plate_imgsz"]}

    def _truck_cfg(self):
        return {"truck_conf_threshold": self.cfg["truck_conf_threshold"],
                "truck_imgsz": self.cfg["truck_imgsz"],
                "truck_plate_class": self.cfg["truck_plate_class"],
                "truck_class_map": self.cfg["truck_class_map"]}

    # ---------- per-frame ----------

    def _fetch(self, bay: str, cam: dict):
        try:
            url = build_snapshot_url(cam, self.config)
            frame, _, _ = fetch_snapshot(
                self.session, url, self.auth,
                self.snap_cfg["connect_timeout_ms"],
                self.snap_cfg["read_timeout_ms"])
            return frame
        except SnapshotError as e:
            self.log.debug(f"({bay}) snapshot fetch failed: {e}")
            return None

    def _region(self, bay: str, cam: dict, frame):
        """The pixels both models are shown. Full frame by default: a
        mis-placed ROI is itself one of the things being measured, and
        cropping to it first would hide that."""
        if not self.use_roi or not cam.get("roi"):
            return frame
        x1, y1, x2, y2 = cam["roi"]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.log.warning(f"({bay}) roi {cam['roi']} is empty against a "
                             f"{frame.shape[1]}x{frame.shape[0]} frame -- "
                             f"probing the full frame instead")
            return frame
        return roi

    def _run_plate_model(self, img) -> dict:
        r = self.plate_model.detect(img)
        return {"count": len(r["plate_boxes"]), "boxes": r["plate_boxes"],
                "inference_ms": r["inference_ms"]}

    def _run_truck_model(self, img) -> dict:
        """The truck model's RAW per-frame output -- every box it
        returned, not just the winning one detect() reports. Which
        classes fire, how often, and how confidently is the whole
        question, so collapsing to a single best box would throw away
        the measurement."""
        t = time.time()
        img_h = float(img.shape[0]) or 1.0
        results = self.truck_model.model(
            img, conf=self.cfg["truck_conf_threshold"],
            imgsz=self.cfg["truck_imgsz"], verbose=False)
        names = self.truck_model.names
        plate_class = self.cfg["truck_plate_class"]
        trucks, plates, counts = [], [], {}
        for r in results:
            for b in r.boxes:
                name = names.get(int(b.cls[0]), str(int(b.cls[0])))
                conf = round(float(b.conf[0]), 3)
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                counts[name] = counts.get(name, 0) + 1
                # How far down the frame this box's BOTTOM edge reaches,
                # 0.0 at the top and 1.0 at the very bottom. A truck up
                # against the dock reaches much further down the frame
                # than one still approaching, which is what separates a
                # real entry from a docked truck the model labelled
                # Enter -- see _is_valid_enter. Recorded on every box so
                # the threshold can be re-picked from real data.
                entry = {"class": name, "conf": conf, "box": [x1, y1, x2, y2],
                         "bottom_frac": round(y2 / img_h, 4)}
                (plates if name == plate_class else trucks).append(entry)
        return {"trucks": trucks, "plates": plates, "class_counts": counts,
                "truck_count": len(trucks), "plate_count": len(plates),
                "inference_ms": int((time.time() - t) * 1000)}

    def _plate_boxes(self, plate_res, truck_res):
        """Every plate box in this frame from BOTH models, tagged with
        which one found it. Both are used because either can see a plate
        the other misses -- that asymmetry is the thing the probe exists
        to measure, so the reader should not be limited to one of them."""
        boxes = [{"source": "plate_model", "box": list(b[:4]), "conf": b[4]}
                 for b in plate_res["boxes"]]
        boxes += [{"source": "truck_model", "box": e["box"], "conf": e["conf"]}
                  for e in truck_res["plates"]]
        return boxes

    def _ocr_trigger(self, truck_res, plate_boxes):
        """What, if anything, should open a reading session on this
        frame. Returns a short reason string, or None.

        Only trucks that pass the geometry test count -- a docked truck
        mislabelled Enter must not start a session, or the reader spends
        its whole budget on a plate that faces away from the camera."""
        for e in truck_res["trucks"]:
            if (e["class"].lower() in self.ocr_trigger_classes
                    and self._is_valid_enter(e)):
                return e["class"]
        if self.ocr_trigger_on_plate and plate_boxes:
            return "plate_detected"
        return None

    def _publish_plate(self, bay, payload):
        """Publishes to ocr_topic_prefix + "/<bay>" by default -- a topic
        of the probe's own, so a measurement run can't inject readings
        into a live result stream. Point ocr_topic_prefix at
        mqtt.enter_result_topic_prefix to feed the existing pipeline
        directly, which the payload shape already matches."""
        if self.publish is None:
            return
        try:
            self.publish(f"{self.ocr_topic_prefix}/{bay}",
                         json.dumps(payload, default=str))
        except Exception:
            self.log.warning(f"({bay}) failed to publish the plate result",
                             exc_info=True)

    def _end_session(self, bay, sess, status, plate=None, conf=0.0, raw=None):
        """Close a reading session, publishing only a SUCCESSFUL read by
        default (ocr_publish_failures to send the rest too).

        The published payload is results.build_reply's -- the exact
        contract the ALPR service already publishes and the existing
        downstream pipeline already consumes -- so a probe read drops
        into that pipeline with no change at either end. Built through
        the shared helper rather than hand-assembled here for the reason
        results.py exists at all: two producers of one message shape is
        how they drift.

        Nothing extra is added to it. The probe's own diagnostics
        (trigger, attempts, frames, elapsed, every per-attempt read) go
        to detections.jsonl instead, where they can't surprise a
        consumer expecting the standard seven keys.

        A failed session is still logged and still recorded, so nothing
        is lost for analysis -- only MQTT stays quiet. Subscribers want
        plate numbers, and a stream where most messages carry the
        unknown placeholder buries the ones that don't."""
        self.sessions.pop(bay, None)
        self.finished_bays.add(bay)
        # event_time is when the ENTRY was detected (the session opened),
        # not when OCR happened -- that's what event_time means in this
        # contract, and it's what lets a consumer line the read up with
        # the moment the truck actually arrived.
        payload = build_reply(
            self.config, bay, "enter",
            plate if status == "READ" else None,
            round(conf, 3),
            "SUCCESS" if status == "READ" else "NO_VALID_PLATE",
            sess.started_ts)
        if status == "READ":
            self.plates_read += 1
            self.log.info(f"({bay}) PLATE READ: {plate} (conf {conf:.2f}) "
                          f"after {sess.attempts} OCR attempt(s) over "
                          f"{sess.frames} frame(s), {sess.elapsed}s since "
                          f"{sess.trigger}")
        else:
            self.log.info(f"({bay}) no valid plate after {sess.attempts} "
                          f"attempt(s) over {sess.frames} frame(s) "
                          f"({sess.elapsed}s since {sess.trigger}) -> {status} "
                          f"(not published; the attempts are in "
                          f"detections.jsonl)")
        if status == "READ" or self.ocr_publish_failures:
            self._publish_plate(bay, payload)
        # The probe's own record keeps the detail the lean payload
        # deliberately omits.
        return {**payload, "_probe": {
            "outcome": status, "trigger": sess.trigger,
            "attempts": sess.attempts, "frames": sess.frames,
            "elapsed_sec": sess.elapsed, "raw": raw, "reads": sess.reads}}

    def _save_crop(self, bay, stamp, crop, plate):
        if self.crops_dir is None or not self.ocr_save_crops:
            return None
        try:
            safe_bay = str(bay).replace("/", "_").replace("\\", "_")
            safe_plate = (plate or "NOREAD").replace("/", "_")
            path = self.crops_dir / f"{safe_bay}_{stamp}_{safe_plate}.jpg"
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                path.write_bytes(buf.tobytes())
                return str(path)
        except Exception:
            self.log.warning(f"({bay}) failed to save a plate crop",
                             exc_info=True)
        return None

    def _read_plates(self, bay, img, stamp, plate_res, truck_res):
        """Open or continue this bay's reading session on this frame.

        A valid entry opens a session; every LATER frame's plate boxes
        are then read into it until one passes plate_text.is_valid() or
        the budget runs out. Reading across frames rather than once is
        the point: a single crop of a moving truck is often blurred or
        half-turned, and the probe is polling this bay anyway.

        Returns the per-frame OCR record for detections.jsonl."""
        from .probe_ocr import ReadSession, crop_with_padding

        plate_boxes = self._plate_boxes(plate_res, truck_res)

        # An empty bay means whatever was read (or given up on) has
        # left, so the next truck gets a fresh session.
        if not truck_res["trucks"] and not plate_boxes:
            if bay in self.finished_bays:
                self.finished_bays.discard(bay)
                self.log.debug(f"({bay}) bay clear -- ready to read again")
            return None

        sess = self.sessions.get(bay)

        if sess is None:
            if bay in self.finished_bays:
                # Already handled this truck; wait for the bay to clear.
                return None
            trigger = self._ocr_trigger(truck_res, plate_boxes)
            if trigger is None:
                return None
            sess = ReadSession(bay, trigger, datetime.now(timezone.utc).isoformat())
            self.sessions[bay] = sess
            self.sessions_opened += 1
            self.log.info(f"({bay}) plate-reading session opened by {trigger}")

        sess.frames += 1
        attempted, found = 0, None
        for pb in plate_boxes:
            if sess.attempts >= self.ocr_max_attempts:
                break
            crop = crop_with_padding(img, pb["box"], self.ocr_crop_padding_pct)
            if crop is None:
                continue
            # Don't spend an attempt re-reading a crop already tried --
            # a stationary truck presents near-identical pixels every
            # frame, and each repeat would burn budget for a result
            # already known.
            thumb = thumbnail(crop, (64, 32))
            if any(duplicate_thumbs(thumb, t, 3.0) for t in sess.tried_thumbs):
                continue
            sess.tried_thumbs.append(thumb)
            sess.attempts += 1
            attempted += 1
            plate, conf, valid, raw = self.reader.read(crop)
            sess.reads.append({"source": pb["source"], "box": pb["box"],
                               "det_conf": pb["conf"], "plate": plate,
                               "ocr_conf": round(conf, 3), "valid": valid,
                               "raw": raw})
            self.log.debug(f"({bay}) OCR attempt {sess.attempts}: "
                           f"raw='{raw}' -> '{plate}' conf={conf:.2f} "
                           f"valid={valid} (from {pb['source']})")
            if valid:
                found = (plate, conf, raw, crop)
                break

        if found is not None:
            plate, conf, raw, crop = found
            saved = self._save_crop(bay, stamp, crop, plate)
            result = self._end_session(bay, sess, "READ", plate, conf, raw)
            return {"session": "closed", "outcome": "READ",
                    "plate": result["truck_number"],
                    "confidence": result["confidence"], "crop": saved,
                    "published": result, "attempts_this_frame": attempted}

        if sess.attempts >= self.ocr_max_attempts:
            self._end_session(bay, sess, "NO_VALID_PLATE")
            return {"session": "closed", "outcome": "NO_VALID_PLATE",
                    "attempts_this_frame": attempted}
        if sess.elapsed >= self.ocr_session_timeout:
            self._end_session(bay, sess, "TIMEOUT")
            return {"session": "closed", "outcome": "TIMEOUT",
                    "attempts_this_frame": attempted}

        return {"session": "open", "trigger": sess.trigger,
                "attempts": sess.attempts, "frames": sess.frames,
                "elapsed_sec": sess.elapsed,
                "attempts_this_frame": attempted}

    def _is_valid_enter(self, entry: dict) -> bool:
        """Is this box geometrically consistent with a truck still
        ENTERING, rather than one already docked?

        The rule is one line across the frame: a genuine entry's box
        must not reach further DOWN than it. A truck that has finished
        reversing in sits right up against the dock and therefore
        against the bottom of the frame, so its box bottom runs past
        that line; one still approaching is further away and its box
        ends higher up.

        This exists because the model cannot make the distinction
        itself -- clearly docked trucks come back as Truck_Enter_Open
        and Truck_Enter_Closed at both high and low confidence -- and
        geometry decides it from the same single frame, with no history
        to keep.

        enter_max_bottom_px wins when set, because that is how the line
        is actually measured: read straight off a frame in the same
        pixel coordinates the boxes use (e.g. 1600 on a 1944-tall
        frame). enter_max_bottom_frac is the resolution-independent
        fallback for when it isn't. Both 0 disables the check."""
        if self.enter_max_bottom_px:
            return entry["box"][3] <= self.enter_max_bottom_px
        if not self.enter_max_bottom_frac:
            return True
        return entry.get("bottom_frac", 0.0) <= self.enter_max_bottom_frac

    def _annotate(self, img, plate_res, truck_res):
        """Both models' boxes drawn on one image, colour-coded, so a
        human can see at a glance where they agree and where they don't:
        green = plate model, blue = truck model's truck classes, yellow =
        the truck model's own Number_Plate."""
        out = img.copy()
        for (x1, y1, x2, y2, conf) in plate_res["boxes"]:
            cv2.rectangle(out, (x1, y1), (x2, y2), PLATE_COLOR, 2)
            cv2.putText(out, f"plate-model {conf:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PLATE_COLOR, 2)
        for e in truck_res["trucks"]:
            x1, y1, x2, y2 = e["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), TRUCK_COLOR, 2)
            cv2.putText(out, f"{e['class']} {e['conf']:.2f}",
                        (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, TRUCK_COLOR, 2)
        for e in truck_res["plates"]:
            x1, y1, x2, y2 = e["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), TRUCK_PLATE_COLOR, 2)
            cv2.putText(out, f"truck-model plate {e['conf']:.2f}",
                        (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, TRUCK_PLATE_COLOR, 2)
        return out

    def _should_save(self, plate_res, truck_res) -> bool:
        if self.save_mode == "none":
            return False
        if self.save_mode == "all":
            return True
        if self.save_mode == "classes":
            # Only frames where the truck model reported one of
            # save_classes -- by default the two ENTERING classes, since
            # entry is the only moment a dock camera can read a plate and
            # a run full of docked-truck frames is mostly noise when
            # that's the question. Compared case-insensitively so a
            # config typo in casing doesn't silently save nothing.
            # A truck box only counts if it ALSO passes the geometry
            # test, so a docked truck mislabelled as Enter is rejected
            # on where its box sits rather than on what it was called.
            # Plate boxes aren't geometry-gated: the rule is about how
            # far down a TRUCK reaches, and a plate is small and low by
            # nature.
            seen = {e["class"].lower() for e in truck_res["trucks"]
                    if self._is_valid_enter(e)}
            seen |= {e["class"].lower() for e in truck_res["plates"]}
            return bool(seen & self.save_classes)
        found_plate = plate_res["count"] > 0 or truck_res["plate_count"] > 0
        found_truck = truck_res["truck_count"] > 0
        if self.save_mode == "plate":
            return found_plate
        if self.save_mode == "truck":
            return found_truck
        # "any": either model saw anything at all
        return found_plate or found_truck

    def _save_image(self, bay: str, ts: str, img, plate_res, truck_res):
        """Best-effort: a disk problem must never stop the probe, since
        the JSONL record is the primary measurement and the image is
        corroboration."""
        if self.out_dir is None:
            return None
        if (self.max_saved_per_bay
                and self.saved_counts.get(bay, 0) >= self.max_saved_per_bay):
            return None
        try:
            # "<bay>_<timestamp>_p<plate-model>_t<trucks>_tp<truck-model
            # plates>.jpg" -- bay and time up front so a sorted listing
            # groups each camera chronologically, then the verdict, so
            # the listing alone shows which model found what without
            # opening anything. A bay name is used as a path component
            # here, so strip separators the way worker.py does for
            # plates.
            safe_bay = str(bay).replace("/", "_").replace("\\", "_")
            name = (f"{safe_bay}_{ts}_p{plate_res['count']}"
                    f"_t{truck_res['truck_count']}"
                    f"_tp{truck_res['plate_count']}.jpg")
            out = (self._annotate(img, plate_res, truck_res)
                   if self.save_annotated else img)
            ok, buf = cv2.imencode(".jpg", out)
            if not ok:
                return None
            path = self.images_dir / name
            path.write_bytes(buf.tobytes())
            self.saved_counts[bay] = self.saved_counts.get(bay, 0) + 1
            return str(path)
        except Exception:
            self.log.warning(f"({bay}) failed to save probe image",
                             exc_info=True)
            return None

    def _record(self, row: dict):
        if self.jsonl_path is None:
            return
        try:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            self.log.warning("failed to append to detections.jsonl",
                             exc_info=True)

    def probe_bay(self, bay: str, cam: dict):
        """One camera, one frame, both models. Returns the record dict
        (or None if the fetch failed)."""
        frame = self._fetch(bay, cam)
        if frame is None:
            self.stats.fetch_failures += 1
            self.per_bay[bay].fetch_failures += 1
            return None

        img = self._region(bay, cam, frame)
        plate_res = self._run_plate_model(img)
        truck_res = self._run_truck_model(img)

        self.stats.record(plate_res, truck_res)
        self.per_bay[bay].record(plate_res, truck_res)

        ts = datetime.now(timezone.utc)
        stamp = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        saved = (self._save_image(bay, stamp, img, plate_res, truck_res)
                 if self._should_save(plate_res, truck_res) else None)

        ocr_row = None
        if self.reader is not None:
            try:
                ocr_row = self._read_plates(bay, img, stamp, plate_res, truck_res)
            except Exception:
                # A reading failure must never stop the detection
                # measurement, which is the primary output.
                self.log.error(f"({bay}) plate reading failed", exc_info=True)

        row = {
            "bay": bay,
            "timestamp": ts.isoformat(),
            "region": "roi" if (self.use_roi and cam.get("roi")) else "full_frame",
            "frame_size": f"{img.shape[1]}x{img.shape[0]}",
            "plate_model": {
                "count": plate_res["count"],
                "boxes": [{"box": list(b[:4]), "conf": b[4]}
                          for b in plate_res["boxes"]],
                "inference_ms": plate_res["inference_ms"],
            },
            "truck_model": {
                "truck_count": truck_res["truck_count"],
                "plate_count": truck_res["plate_count"],
                "trucks": truck_res["trucks"],
                "plates": truck_res["plates"],
                "class_counts": truck_res["class_counts"],
                "inference_ms": truck_res["inference_ms"],
            },
            "image": saved,
            "ocr": ocr_row,
        }
        self._record(row)

        if plate_res["count"] or truck_res["truck_count"] or truck_res["plate_count"]:
            classes = ", ".join(f"{e['class']} {e['conf']:.2f}"
                                for e in truck_res["trucks"]) or "no truck"
            self.log.info(
                f"({bay}) plate-model={plate_res['count']} box(es) "
                f"({plate_res['inference_ms']}ms) | truck-model: {classes}, "
                f"plate x{truck_res['plate_count']} "
                f"({truck_res['inference_ms']}ms)"
                + (f" -> {saved}" if saved else ""))

        if self.publish is not None:
            try:
                self.publish(f"{self.topic_prefix}/{bay}",
                             json.dumps(row, default=str))
            except Exception:
                self.log.warning(f"({bay}) failed to publish probe result",
                                 exc_info=True)
        return row

    def log_summary(self):
        """The numbers that answer the question, per bay and overall."""
        s = self.stats.as_dict()
        a = s["plate_agreement"]
        self.log.info(
            f"SUMMARY after {s['frames']} frame(s) "
            f"({s['fetch_failures']} fetch failure(s)): "
            f"plate model found something in {s['plate_model']['frames_with_detection']} "
            f"({s['plate_model']['pct_of_frames']}%, avg "
            f"{s['plate_model']['avg_inference_ms']}ms); truck model found a truck "
            f"in {s['truck_model']['frames_with_truck']} "
            f"({s['truck_model']['pct_of_frames']}%, avg "
            f"{s['truck_model']['avg_inference_ms']}ms), classes="
            f"{s['truck_model']['class_counts']}"
            + (f"; plate reading: {self.plates_read} read from "
               f"{self.sessions_opened} session(s)"
               if self.sessions_opened else ""))
        self.log.info(
            f"SUMMARY plate agreement: both={a['both_found']} "
            f"plate-model-only={a['plate_model_only']} "
            f"truck-model-only={a['truck_model_only']} "
            f"neither={a['neither']}")
        for bay, st in self.per_bay.items():
            if not st.frames:
                continue
            d = st.as_dict()
            self.log.info(
                f"SUMMARY ({bay}) {d['frames']} frames: "
                f"plate={d['plate_model']['frames_with_detection']} "
                f"({d['plate_model']['pct_of_frames']}%) "
                f"truck={d['truck_model']['frames_with_truck']} "
                f"({d['truck_model']['pct_of_frames']}%) "
                f"classes={d['truck_model']['class_counts']}")
        self._write_summary_file()

    def _write_summary_file(self):
        if self.out_dir is None:
            return
        try:
            payload = {
                "generated": datetime.now(timezone.utc).isoformat(),
                "overall": self.stats.as_dict(),
                "per_bay": {b: s.as_dict() for b, s in self.per_bay.items()
                            if s.frames},
            }
            tmp = self.out_dir / "summary.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2, default=str))
            tmp.replace(self.out_dir / "summary.json")
        except Exception:
            self.log.warning("failed to write summary.json", exc_info=True)

    def run(self, stop_event: threading.Event):
        """Round-robin every enabled camera until stopped. Each bay's turn
        is wrapped: one camera failing must not end the run, since a
        partial measurement across the other bays is still worth having."""
        rounds = 0
        while not stop_event.is_set():
            for bay, cam in self.cameras.items():
                if stop_event.is_set():
                    break
                if not cam.get("enabled", True):
                    continue
                try:
                    self.probe_bay(bay, cam)
                except Exception:
                    self.log.error(f"({bay}) probe failed", exc_info=True)
                if (self.summary_every
                        and self.stats.frames
                        and self.stats.frames % self.summary_every == 0):
                    self.log_summary()
                stop_event.wait(self.interval_sec)
            rounds += 1
        self.log.info(f"probe stopped after {rounds} round(s)")
        self.log_summary()
