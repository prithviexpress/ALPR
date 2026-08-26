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

from .logging_setup import get_logger
from .snapshot import build_snapshot_url, build_auth, fetch_snapshot, SnapshotError

# BGR, drawn on the saved frames.
PLATE_COLOR = (0, 255, 0)        # green  -- the dedicated plate model
TRUCK_COLOR = (255, 128, 0)      # blue   -- the truck model's truck classes
TRUCK_PLATE_COLOR = (0, 255, 255)  # yellow -- the truck model's Number_Plate



def box_iou(a, b) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
        # Per-bay motion tracking. "Entering" versus "docked" is
        # inherently TEMPORAL -- a truck is entering because it is
        # moving -- and a single still frame simply doesn't carry that,
        # which is why the model confuses the two. Polling every couple
        # of seconds does carry it, so measure it directly: how much the
        # biggest truck box moved since this bay's last frame.
        self.last_box = {}            # bay -> [x1,y1,x2,y2] of the biggest truck
        self.last_box_time = {}       # bay -> monotonic seconds
        self.stationary_run = {}      # bay -> consecutive stationary frames
        self.skipped_stationary = 0
        self.stationary_iou = self.cfg["stationary_iou_threshold"]
        self.stationary_min_frames = self.cfg["stationary_min_frames"]
        self.skip_stationary_saves = self.cfg["skip_stationary_saves"]

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

        self.session = requests.Session()
        self.session.trust_env = False   # cameras are on the local network
        self.auth = build_auth(config)
        self.log.info(
            f"model probe ready: {len(cameras)} camera(s), "
            f"poll_interval={self.cfg['poll_interval_ms']}ms "
            f"region={'roi' if self.use_roi else 'full_frame'} "
            f"save_images={self.save_mode}"
            + (f" ({', '.join(sorted(self.save_classes))})"
               if self.save_mode == "classes" else "")
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
                entry = {"class": name, "conf": conf, "box": [x1, y1, x2, y2]}
                (plates if name == plate_class else trucks).append(entry)
        return {"trucks": trucks, "plates": plates, "class_counts": counts,
                "truck_count": len(trucks), "plate_count": len(plates),
                "inference_ms": int((time.time() - t) * 1000)}

    def _motion(self, bay: str, img, truck_res: dict) -> dict:
        """How much this bay's biggest truck box moved since its last
        frame -- the numeric answer to "is this really entering, or is it
        a docked truck the model mislabelled as Enter".

        A truck that has finished reversing in sits still: its box
        overlaps its previous box almost perfectly (iou_prev near 1.0)
        frame after frame. One that is genuinely entering moves, so the
        overlap drops. stationary_frames counts consecutive still frames,
        which is stronger evidence than any single reading -- one frame
        of stillness is just a truck pausing, twenty is a parked one.

        The BIGGEST box is tracked rather than the highest-confidence
        one: this is a geometric question, and area is far more stable
        across frames than confidence, which can swing while the box
        barely moves.

        Also records size and position, so a threshold can be picked
        from the recorded data afterwards rather than guessed now -- on a
        dome camera a docked truck fills much more of the frame than an
        approaching one."""
        h, w = img.shape[:2]
        frame_area = float(w * h) or 1.0
        boxes = [e["box"] for e in truck_res["trucks"]]
        now = time.monotonic()

        if not boxes:
            # No truck: nothing to compare, and any run of stillness
            # ends here rather than carrying across an empty gap.
            self.last_box.pop(bay, None)
            self.last_box_time.pop(bay, None)
            self.stationary_run[bay] = 0
            return {"iou_prev": None, "stationary": None,
                    "stationary_frames": 0, "secs_since_prev": None,
                    "area_frac": None, "center": None}

        biggest = max(boxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
        x1, y1, x2, y2 = biggest
        area_frac = round(max(0, x2 - x1) * max(0, y2 - y1) / frame_area, 4)
        center = [round((x1 + x2) / 2 / w, 4), round((y1 + y2) / 2 / h, 4)]

        prev = self.last_box.get(bay)
        iou = round(box_iou(biggest, prev), 4) if prev else None
        secs = (round(now - self.last_box_time[bay], 2)
                if bay in self.last_box_time else None)

        stationary = None
        if iou is not None:
            stationary = iou >= self.stationary_iou
            self.stationary_run[bay] = (self.stationary_run.get(bay, 0) + 1
                                        if stationary else 0)
        else:
            # First sighting of this truck -- unknown, not stationary.
            self.stationary_run[bay] = 0

        self.last_box[bay] = biggest
        self.last_box_time[bay] = now
        return {"iou_prev": iou, "stationary": stationary,
                "stationary_frames": self.stationary_run.get(bay, 0),
                "secs_since_prev": secs, "area_frac": area_frac,
                "center": center}

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

    def _should_save(self, plate_res, truck_res, motion) -> bool:
        if self.save_mode == "none":
            return False
        # Numerically eliminate the docked trucks the model mislabels as
        # Enter: after stationary_min_frames consecutive frames where the
        # box barely moved, this truck is parked whatever class came
        # back. Applied before the mode checks so it holds for all of
        # them, and never to "all", which means exactly what it says.
        if (self.skip_stationary_saves and self.save_mode != "all"
                and motion.get("stationary_frames", 0) >= self.stationary_min_frames):
            self.skipped_stationary += 1
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
            seen = {e["class"].lower()
                    for e in truck_res["trucks"] + truck_res["plates"]}
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

        # Computed before the save decision: whether this truck is
        # actually moving is one of the things that decides it.
        motion = self._motion(bay, img, truck_res)

        ts = datetime.now(timezone.utc)
        stamp = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        saved = (self._save_image(bay, stamp, img, plate_res, truck_res)
                 if self._should_save(plate_res, truck_res, motion) else None)

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
            "motion": motion,
            "image": saved,
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
                + (f" | iou_prev={motion['iou_prev']} "
                   f"still x{motion['stationary_frames']} "
                   f"area={motion['area_frac']}"
                   if motion["iou_prev"] is not None else "")
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
            + (f"; {self.skipped_stationary} frame(s) not saved as "
               f"stationary (>= {self.stationary_min_frames} frames at "
               f"iou >= {self.stationary_iou} -- docked, whatever class "
               f"was reported)" if self.skipped_stationary else ""))
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
