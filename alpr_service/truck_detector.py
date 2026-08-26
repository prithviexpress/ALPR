"""Truck + door-state detection: bay_monitor's fast classifier backend.

An alternative to the Ollama vision model (selected by
bay_monitor.classifier) that answers the same question -- "what is
happening at this bay" -- from a purpose-trained YOLO model instead of a
VLM prompt. Roughly 50ms per frame against multiple seconds, and
deterministic, which matters more than it sounds: bay_monitor scans
every bay from ONE thread, so a multi-second VLM call doesn't just delay
that bay, it stalls the round-robin for every other bay behind it. It
also removes the failure mode where a flaky/timing-out VLM call leaves a
bay with no status at all (activity: null) indefinitely.

What the VLM still does better: this model reports that a docked truck's
doors are open, but not whether cargo is moving IN or OUT -- "loading"
below is really "doors open at a dock". Distinguishing loading from
unloading is exactly the kind of open question worth spending a VLM call
on, on demand (see BayMonitor.ask / the webhook's /ask route), rather
than on every frame.

The Number_Plate class is deliberately NOT used to gate anything here --
a bay is occupied or not regardless of whether a plate happens to be
readable from the dock camera's angle, which is the whole reason
bay_monitor stopped using the ALPR plate model for presence. It's
reported (plate_visible/plate_boxes) purely as a signal a caller may use
to time an ALPR read for when a plate is actually in view.
"""
import time

from ultralytics import YOLO

from .logging_setup import get_logger

# The trained model's class names -> what they mean to this service.
# "status" joins bay_monitor.status_values (the same vocabulary the
# Ollama backend answers in, so everything downstream -- bay_state.py's
# sessions, the MQTT contract, the CSV -- is unchanged by which backend
# produced it). Overridable via bay_monitor.truck_class_map so a
# retrained model with different class names doesn't need a code change.
DEFAULT_CLASS_MAP = {
    "Truck_Enter_Closed":  {"status": "arriving", "door_state": "closed"},
    "Truck_Enter_Open":    {"status": "arriving", "door_state": "open"},
    "Truck_Docked_Closed": {"status": "docked",   "door_state": "closed"},
    "Truck_Docked_Open":   {"status": "loading",  "door_state": "open"},
}

# Detected, reported, but never treated as truck presence -- see the
# module docstring.
PLATE_CLASS_NAME = "Number_Plate"


class PlateAssistDetector:
    """The dedicated plate-only model (the same weights the ALPR workers
    use, config model_path) run alongside the truck model as a SECOND
    OPINION on whether anything is at the bay.

    Why bother when the truck model already has a Number_Plate class: the
    two models fail in different places. The truck model can miss a truck
    at an awkward angle or in poor light; the plate-only model is trained
    on one thing and often still picks the plate out of exactly those
    frames. Taking the union of the two -- a truck OR a plate counts as
    presence -- catches entries either alone would miss, which is the
    whole point of running both.

    It answers only "is a plate visible, and where". It deliberately does
    NOT decide status or door state: it cannot see either, and guessing
    from a plate box would invent information the model never reported."""

    def __init__(self, model_path, bm_cfg: dict, log=None):
        self.log = log or get_logger("PLATE_ASSIST")
        self.conf_threshold = bm_cfg["plate_assist_conf_threshold"]
        self.imgsz = bm_cfg["plate_assist_imgsz"]
        t = time.time()
        self.model = YOLO(str(model_path))
        self.names = dict(getattr(self.model, "names", {}) or {})
        self.log.info(f"plate-assist model loaded from {model_path} in "
                      f"{round(time.time() - t, 1)}s "
                      f"(conf>={self.conf_threshold}, imgsz={self.imgsz})")

    def detect(self, frame) -> dict:
        """Every box this model returns counts as a plate: it is a
        single-purpose plate detector, so unlike the truck model there is
        no class to filter on -- and filtering by NAME here would break
        silently against a model whose one class happens to be labelled
        something else."""
        t = time.time()
        results = self.model(frame, conf=self.conf_threshold,
                             imgsz=self.imgsz, verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                boxes.append((x1, y1, x2, y2, round(float(b.conf[0]), 3)))
        return {
            "plate_visible": bool(boxes),
            "plate_boxes": boxes,
            "inference_ms": int((time.time() - t) * 1000),
        }


class TruckDetector:
    """Wraps the truck/door YOLO model. One instance per BayMonitor (which
    is single-threaded), NOT one per worker -- the ALPR workers have their
    own plate model and don't need this one."""

    def __init__(self, model_path, bm_cfg: dict, log=None):
        self.log = log or get_logger("TRUCK_DETECT")
        self.conf_threshold = bm_cfg["truck_conf_threshold"]
        self.imgsz = bm_cfg["truck_imgsz"]
        self.plate_class = bm_cfg["truck_plate_class"]
        # Config entries override/extend the built-in map rather than
        # replacing it, so a partial override (e.g. renaming one class)
        # doesn't silently drop the other three.
        self.class_map = dict(DEFAULT_CLASS_MAP)
        self.class_map.update(bm_cfg["truck_class_map"] or {})

        t = time.time()
        self.model = YOLO(str(model_path))
        self.names = dict(getattr(self.model, "names", {}) or {})
        self.log.info(f"truck model loaded from {model_path} in "
                      f"{round(time.time() - t, 1)}s "
                      f"(conf>={self.conf_threshold}, imgsz={self.imgsz}, "
                      f"classes={sorted(self.names.values())})")
        self._warn_about_unmapped_classes()

    def _warn_about_unmapped_classes(self):
        """A model whose class names don't match the map detects things
        this service then silently ignores -- every frame reads as
        "empty" and no truck is ever seen, with nothing in the logs
        pointing at why. Say so loudly at startup instead, naming both
        sides, since the fix (bay_monitor.truck_class_map) needs the
        model's actual names."""
        known = set(self.class_map) | {self.plate_class}
        actual = set(self.names.values())
        unmapped = sorted(actual - known)
        missing = sorted(known - actual)
        if unmapped:
            self.log.warning(
                f"truck model reports classes with no mapping, which will "
                f"be ignored entirely: {unmapped} -- map them via "
                f"bay_monitor.truck_class_map if they mean something")
        if missing:
            self.log.warning(
                f"expected classes not present in this model: {missing} "
                f"(model has {sorted(actual)}) -- if the model was trained "
                f"with different names, set bay_monitor.truck_class_map")
        if not actual & set(self.class_map):
            self.log.error(
                f"NONE of this model's classes {sorted(actual)} map to a "
                f"bay status -- every frame will read as empty and no truck "
                f"will ever be detected. Fix bay_monitor.truck_class_map.")

    def detect(self, frame, empty_status: str) -> dict:
        """One frame -> what's at this bay. Returns a dict with:

          status       one of bay_monitor.status_values (empty_status if
                       no truck box cleared the confidence threshold)
          door_state   "open"/"closed", or None when no truck is detected
          confidence   the winning truck box's confidence (0.0 if none)
          class_name   the raw detected class, for logs/troubleshooting
          plate_boxes  Number_Plate boxes, as (x1,y1,x2,y2,conf) tuples
          plate_visible whether any plate cleared the threshold
          counts       per-class detection counts for this frame
          comment      a human-readable summary, so the `comment` field
                       carried by MQTT/webhook payloads stays populated
                       and useful whichever backend produced it

        The single highest-confidence truck box decides the status, not
        "any box of class X": two overlapping detections of the SAME
        truck as different classes (Docked_Open at 0.9 and Docked_Closed
        at 0.3) is the normal ambiguous case, and taking whichever came
        first in the result list would make the reading depend on box
        ordering rather than on the model's actual confidence."""
        t = time.time()
        results = self.model(frame, conf=self.conf_threshold,
                             imgsz=self.imgsz, verbose=False)

        best = None
        plate_boxes = []
        counts = {}
        for r in results:
            for b in r.boxes:
                name = self.names.get(int(b.cls[0]), str(int(b.cls[0])))
                conf = float(b.conf[0])
                counts[name] = counts.get(name, 0) + 1
                if name == self.plate_class:
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                    plate_boxes.append((x1, y1, x2, y2, round(conf, 3)))
                    continue
                mapped = self.class_map.get(name)
                if mapped is None:
                    continue
                if best is None or conf > best["confidence"]:
                    best = {"class_name": name, "confidence": conf,
                            "status": mapped["status"],
                            "door_state": mapped["door_state"]}

        inference_ms = int((time.time() - t) * 1000)
        if best is None:
            return {
                "status": empty_status, "door_state": None, "confidence": 0.0,
                "class_name": None, "plate_boxes": plate_boxes,
                "plate_visible": bool(plate_boxes), "counts": counts,
                "comment": "No truck detected in frame.",
                "inference_ms": inference_ms,
            }

        plate_note = (f" {len(plate_boxes)} number plate(s) visible."
                      if plate_boxes else " No number plate visible.")
        return {
            "status": best["status"],
            "door_state": best["door_state"],
            "confidence": round(best["confidence"], 3),
            "class_name": best["class_name"],
            "plate_boxes": plate_boxes,
            "plate_visible": bool(plate_boxes),
            "counts": counts,
            "comment": (f"Detected {best['class_name']} at "
                        f"{best['confidence']:.2f} confidence, so the bay "
                        f"reads as '{best['status']}' with the truck's "
                        f"doors {best['door_state']}.{plate_note}"),
            "inference_ms": inference_ms,
        }
