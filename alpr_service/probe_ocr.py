"""Plate reading for the model probe: turn a detected plate box into
text, and hold one reading session per bay across consecutive frames.

Why sessions rather than one-shot reads: a single frame's crop of a
moving truck is often motion-blurred, half-turned or too small, and the
probe is already polling that bay every couple of seconds anyway. So a
valid entry STARTS a session, and every subsequent frame's plate boxes
are read into it until one produces a plate that passes
plate_text.is_valid() or the budget runs out. That is the same
"keep trying across the visit" idea the ALPR service's own state engine
uses, reduced to what the probe needs.

Deliberately does NOT apply the geometry filters (too_small /
upper_half / off_center) the ALPR worker runs before OCR. Those
discarded all 11 boxes on a bay where the plate model was working
perfectly, and the whole point of the probe is to find out what the
models can actually do -- so every detected box gets read.
"""
import time

import cv2
import numpy as np

from .image_ops import prep, thumbnail, duplicate_thumbs
from .logging_setup import get_logger
from .plate_text import fix_indian_plate, is_valid


class PlateReader:
    """Wraps PaddleOCR. One instance for the whole probe -- it runs
    single-threaded, so there is no reason to load the model per bay."""

    def __init__(self, config: dict, log=None):
        self.log = log or get_logger("PROBE_OCR")
        cfg = config["model_probe"]
        alpr = config["alpr"]
        self.min_conf = cfg["ocr_min_conf"]
        self.prep_height = alpr["ocr_prep_target_height"]
        self.prep_padding = alpr["ocr_prep_padding"]

        from pathlib import Path
        from paddleocr import PaddleOCR
        from .worker import ensure_cls_placeholder

        config_dir = Path(config["_config_dir"])
        det = config_dir / alpr["paddleocr_det_model_dir"]
        rec = config_dir / alpr["paddleocr_rec_model_dir"]
        cls = config_dir / alpr["paddleocr_cls_model_dir"]
        for d in (det, rec, cls):
            d.mkdir(parents=True, exist_ok=True)
        # Same reason worker.py does this: PaddleOCR's constructor tries
        # to download the angle classifier even though use_angle_cls is
        # False and it is never used for inference.
        ensure_cls_placeholder(cls, self.log)
        t = time.time()
        self.ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False,
                             det_model_dir=str(det), rec_model_dir=str(rec),
                             cls_model_dir=str(cls))
        self.log.info(f"PaddleOCR loaded in {round(time.time() - t, 1)}s "
                      f"(min_conf={self.min_conf})")

    def read(self, crop):
        """One crop -> (plate, mean_conf, valid, raw). Empty strings and
        valid=False when nothing legible came back."""
        try:
            pimg = prep(crop, self.prep_height, self.prep_padding)
            result = self.ocr.ocr(pimg, cls=False)
        except Exception:
            self.log.warning("OCR call failed on a crop", exc_info=True)
            return "", 0.0, False, ""
        if not result or result[0] is None:
            return "", 0.0, False, ""
        parts, confs = [], []
        for item in result[0]:
            try:
                _box, (txt, conf) = item
            except (TypeError, ValueError):
                continue
            if conf < self.min_conf:
                continue
            parts.append(txt)
            confs.append(conf)
        if not parts:
            return "", 0.0, False, ""
        raw = "".join(parts)
        txt = fix_indian_plate(raw)
        return txt, float(np.mean(confs)), is_valid(txt), raw


class ReadSession:
    """One bay's in-progress attempt to read a plate, spanning frames."""

    def __init__(self, bay: str, trigger: str, started_ts: str):
        self.bay = bay
        self.trigger = trigger          # what opened it, for the payload
        self.started = time.time()
        self.started_ts = started_ts
        self.attempts = 0
        self.frames = 0
        self.tried_thumbs = []          # crops already read, to not re-read them
        self.reads = []                 # every attempt, valid or not

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.started, 1)


def crop_with_padding(img, box, pad_pct: int):
    """Crop `box` out of `img`, expanded by pad_pct of its own size on
    each side and clamped to the image.

    The padding matters: a slightly undersized detection clips a
    character off the plate's edge, which was confirmed in the field as
    two adjacent boxes each capturing half a plate. Cheap insurance
    against the same thing here."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    px = int(bw * pad_pct / 100)
    py = int(bh * pad_pct / 100)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
    crop = img[cy1:cy2, cx1:cx2]
    return crop if crop.size else None
