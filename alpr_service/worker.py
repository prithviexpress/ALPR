"""Per-job pipeline: capture -> detect -> OCR -> vote -> publish.

One Worker thread per NUM_WORKERS, each owning its own YOLO + PaddleOCR
model instances (loaded once at thread start) so concurrent jobs don't
share model state across threads.
"""
import json
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

from .config import BASE_DIR
from .image_ops import prep, sharpness, duplicate, save_debug_image, check_frame_size
from .logging_setup import get_logger
from .plate_text import is_valid, fix_indian_plate, weighted_vote
from .rtsp import build_rtsp_url, open_capture, redact, RtspOpenError

# Collection-stage error reasons mapped to a result-level status, so a
# downstream consumer waiting on RESULT_TOPIC_PREFIX/<bay> can tell
# "camera/stream problem" apart from a normal "no plate visible" miss --
# in R2 both cases were reported as NO_VALID_PLATE.
ERROR_STATUS = {
    'rtsp_config_error': 'CAMERA_CONFIG_ERROR',
    'rtsp_open_failed': 'CAMERA_UNREACHABLE',
    'no_frame_received': 'CAMERA_UNREACHABLE',
    'frame_size_mismatch': 'FRAME_SIZE_ERROR',
}

MAX_CONSECUTIVE_READ_FAILURES = 30
READ_FAILURE_BACKOFF_SEC = 0.05


class Worker(threading.Thread):
    def __init__(self, wid, jobs, cameras: dict, config: dict, publish_fn,
                 job_bus, audit_dir: Path):
        super().__init__(daemon=True, name=f"worker-{wid}")
        self.wid = wid
        self.jobs = jobs
        self.cameras = cameras
        self.config = config
        self.publish = publish_fn
        self.job_bus = job_bus
        self.audit_dir = audit_dir
        self.model = None
        self.ocr = None
        self.log = get_logger(f"WORKER-{wid}")

        alpr = config["alpr"]
        self.collection_timeout = alpr["collection_timeout"]
        self.frame_skip = alpr["frame_skip"]
        self.max_raw_samples = alpr["max_raw_samples"]
        self.best_samples = alpr["best_samples"]
        self.min_plate_width = alpr["min_plate_width"]
        self.min_plate_height = alpr["min_plate_height"]
        self.center_distance_limit = alpr["center_distance_limit"]
        self.debug_save_images = alpr.get("debug_save_images", True)
        self.expected_frame_width = alpr.get("expected_frame_width")
        self.expected_frame_height = alpr.get("expected_frame_height")
        self.frame_size_tolerance_pct = alpr.get("frame_size_tolerance_pct", 10)
        self.min_ocr_conf = alpr.get("min_ocr_conf", 0.35)
        self.rtsp_timeout_ms = config["rtsp"].get("timeout_ms", 8000)
        self.rtsp_timeout_option = config["rtsp"].get("timeout_option_name", "stimeout")
        self.rtsp_mode = config["rtsp"].get("mode")
        self.rtsp_stream = config["rtsp"].get("stream")

    def run(self):
        self.log.info("loading models...")
        t = time.time()
        self.model = YOLO(str(BASE_DIR / self.config["model_path"]))
        self.ocr = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)
        self.log.info(f"ready ({round(time.time() - t, 1)}s)")

        while True:
            job = self.jobs.get()
            bay = job['bay']
            try:
                self.handle(job)
            except Exception:
                tb = traceback.format_exc()
                self.log.error(f"({bay}) job failed unexpectedly:\n{tb}")
                self._publish_error(job, "WORKER_EXCEPTION", tb)
            finally:
                self.job_bus.release(bay)
                self.jobs.task_done()

    def _publish_error(self, job, reason, detail=""):
        """Last-resort publish so a downstream consumer waiting on this
        bay's result topic gets a signal instead of hanging forever --
        used both for camera/config errors and unhandled exceptions."""
        bay = job.get('bay', 'unknown')
        result = {
            'bay': bay,
            'truck_number': None,
            'status': 'ERROR',
            'error_reason': reason,
            'error_detail': str(detail)[-2000:],
            'event_time': job.get('event_time'),
            'ocr_time': datetime.now(timezone.utc).isoformat(),
        }
        try:
            topic = f"{self.config['mqtt']['result_topic_prefix']}/{bay}"
            self.publish(topic, json.dumps(result, default=str))
            self.log.error(f"({bay}) published ERROR result: {reason}")
        except Exception:
            self.log.error(f"({bay}) failed to publish ERROR result:\n"
                            f"{traceback.format_exc()}")

    # --------- per-job pipeline ---------
    def handle(self, job):
        bay = job['bay']
        event_time = job['event_time']
        cam = self.cameras.get(bay)
        if cam is None:
            self.log.warning(f"({bay}) no camera config, skipping")
            self._publish_error(job, "UNKNOWN_CAMERA")
            return
        if not cam.get('enabled', True):
            self.log.info(f"({bay}) camera disabled, skipping")
            return

        t0 = time.time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        folder = self.audit_dir / bay / ts
        debug_dir = folder / "debug"
        folder.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(exist_ok=True)
        self.log.info(f"({bay}) started, audit -> {folder}")

        samples, cstats = self.collect(cam, debug_dir)
        self.log.info(
            f"({bay}) frames={cstats['frames_read']} "
            f"evaluated={cstats['frames_evaluated']} "
            f"raw_cands={cstats['raw_candidates']} "
            f"kept={len(samples)} "
            f"rejected={cstats['rejected']} "
            f"in {cstats['collect_sec']}s "
            f"(rtsp open {cstats['rtsp_open_ms']}ms)")
        if cstats.get('error'):
            self.log.error(f"({bay}) collection aborted: {cstats['error']}")

        reads = []
        for i, s in enumerate(samples, 1):
            cv2.imwrite(str(folder / f"crop_{i:02d}.jpg"), s['crop'])
            prep_path = (debug_dir / f"prep_{i:02d}.jpg"
                         if self.debug_save_images else None)
            plate, conf, valid, raw = self.ocr_image(s['crop'], prep_path)
            self.log.debug(
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
        self.log.info(f"({bay}) {len(reads)} reads -> "
                       f"'{final}' (support {supporting_reads}, conf {confidence})")
        elapsed = round(time.time() - t0, 1)

        status = ERROR_STATUS.get(cstats.get('error'))
        if status is None:
            status = 'SUCCESS' if final else 'NO_VALID_PLATE'

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
            'status': status,
            'reads': reads,
            'collection': cstats,
            'audit_folder': str(folder),
        }

        (folder / 'result.json').write_text(
            json.dumps(result, indent=2, default=str))
        (folder / 'event.json').write_text(
            json.dumps(job, indent=2, default=str))
        topic = f"{self.config['mqtt']['result_topic_prefix']}/{bay}"
        self.publish(topic, json.dumps(result, default=str))
        self.log.info(f"({bay}) {final or status} published to {topic} "
                       f"({elapsed}s total, {len(reads)} reads)")

    def collect(self, cam: dict, debug_dir=None):
        stats = {'rtsp_open_ms': 0, 'frames_read': 0,
                  'frames_evaluated': 0, 'raw_candidates': 0,
                  'rejected': {}, 'collect_sec': 0.0}

        try:
            rtsp_url = build_rtsp_url(cam, self.config)
        except RtspOpenError as e:
            self.log.error(f"({cam.get('ip', cam.get('guid', '?'))}) "
                            f"RTSP config error: {e}")
            stats['error'] = 'rtsp_config_error'
            return [], stats

        self.log.debug(f"opening {redact(rtsp_url)} "
                        f"(mode={self.rtsp_mode}, stream={self.rtsp_stream}, "
                        f"timeout={self.rtsp_timeout_ms}ms)")
        cap, open_ms = open_capture(rtsp_url, self.rtsp_timeout_ms,
                                     self.rtsp_timeout_option)
        stats['rtsp_open_ms'] = open_ms

        if not cap.isOpened():
            self.log.error(f"FAILED to open stream ({cam.get('ip', '?')}) "
                            f"after {open_ms}ms")
            cap.release()
            stats['error'] = 'rtsp_open_failed'
            return [], stats
        self.log.info(f"stream open in {open_ms}ms ({cam.get('ip', '?')})")

        cands = []
        rejected = Counter()
        frame_no = 0
        first_saved = False
        annotated = None
        start = time.time()
        consecutive_read_failures = 0

        while time.time() - start < self.collection_timeout:
            ok, frame = cap.read()
            if not ok:
                rejected['read_fail'] += 1
                consecutive_read_failures += 1
                if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    self.log.error(
                        f"({cam.get('ip', '?')}) {consecutive_read_failures} "
                        f"consecutive read failures, aborting collection")
                    if not stats['frames_read']:
                        stats['error'] = 'no_frame_received'
                    break
                time.sleep(READ_FAILURE_BACKOFF_SEC)
                continue
            consecutive_read_failures = 0
            stats['frames_read'] += 1
            frame_no += 1

            if stats['frames_read'] == 1:
                ok_size, actual_w, actual_h = check_frame_size(
                    frame, self.expected_frame_width, self.expected_frame_height,
                    self.frame_size_tolerance_pct)
                stats['frame_size'] = f"{actual_w}x{actual_h}"
                if not ok_size:
                    self.log.error(
                        f"({cam.get('ip', '?')}) frame size {actual_w}x{actual_h} "
                        f"does not match expected "
                        f"{self.expected_frame_width}x{self.expected_frame_height} "
                        f"(tolerance {self.frame_size_tolerance_pct}%) -- check "
                        f"RTSP stream selection (got a substream instead of the "
                        f"main stream?)")
                    stats['error'] = 'frame_size_mismatch'
                    break
                self.log.debug(f"({cam.get('ip', '?')}) frame size OK: "
                                f"{actual_w}x{actual_h}")

            if frame_no % self.frame_skip:
                continue
            stats['frames_evaluated'] += 1

            if self.debug_save_images and debug_dir and not first_saved:
                save_debug_image(debug_dir, "00_first_frame.jpg", frame, self.log)
                first_saved = True

            x1r, y1r, x2r, y2r = cam['roi']
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
                    if w < self.min_plate_width or h < self.min_plate_height:
                        rejected['too_small'] += 1
                        continue
                    if ((by1 + by2) / 2) < roi.shape[0] * 0.45:
                        rejected['upper_half'] += 1
                        continue
                    dist = abs((bx1 + bx2) / 2 - center)
                    if dist > self.center_distance_limit:
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
                    self.log.debug(
                        f"candidate kept: box=({bx1},{by1},{bx2},{by2}) "
                        f"score={sc:.2f} yolo_conf={float(b.conf[0]):.2f}")
                    if self.debug_save_images and debug_dir:
                        if annotated is None:
                            annotated = roi.copy()
                        cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                                      (0, 255, 0), 2)
                        cv2.putText(annotated, f"{sc:.2f}", (bx1, by1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                    if len(cands) >= self.max_raw_samples:
                        break
                if len(cands) >= self.max_raw_samples:
                    break
            if len(cands) >= self.max_raw_samples:
                break

        cap.release()
        if self.debug_save_images and debug_dir and annotated is not None:
            save_debug_image(debug_dir, "01_annotated_roi.jpg", annotated, self.log)

        stats['raw_candidates'] = len(cands)
        stats['rejected'] = dict(rejected)
        stats['collect_sec'] = round(time.time() - start, 1)
        cands.sort(key=lambda x: x['score'], reverse=True)
        return cands[:self.best_samples], stats

    def ocr_image(self, img, debug_path=None):
        pimg = prep(img)
        if debug_path is not None:
            save_debug_image(debug_path.parent, debug_path.name, pimg, self.log)
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
            if conf < self.min_ocr_conf:
                self.log.debug(f"OCR fragment '{txt}' below min_ocr_conf "
                                f"({conf:.2f} < {self.min_ocr_conf}), dropped")
                continue
            parts.append(txt); confs.append(conf)
        if not parts:
            return '', 0.0, False, ''
        raw = ''.join(parts)
        txt = fix_indian_plate(raw)
        return txt, float(np.mean(confs)), is_valid(txt), raw
