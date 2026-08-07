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
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR

from .config import BASE_DIR
from .image_ops import prep, sharpness, duplicate, save_debug_image, check_frame_size
from .logging_setup import get_logger
from .plate_text import is_valid, fix_indian_plate, weighted_vote
from .snapshot import build_snapshot_url, build_auth, fetch_snapshot, SnapshotError

# Collection-stage error reasons mapped to a result-level status, so a
# downstream consumer waiting on RESULT_TOPIC_PREFIX/<bay> can tell
# "camera/stream problem" apart from a normal "no plate visible" miss --
# in R2 both cases were reported as NO_VALID_PLATE.
ERROR_STATUS = {
    'snapshot_config_error': 'CAMERA_CONFIG_ERROR',
    'no_frame_received': 'CAMERA_UNREACHABLE',
    'frame_size_mismatch': 'FRAME_SIZE_ERROR',
}

MAX_CONSECUTIVE_FETCH_FAILURES = 10
FETCH_FAILURE_BACKOFF_SEC = 0.2


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
        self.session = None
        self.auth = None
        self.log = get_logger(f"WORKER-{wid}")

        alpr = config["alpr"]
        self.collection_timeout = alpr["collection_timeout"]
        self.max_raw_samples = alpr["max_raw_samples"]
        self.best_samples = alpr["best_samples"]
        self.min_plate_width = alpr["min_plate_width"]
        self.min_plate_height = alpr["min_plate_height"]
        self.center_distance_limit = alpr["center_distance_limit"]
        self.upper_half_fraction = alpr.get("upper_half_fraction", 0.45)
        self.plate_crop_padding_pct = alpr.get("plate_crop_padding_pct", 15)
        self.troubleshooting = alpr.get("diagnostics_mode", "basic") == "troubleshooting"
        self.publish_no_valid_plate = alpr.get("publish_no_valid_plate", False)
        self.expected_frame_width = alpr.get("expected_frame_width")
        self.expected_frame_height = alpr.get("expected_frame_height")
        self.frame_size_tolerance_pct = alpr.get("frame_size_tolerance_pct", 10)
        self.min_ocr_conf = alpr.get("min_ocr_conf", 0.35)
        snap_cfg = config["snapshot"]
        self.connect_timeout_ms = snap_cfg.get("connect_timeout_ms", 3000)
        self.read_timeout_ms = snap_cfg.get("read_timeout_ms", 3000)
        self.poll_interval_ms = snap_cfg.get("poll_interval_ms", 0)

    def run(self):
        self.log.info("loading models...")
        t = time.time()
        self.model = YOLO(str(BASE_DIR / self.config["model_path"]))
        self.ocr = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)
        # One Session/HTTPDigestAuth per worker thread, reused across every
        # job it handles: a Session keeps the TCP connection (and, for
        # HTTPDigestAuth, the last nonce) alive across requests, so only
        # the first snapshot fetch to a given camera pays for a fresh
        # handshake -- not every single one.
        self.session = requests.Session()
        self.auth = build_auth(self.config)
        self.log.info(f"ready ({round(time.time() - t, 1)}s)")

        while True:
            job = self.jobs.get()
            bay = job['bay']
            direction = job.get('direction', '')
            try:
                self.handle(job)
            except Exception:
                tb = traceback.format_exc()
                self.log.error(f"({bay}/{direction}) job failed unexpectedly:\n{tb}")
                self._publish_error(job, "WORKER_EXCEPTION", tb)
            finally:
                self.job_bus.release(bay, direction)
                self.jobs.task_done()

    def _result_topic(self, direction: str, bay: str) -> str:
        prefix_key = f"{direction}_result_topic_prefix"
        prefix = self.config['mqtt'].get(prefix_key)
        if not prefix:
            # Shouldn't happen -- both are required config keys -- but
            # fail toward a topic that's at least identifiable rather
            # than raising and losing the result entirely.
            prefix = f"alpr_result/{direction or 'unknown'}"
        return f"{prefix}/{bay}"

    def _publish_error(self, job, reason, detail=""):
        """Last-resort publish so a downstream consumer waiting on this
        bay's result topic gets a signal instead of hanging forever --
        used both for camera/config errors and unhandled exceptions."""
        bay = job.get('bay', 'unknown')
        direction = job.get('direction', '')
        result = {
            'bay': bay,
            'direction': direction,
            'truck_number': None,
            'status': 'ERROR',
            'error_reason': reason,
            'error_detail': str(detail)[-2000:],
            'event_time': job.get('event_time'),
            'ocr_time': datetime.now(timezone.utc).isoformat(),
        }
        try:
            topic = self._result_topic(direction, bay)
            self.publish(topic, json.dumps(result, default=str))
            self.log.error(f"({bay}/{direction}) published ERROR result: {reason}")
        except Exception:
            self.log.error(f"({bay}/{direction}) failed to publish ERROR result:\n"
                            f"{traceback.format_exc()}")

    # --------- per-job pipeline ---------
    def handle(self, job):
        bay = job['bay']
        direction = job.get('direction', '')
        event_time = job['event_time']
        cam = self.cameras.get(bay)
        if cam is None:
            self.log.warning(f"({bay}/{direction}) no camera config, skipping")
            self._publish_error(job, "UNKNOWN_CAMERA")
            return
        if not cam.get('enabled', True):
            self.log.info(f"({bay}/{direction}) camera disabled, skipping")
            return

        t0 = time.time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        folder = self.audit_dir / bay / f"{ts}_{direction}"
        debug_dir = folder / "debug"
        folder.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(exist_ok=True)
        self.log.info(f"({bay}/{direction}) started, audit -> {folder}")

        samples, cstats = self.collect(cam, debug_dir)
        self.log.info(
            f"({bay}/{direction}) frames={cstats['frames_read']} "
            f"boxes_detected={cstats['total_boxes_detected']} "
            f"raw_cands={cstats['raw_candidates']} "
            f"kept={len(samples)} "
            f"rejected={cstats['rejected']} "
            f"in {cstats['collect_sec']}s "
            f"(first fetch {cstats['first_fetch_ms']}ms, "
            f"avg fetch {cstats['avg_fetch_ms']}ms, "
            f"{cstats['total_bytes'] // 1024}KB total)")
        if cstats.get('error'):
            self.log.error(f"({bay}/{direction}) collection aborted: {cstats['error']}")
        elif cstats['total_boxes_detected'] == 0:
            self.log.warning(
                f"({bay}/{direction}) model returned zero boxes across all "
                f"{cstats['frames_read']} frames -- check ROI placement, "
                f"model file, and camera framing (see debug/ images; "
                f"set alpr.diagnostics_mode='troubleshooting' for a full "
                f"per-frame dump)")

        reads = []
        for i, s in enumerate(samples, 1):
            cv2.imwrite(str(folder / f"crop_{i:02d}.jpg"), s['crop'])
            prep_path = debug_dir / f"prep_{i:02d}.jpg"
            plate, conf, valid, raw = self.ocr_image(s['crop'], prep_path)
            self.log.debug(
                f"({bay}/{direction}) sample {i}: raw='{raw}' fixed='{plate}' "
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
        self.log.info(f"({bay}/{direction}) {len(reads)} reads -> "
                       f"'{final}' (support {supporting_reads}, conf {confidence})")
        elapsed = round(time.time() - t0, 1)

        status = ERROR_STATUS.get(cstats.get('error'))
        if status is None:
            # weighted_vote() falls back to voting among invalid reads
            # when nothing valid exists (a best-effort guess, e.g. OCR
            # fragments like "4551" or "SEDRAD" that never matched a
            # plate pattern) -- that vote must not count as SUCCESS just
            # because it's non-empty, or garbage gets published as a
            # "read" plate.
            status = 'SUCCESS' if final and is_valid(final) else 'NO_VALID_PLATE'

        # Full detail for the audit trail on disk.
        result = {
            'bay': bay,
            'direction': direction,
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

        # Lean payload for the MQTT reply -- just what a downstream gate/
        # dock system needs (truck number, bay, which direction, and
        # enough to sanity-check the read) -- the full reads/collection
        # detail stays in result.json on disk for troubleshooting.
        reply = {
            'bay': bay,
            'direction': direction,
            'truck_number': final,
            'confidence': confidence,
            'status': status,
            'event_time': event_time,
            'ocr_time': result['ocr_time'],
        }

        if status == 'NO_VALID_PLATE' and not self.publish_no_valid_plate:
            self.log.info(f"({bay}/{direction}) no valid plate found, result "
                           f"saved to {folder} but not published "
                           f"({elapsed}s total, {len(reads)} reads)")
        else:
            topic = self._result_topic(direction, bay)
            self.publish(topic, json.dumps(reply, default=str))
            self.log.info(f"({bay}/{direction}) {final or status} published "
                           f"to {topic} ({elapsed}s total, {len(reads)} reads)")

    def collect(self, cam: dict, debug_dir=None):
        stats = {'first_fetch_ms': 0, 'avg_fetch_ms': 0.0, 'total_bytes': 0,
                  'frames_read': 0, 'raw_candidates': 0, 'total_boxes_detected': 0,
                  'rejected': {}, 'collect_sec': 0.0}

        try:
            url = build_snapshot_url(cam, self.config)
        except SnapshotError as e:
            self.log.error(f"({cam.get('ip', '?')}) snapshot config error: {e}")
            stats['error'] = 'snapshot_config_error'
            return [], stats

        self.log.debug(f"polling {url} "
                        f"(connect_timeout={self.connect_timeout_ms}ms, "
                        f"read_timeout={self.read_timeout_ms}ms, "
                        f"diagnostics_mode={'troubleshooting' if self.troubleshooting else 'basic'})")

        cands = []
        candidate_no = 0
        rejected = Counter()
        start = time.time()
        consecutive_failures = 0
        fetch_ms_total = 0

        while time.time() - start < self.collection_timeout:
            try:
                frame, fetch_ms, size_bytes = fetch_snapshot(
                    self.session, url, self.auth,
                    self.connect_timeout_ms, self.read_timeout_ms)
            except SnapshotError as e:
                rejected['fetch_fail'] += 1
                consecutive_failures += 1
                self.log.debug(f"({cam.get('ip', '?')}) snapshot fetch failed: {e}")
                if consecutive_failures >= MAX_CONSECUTIVE_FETCH_FAILURES:
                    self.log.error(
                        f"({cam.get('ip', '?')}) {consecutive_failures} "
                        f"consecutive snapshot failures, aborting collection")
                    if not stats['frames_read']:
                        stats['error'] = 'no_frame_received'
                    break
                time.sleep(FETCH_FAILURE_BACKOFF_SEC)
                continue
            consecutive_failures = 0
            stats['frames_read'] += 1
            frame_no = stats['frames_read']
            fetch_ms_total += fetch_ms
            stats['total_bytes'] += size_bytes
            self.log.debug(f"({cam.get('ip', '?')}) fetched frame "
                            f"{frame_no} in {fetch_ms}ms ({size_bytes // 1024}KB)")

            if frame_no == 1:
                stats['first_fetch_ms'] = fetch_ms
                ok_size, actual_w, actual_h = check_frame_size(
                    frame, self.expected_frame_width, self.expected_frame_height,
                    self.frame_size_tolerance_pct)
                stats['frame_size'] = f"{actual_w}x{actual_h}"
                if not ok_size:
                    self.log.error(
                        f"({cam.get('ip', '?')}) frame size {actual_w}x{actual_h} "
                        f"does not match expected "
                        f"{self.expected_frame_width}x{self.expected_frame_height} "
                        f"(tolerance {self.frame_size_tolerance_pct}%)")
                    stats['error'] = 'frame_size_mismatch'
                    break
                self.log.debug(f"({cam.get('ip', '?')}) frame size OK: "
                                f"{actual_w}x{actual_h}")
                if debug_dir:
                    save_debug_image(debug_dir, "00_first_frame.jpg", frame, self.log)
            elif debug_dir and self.troubleshooting:
                # "all collected images" -- every fetched frame, not just
                # the first, only when actively troubleshooting.
                save_debug_image(debug_dir, f"frame_{frame_no:02d}_full.jpg",
                                  frame, self.log)

            x1r, y1r, x2r, y2r = cam['roi']
            roi = frame[y1r:y2r, x1r:x2r]
            if roi.size == 0:
                rejected['empty_roi'] += 1
                continue
            center = roi.shape[1] / 2
            # Annotated copy of this frame's ROI: always built for frame 1
            # (so there's *something* to look at even with zero
            # detections -- previously this was only ever created if a
            # box was kept, so a fully-empty result left no ROI image at
            # all) and, in troubleshooting mode, for every frame.
            annotated = roi.copy() if (frame_no == 1 or self.troubleshooting) else None

            results = self.model(roi, verbose=False)
            boxes_this_frame = 0
            for r in results:
                for b in r.boxes:
                    boxes_this_frame += 1
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                    w = bx2 - bx1; h = by2 - by1

                    reason = None
                    if w < self.min_plate_width or h < self.min_plate_height:
                        reason = 'too_small'
                    elif ((by1 + by2) / 2) < roi.shape[0] * self.upper_half_fraction:
                        reason = 'upper_half'
                    else:
                        dist = abs((bx1 + bx2) / 2 - center)
                        if dist > self.center_distance_limit:
                            reason = 'off_center'

                    if reason:
                        rejected[reason] += 1
                        if annotated is not None and self.troubleshooting:
                            cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                                          (0, 0, 255), 1)
                            cv2.putText(annotated, reason, (bx1, by1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 1)
                        continue

                    # Pad the crop beyond the detected box so a slightly
                    # undersized box (common cause of a plate getting
                    # clipped/split across two adjacent detections)
                    # doesn't cut off a character at the edge.
                    pad_x = int(w * self.plate_crop_padding_pct / 100)
                    pad_y = int(h * self.plate_crop_padding_pct / 100)
                    px1 = max(0, bx1 - pad_x)
                    py1 = max(0, by1 - pad_y)
                    px2 = min(roi.shape[1], bx2 + pad_x)
                    py2 = min(roi.shape[0], by2 + pad_y)
                    crop = roi[py1:py2, px1:px2]
                    if any(duplicate(crop, c['crop']) for c in cands):
                        rejected['duplicate'] += 1
                        if annotated is not None and self.troubleshooting:
                            cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                                          (0, 165, 255), 1)
                            cv2.putText(annotated, 'duplicate', (bx1, by1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 165, 255), 1)
                        continue

                    area = min(crop.shape[0] * crop.shape[1] / 25000, 1.0)
                    shp = min(sharpness(crop) / 600, 1.0)
                    sc = (float(b.conf[0]) * 0.4 + area * 0.25 +
                          shp * 0.2 + (1 - dist / center) * 0.15)
                    cands.append({'crop': crop, 'score': sc})
                    candidate_no += 1
                    self.log.debug(
                        f"candidate kept: frame={frame_no} "
                        f"box=({bx1},{by1},{bx2},{by2}) "
                        f"score={sc:.2f} yolo_conf={float(b.conf[0]):.2f}")
                    if debug_dir and self.troubleshooting:
                        # "after crop" -- every raw candidate crop, saved
                        # the moment it's found, not just the final
                        # best_samples subset written later in handle().
                        save_debug_image(
                            debug_dir,
                            f"candidate_{candidate_no:02d}_score{sc:.2f}.jpg",
                            crop, self.log)
                    if annotated is not None:
                        cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                                      (0, 255, 0), 2)
                        cv2.putText(annotated, f"{sc:.2f}", (bx1, by1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                    if len(cands) >= self.max_raw_samples:
                        break
                if len(cands) >= self.max_raw_samples:
                    break

            stats['total_boxes_detected'] += boxes_this_frame
            self.log.debug(f"({cam.get('ip', '?')}) frame {frame_no}: "
                            f"{boxes_this_frame} boxes from model, "
                            f"{len(cands)} kept so far")

            if annotated is not None and debug_dir:
                name = ("01_roi_first.jpg" if frame_no == 1
                        else f"frame_{frame_no:02d}_roi.jpg")
                save_debug_image(debug_dir, name, annotated, self.log)

            if len(cands) >= self.max_raw_samples:
                break

            if self.poll_interval_ms:
                time.sleep(self.poll_interval_ms / 1000)

        stats['raw_candidates'] = len(cands)
        stats['rejected'] = dict(rejected)
        stats['collect_sec'] = round(time.time() - start, 1)
        if stats['frames_read']:
            stats['avg_fetch_ms'] = round(fetch_ms_total / stats['frames_read'], 1)
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
