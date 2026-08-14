"""Per-job pipeline: capture -> detect -> OCR -> vote -> publish.

One Worker thread per config "service.num_workers", each owning its own
YOLO + PaddleOCR model instances (loaded once at thread start) so
concurrent jobs don't share model state across threads.
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

from .image_ops import (prep, sharpness, thumbnail, duplicate_thumbs,
                         save_debug_image, check_frame_size)
from .logging_setup import get_logger
from .results import build_reply, result_topic, should_publish
from .plate_text import is_valid, fix_indian_plate
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


def ensure_cls_placeholder(cls_dir: Path, log):
    """PaddleOCR's constructor unconditionally checks cls_model_dir for
    inference.pdiparams/inference.pdmodel and downloads the real cls
    (angle classifier) model .tar over the network if either is missing
    -- regardless of use_angle_cls (confirmed from paddleocr==2.7.3's own
    source: that check runs before use_angle_cls is even consulted). But
    the classifier object itself -- the only thing that ever actually
    reads those files as model data -- is instantiated in
    tools/infer/predict_system.py's TextSystem.__init__ ONLY when
    self.use_angle_cls is True. We always pass use_angle_cls=False (see
    Worker.run), so with that confirmed, these two files are never read
    as real model weights: only their EXISTENCE matters, to satisfy the
    unconditional check and skip the download, not their content.

    So rather than requiring every deployment to source a real cls model
    (or fail here on any network hiccup reaching paddleocr.bj.bcebos.com
    -- confirmed in the field behind a corporate proxy that blocks the
    download's HTTPS CONNECT tunnel, taking every worker thread down with
    it), create empty placeholders here if they don't already exist. A
    real cls model already present is left untouched -- this only fills
    the gap when there's nothing there at all."""
    for name in ("inference.pdiparams", "inference.pdmodel"):
        f = cls_dir / name
        if not f.exists():
            f.write_bytes(b"")
            log.info(f"created empty cls placeholder: {f} (cls is never "
                      f"used since use_angle_cls=False -- see "
                      f"ensure_cls_placeholder's docstring)")


class Worker(threading.Thread):
    def __init__(self, wid, jobs, cameras: dict, config: dict, publish_fn,
                 job_bus, audit_dir: Path, model_load_lock: threading.Lock = None,
                 on_result=None):
        super().__init__(daemon=True, name=f"worker-{wid}")
        self.wid = wid
        self.jobs = jobs
        self.cameras = cameras
        self.config = config
        self.publish = publish_fn
        self.job_bus = job_bus
        self.audit_dir = audit_dir
        # Optional hook invoked with the reply dict after every completed
        # job (success, NO_VALID_PLATE, or an error result), independent
        # of whether it was actually published to MQTT (publish_no_valid_
        # plate can suppress that, but a consumer of this hook -- e.g.
        # bay_state.py's BayStateEngine -- still needs to know a read
        # attempt happened and how it went). Never lets a hook exception
        # take down job processing.
        self.on_result = on_result
        # Shared across every Worker: all workers point at the same local
        # PaddleOCR model folder, and on a fresh install nothing's
        # downloaded there yet. Without this lock, every worker thread
        # hits PaddleOCR's download-if-missing check at once and they all
        # race to fetch + extract the same .tar into the same path --
        # confirmed in the field: three concurrent downloads to one
        # target file, any of which could corrupt/truncate what another
        # was still writing. Serializing means only the first worker
        # downloads; the rest find the files already there and just load.
        self.model_load_lock = model_load_lock or threading.Lock()
        self.model = None
        self.ocr = None
        self.session = None
        self.auth = None
        self.log = get_logger(f"WORKER-{wid}")

        # Subscripted, not .get(key, default): load_config() applies every
        # default in one place, so repeating the literals here would give
        # each setting two homes that can silently disagree. They already
        # had -- publish_no_valid_plate defaulted True in config.py and
        # False here, and only the config.py one was ever reachable.
        alpr = config["alpr"]
        self.collection_timeout = alpr["collection_timeout"]
        self.max_ocr_attempts = alpr["max_ocr_attempts"]
        self.min_plate_width = alpr["min_plate_width"]
        self.min_plate_height = alpr["min_plate_height"]
        self.center_distance_limit = alpr["center_distance_limit"]
        self.upper_half_fraction = alpr["upper_half_fraction"]
        self.plate_crop_padding_pct = alpr["plate_crop_padding_pct"]
        self.troubleshooting = alpr["diagnostics_mode"] == "troubleshooting"
        self.publish_no_valid_plate = alpr["publish_no_valid_plate"]
        self.expected_frame_width = alpr["expected_frame_width"]
        self.expected_frame_height = alpr["expected_frame_height"]
        self.frame_size_tolerance_pct = alpr["frame_size_tolerance_pct"]
        self.min_ocr_conf = alpr["min_ocr_conf"]
        self.unknown_plate_value = alpr["unknown_plate_value"]
        self.save_detected_plate_frames = alpr["save_detected_plate_frames"]
        self.save_all_attempt_frames = alpr["save_all_attempt_frames"]
        self.yolo_conf_threshold = alpr["yolo_conf_threshold"]
        self.max_consecutive_fetch_failures = alpr["max_consecutive_fetch_failures"]
        self.fetch_failure_backoff_sec = alpr["fetch_failure_backoff_sec"]
        self.score_weights = alpr["score_weights"]
        self.score_area_norm = alpr["score_area_norm"]
        self.score_sharpness_norm = alpr["score_sharpness_norm"]
        self.duplicate_resize = (alpr["duplicate_resize_width"],
                                  alpr["duplicate_resize_height"])
        self.duplicate_diff_threshold = alpr["duplicate_diff_threshold"]
        self.ocr_prep_target_height = alpr["ocr_prep_target_height"]
        self.ocr_prep_padding = alpr["ocr_prep_padding"]
        self.max_read_attempts = alpr["max_read_attempts"]
        snap_cfg = config["snapshot"]
        self.connect_timeout_ms = snap_cfg["connect_timeout_ms"]
        self.read_timeout_ms = snap_cfg["read_timeout_ms"]
        self.poll_interval_ms = snap_cfg["poll_interval_ms"]

    def run(self):
        self._load_models_with_retry()
        # One Session/HTTPDigestAuth per worker thread, reused across every
        # job it handles: a Session keeps the TCP connection (and, for
        # HTTPDigestAuth, the last nonce) alive across requests, so only
        # the first snapshot fetch to a given camera pays for a fresh
        # handshake -- not every single one.
        self.session = requests.Session()
        # Cameras are on the local/internal network; don't let a
        # corporate proxy configured via HTTP_PROXY/HTTPS_PROXY env vars
        # or Windows system settings intercept (or block) these requests
        # -- see the matching fix in bay_monitor.py for the confirmed
        # symptom (curl direct works, requests via the proxy doesn't).
        self.session.trust_env = False
        self.auth = build_auth(self.config)

        while True:
            job = self.jobs.get()
            bay = job['bay']
            direction = job.get('direction', '')
            try:
                if self.model is None or self.ocr is None:
                    # Every load attempt in _load_models_with_retry()
                    # failed. Rather than the thread having died silently
                    # (the previous behavior -- an uncaught exception in
                    # a Thread.run() just prints a traceback and vanishes,
                    # with no signal anywhere else that jobs will now
                    # queue forever), this worker stays alive and gives
                    # every job it's handed a real answer.
                    self._publish_error(job, "MODEL_LOAD_FAILED",
                                        self.model_load_error)
                else:
                    self.handle(job)
            except Exception:
                tb = traceback.format_exc()
                self.log.error(f"({bay}/{direction}) job failed unexpectedly:\n{tb}")
                self._publish_error(job, "WORKER_EXCEPTION", tb)
            finally:
                self.job_bus.release(bay, direction)
                self.jobs.task_done()

    def _load_models_with_retry(self):
        """Loads YOLO + PaddleOCR, retrying alpr.model_load_max_retries
        times (alpr.model_load_retry_backoff_sec apart) on any failure --
        a transient network blip fetching a model file shouldn't need a
        process restart to recover from. If every attempt fails, self.model
        and self.ocr are left None and self.model_load_error holds the
        last failure, so run()'s job loop can answer every job it's handed
        with a real MODEL_LOAD_FAILED result instead of the thread having
        silently died with an uncaught exception (the previous behavior --
        confirmed in the field: a proxy blocking PaddleOCR's cls-model
        download took out all three worker threads at once, one after
        another through the shared model_load_lock, with nothing published
        anywhere and jobs left to queue forever)."""
        self.log.info("loading models...")
        t = time.time()
        # Resolved against the directory config.json actually loaded
        # from (see config._config_dir) -- not BASE_DIR -- so both the
        # YOLO weights and PaddleOCR's det/rec/cls models live next to
        # config.json rather than assuming they're next to the code.
        config_dir = Path(self.config["_config_dir"])
        model_path = config_dir / self.config["model_path"]
        det_dir = config_dir / self.config["alpr"]["paddleocr_det_model_dir"]
        rec_dir = config_dir / self.config["alpr"]["paddleocr_rec_model_dir"]
        # cls (angle classifier): PaddleOCR downloads this unconditionally
        # at construction time even though use_angle_cls=False below means
        # it's never actually used for inference -- pointed at a local
        # folder same as det/rec so it doesn't fall back to ~/.paddleocr,
        # and backed by a harmless placeholder (see ensure_cls_placeholder)
        # so a machine with no route to paddleocr.bj.bcebos.com never
        # needs a real cls model at all.
        cls_dir = config_dir / self.config["alpr"]["paddleocr_cls_model_dir"]
        self.log.info(f"model_path={model_path} "
                       f"paddleocr det={det_dir} rec={rec_dir} cls={cls_dir}")

        max_retries = self.config["alpr"]["model_load_max_retries"]
        backoff_sec = self.config["alpr"]["model_load_retry_backoff_sec"]
        self.model_load_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Serialized across all workers -- see model_load_lock's
                # comment in __init__. Only matters for the one-time
                # download; once the files exist this lock is held only
                # as long as it takes each worker to load from disk.
                with self.model_load_lock:
                    model = YOLO(str(model_path))
                    det_dir.mkdir(parents=True, exist_ok=True)
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    ensure_cls_placeholder(cls_dir, self.log)
                    ocr = PaddleOCR(lang='en', use_angle_cls=False, show_log=False,
                                     det_model_dir=str(det_dir), rec_model_dir=str(rec_dir),
                                     cls_model_dir=str(cls_dir))
                self.model, self.ocr = model, ocr
                self.model_load_error = None  # a retry that eventually
                # succeeded must not leave a stale failure message behind
                self.log.info(f"ready ({round(time.time() - t, 1)}s, "
                               f"attempt {attempt}/{max_retries})")
                return
            except Exception as e:
                self.model_load_error = f"{type(e).__name__}: {e}"
                self.log.error(
                    f"model load attempt {attempt}/{max_retries} failed:\n"
                    f"{traceback.format_exc()}")
                if attempt < max_retries:
                    time.sleep(backoff_sec)

        self.log.error(
            f"giving up on model loading after {max_retries} attempts -- "
            f"this worker will stay alive and publish MODEL_LOAD_FAILED "
            f"for every job it receives instead of processing them. Fix "
            f"the underlying cause ({self.model_load_error}) and restart "
            f"the service.")

    def _publish_error(self, job, reason, detail=""):
        """Last-resort publish so a downstream consumer waiting on this
        bay's result topic gets a signal instead of hanging forever --
        used both for camera/config errors and unhandled exceptions."""
        bay = job.get('bay', 'unknown')
        direction = job.get('direction', '')
        result = build_reply(
            self.config, bay, direction, None, 0.0, 'ERROR',
            job.get('event_time'),
            error_reason=reason, error_detail=str(detail)[-2000:])
        try:
            topic = result_topic(self.config, direction, bay)
            self.publish(topic, json.dumps(result, default=str))
            self.log.error(f"({bay}/{direction}) published ERROR result: {reason}")
        except Exception:
            self.log.error(f"({bay}/{direction}) failed to publish ERROR result:\n"
                            f"{traceback.format_exc()}")
        self._notify_result(result)

    def _notify_result(self, reply: dict):
        if self.on_result is None:
            return
        try:
            self.on_result(reply)
        except Exception:
            self.log.error(f"on_result hook raised:\n{traceback.format_exc()}")

    def _save_detected_plate(self, bay, direction, ts, plate, reads, samples):
        """Copies the crop that actually produced the winning plate into
        a separate, flat audit/detected_plates/ folder -- so confirmed
        reads can be browsed at a glance, chronologically, without
        hunting through each job's own per-event audit subfolder. Only
        called on a SUCCESS result (see handle()); best-effort, never
        lets a save failure affect the job itself."""
        matches = [r for r in reads if r['plate'] == plate]
        if not matches:
            return
        best = max(matches, key=lambda r: r['conf'])
        crop = samples[best['sample'] - 1]['crop']
        try:
            out_dir = self.audit_dir / "detected_plates"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_plate = plate.replace('/', '_')
            cv2.imwrite(str(out_dir / f"{ts}_{bay}_{direction}_{safe_plate}.jpg"), crop)
        except Exception:
            self.log.warning(f"({bay}/{direction}) failed to save "
                             f"detected-plate frame", exc_info=True)

    def _save_attempt_frame(self, bay, direction, ts, status, plate, samples):
        """Every completed job's best-scoring attempted crop -- SUCCESS
        or not -- into one flat, chronologically browsable
        audit/all_attempts/ folder, named with the outcome baked into
        the filename. Unlike _save_detected_plate (SUCCESS only, for "which
        trucks did we confirm"), this is a "what is the camera actually
        seeing, why isn't a given bay reading" view: seeing a NO_VALID_PLATE
        run's actual crop (blurry? genuinely not a plate? wrong ROI?)
        without opening that job's own nested audit subfolder one at a
        time. Best-effort; never lets a save failure affect the job itself."""
        if not samples:
            return
        best = max(samples, key=lambda s: s['score'])
        try:
            out_dir = self.audit_dir / "all_attempts"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_plate = (plate or self.unknown_plate_value).replace('/', '_')
            cv2.imwrite(str(out_dir / f"{ts}_{bay}_{direction}_{status}_"
                                      f"{safe_plate}.jpg"), best['crop'])
        except Exception:
            self.log.warning(f"({bay}/{direction}) failed to save "
                             f"all_attempts frame", exc_info=True)

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

        reads, final, samples, cstats = self.collect_and_read(cam, debug_dir)
        for i, s in enumerate(samples, 1):
            cv2.imwrite(str(folder / f"crop_{i:02d}.jpg"), s['crop'])
        self.log.info(
            f"({bay}/{direction}) frames={cstats['frames_read']} "
            f"boxes_detected={cstats['total_boxes_detected']} "
            f"ocr_attempts={cstats['ocr_attempts']} "
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
            # collect_and_read() only ever sets `final` to a read that
            # already passed is_valid() -- it stops the instant one does,
            # and reports nothing (None) if the try budget runs out
            # first. No fallback "best guess" among invalid reads: a
            # single frame's OCR either matches the Indian-plate format
            # or it doesn't, and picking the least-wrong garbage read
            # once the budget is spent would risk publishing a fabricated
            # plate as if it were a real one.
            status = 'SUCCESS' if final else 'NO_VALID_PLATE'

        # Every result carries a truck_number string, never null: on
        # anything short of a valid read it's alpr.unknown_plate_value
        # ("UNKNOWN" by default), so a downstream consumer can always read
        # the field without null-handling. "status" is what distinguishes
        # a genuine read from the placeholder -- and since the placeholder
        # can never satisfy is_valid(), it can't be mistaken for one.
        truck_number = final if status == 'SUCCESS' else self.unknown_plate_value

        if status == 'SUCCESS' and self.save_detected_plate_frames:
            self._save_detected_plate(bay, direction, ts, final, reads, samples)
        if self.save_all_attempt_frames:
            self._save_attempt_frame(bay, direction, ts, status, truck_number, samples)

        # Full detail for the audit trail on disk. 'final_plate' is the
        # same value as truck_number on SUCCESS, and None otherwise --
        # unlike the old weighted-vote design there's no "best guess
        # among invalid reads" left to preserve here; the full attempt
        # history (raw OCR text per try, valid or not) is still in
        # 'reads' below for forensics.
        result = {
            'bay': bay,
            'direction': direction,
            'truck_number': truck_number,
            'final_plate': final,
            'confidence': confidence,
            'supporting_reads': supporting_reads,
            'total_reads': len(reads),
            'valid': final is not None,
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
        reply = build_reply(self.config, bay, direction, truck_number,
                             confidence, status, event_time,
                             ocr_time=result['ocr_time'])

        # bay_state_engine retries the SAME visit's plate read many times
        # (every classification, until success or alpr.max_read_attempts)
        # -- publishing every failed attempt would flood the enter topic
        # with NO_VALID_PLATE noise for one physical truck, when only the
        # eventual SUCCESS (or the final give-up) actually matters to a
        # downstream consumer. bay_state_engine's on_alpr_result hook
        # still sees every attempt via _notify_result below regardless of
        # whether it got published, so a plate found on attempt 7 is
        # captured exactly the same either way -- this only silences the
        # MQTT spam, not the retry logic itself. One-shot MQTT/HTTP
        # triggers (source=None) are unaffected: one trigger, one
        # attempt, always worth publishing.
        is_bay_state_retry_noise = (job.get('source') == 'bay_state'
                                     and status == 'NO_VALID_PLATE')
        if not should_publish(self.config, status) or is_bay_state_retry_noise:
            self.log.info(f"({bay}/{direction}) no valid plate found, result "
                           f"saved to {folder} but not published "
                           f"({elapsed}s total, {len(reads)} reads)")
        else:
            topic = result_topic(self.config, direction, bay)
            self.publish(topic, json.dumps(reply, default=str))
            self.log.info(f"({bay}/{direction}) {truck_number} ({status}) "
                           f"published to {topic} ({elapsed}s total, "
                           f"{len(reads)} reads)")

        # Fires regardless of whether publish_no_valid_plate suppressed
        # the MQTT publish above -- a consumer of this hook needs to know
        # a read attempt completed and how it went either way.
        self._notify_result(reply)

    def collect_and_read(self, cam: dict, debug_dir=None):
        """Fetch frames and run YOLO on every one -- but unlike the old
        collect-a-batch-then-vote design, OCR is no longer deferred to
        the end. The first detected plate box is the trigger to start
        OCRing immediately, best-scoring candidate first within each
        frame, continuing across subsequent frames until either one
        produces a plate that passes plate_text.is_valid() -- stopping
        the ENTIRE collection right there, no need to wait out the rest
        of collection_timeout -- or alpr.max_ocr_attempts runs out.

        No fallback vote across whatever was tried if nothing validates:
        a single read either matches the Indian-plate format or it
        doesn't, and there's no "best guess among garbage" worth
        publishing once the try budget is spent. Every individual
        attempt (valid or not) is still returned in `reads` for the
        audit trail.

        Returns (reads, final, samples, stats): `final` is the winning
        plate or None, `samples` is every crop actually OCR'd (in
        attempt order, 1-based via list position) for handle() to write
        to the audit folder and for _save_detected_plate() to pull the
        winning crop from."""
        stats = {'first_fetch_ms': 0, 'avg_fetch_ms': 0.0, 'total_bytes': 0,
                  'frames_read': 0, 'ocr_attempts': 0, 'total_boxes_detected': 0,
                  'rejected': {}, 'collect_sec': 0.0}

        try:
            url = build_snapshot_url(cam, self.config)
        except SnapshotError as e:
            self.log.error(f"({cam.get('ip', '?')}) snapshot config error: {e}")
            stats['error'] = 'snapshot_config_error'
            return [], None, [], stats

        self.log.debug(f"polling {url} "
                        f"(connect_timeout={self.connect_timeout_ms}ms, "
                        f"read_timeout={self.read_timeout_ms}ms, "
                        f"diagnostics_mode={'troubleshooting' if self.troubleshooting else 'basic'})")

        samples = []       # every crop actually OCR'd, in attempt order
        tried_thumbs = []  # dedupe against every crop already OCR'd
        reads = []
        final = None
        candidate_no = 0
        rejected = Counter()
        start = time.time()
        consecutive_failures = 0
        fetch_ms_total = 0

        while (time.time() - start < self.collection_timeout
               and stats['ocr_attempts'] < self.max_ocr_attempts):
            try:
                frame, fetch_ms, size_bytes = fetch_snapshot(
                    self.session, url, self.auth,
                    self.connect_timeout_ms, self.read_timeout_ms)
            except SnapshotError as e:
                rejected['fetch_fail'] += 1
                consecutive_failures += 1
                self.log.debug(f"({cam.get('ip', '?')}) snapshot fetch failed: {e}")
                if consecutive_failures >= self.max_consecutive_fetch_failures:
                    self.log.error(
                        f"({cam.get('ip', '?')}) {consecutive_failures} "
                        f"consecutive snapshot failures, aborting collection")
                    if not stats['frames_read']:
                        stats['error'] = 'no_frame_received'
                    break
                time.sleep(self.fetch_failure_backoff_sec)
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

            results = self.model(roi, conf=self.yolo_conf_threshold, verbose=False)
            boxes_this_frame = 0
            frame_cands = []
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
                    # Thumbnail this crop once, then compare against every
                    # already-attempted candidate's cached thumbnail --
                    # re-OCRing pixels that already failed (or would
                    # produce the exact same read again) wastes a try.
                    crop_thumb = thumbnail(crop, self.duplicate_resize)
                    if any(duplicate_thumbs(crop_thumb, t,
                                             self.duplicate_diff_threshold)
                           for t in tried_thumbs):
                        rejected['duplicate'] += 1
                        if annotated is not None and self.troubleshooting:
                            cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                                          (0, 165, 255), 1)
                            cv2.putText(annotated, 'duplicate', (bx1, by1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 165, 255), 1)
                        continue

                    w_cfg = self.score_weights
                    area = min(crop.shape[0] * crop.shape[1] / self.score_area_norm, 1.0)
                    shp = min(sharpness(crop) / self.score_sharpness_norm, 1.0)
                    sc = (float(b.conf[0]) * w_cfg["yolo_conf"] +
                          area * w_cfg["area"] + shp * w_cfg["sharpness"] +
                          (1 - dist / center) * w_cfg["center"])
                    candidate_no += 1
                    self.log.debug(
                        f"candidate kept: frame={frame_no} "
                        f"box=({bx1},{by1},{bx2},{by2}) "
                        f"score={sc:.2f} yolo_conf={float(b.conf[0]):.2f}")
                    if debug_dir and self.troubleshooting:
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
                    frame_cands.append({'crop': crop, 'thumb': crop_thumb, 'score': sc})

            stats['total_boxes_detected'] += boxes_this_frame
            self.log.debug(f"({cam.get('ip', '?')}) frame {frame_no}: "
                            f"{boxes_this_frame} boxes from model, "
                            f"{len(frame_cands)} kept")

            if annotated is not None and debug_dir:
                name = ("01_roi_first.jpg" if frame_no == 1
                        else f"frame_{frame_no:02d}_roi.jpg")
                save_debug_image(debug_dir, name, annotated, self.log)

            # This is the actual "plate detected -> start reading"
            # trigger: OCR this frame's kept candidates right now,
            # best-scoring first, instead of only scoring them for a
            # batch vote later. Stops the whole collection the instant
            # one produces a valid plate.
            frame_cands.sort(key=lambda c: c['score'], reverse=True)
            for c in frame_cands:
                if stats['ocr_attempts'] >= self.max_ocr_attempts:
                    break
                tried_thumbs.append(c['thumb'])
                stats['ocr_attempts'] += 1
                i = len(samples) + 1
                samples.append(c)
                prep_path = debug_dir / f"prep_{i:02d}.jpg" if debug_dir else None
                plate, conf, valid, raw = self.ocr_image(c['crop'], prep_path)
                self.log.debug(
                    f"({cam.get('ip', '?')}) attempt {i}: raw='{raw}' "
                    f"fixed='{plate}' conf={conf:.2f} valid={valid} "
                    f"score={c['score']:.2f}")
                if plate:
                    reads.append({'plate': plate, 'conf': round(conf, 3),
                                  'valid': valid, 'raw': raw, 'sample': i})
                if valid:
                    final = plate
                    break

            if final:
                break

            if self.poll_interval_ms:
                time.sleep(self.poll_interval_ms / 1000)

        stats['rejected'] = dict(rejected)
        stats['collect_sec'] = round(time.time() - start, 1)
        if stats['frames_read']:
            stats['avg_fetch_ms'] = round(fetch_ms_total / stats['frames_read'], 1)
        return reads, final, samples, stats

    def ocr_image(self, img, debug_path=None):
        pimg = prep(img, self.ocr_prep_target_height, self.ocr_prep_padding)
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
