#!/usr/bin/env python3
# 05_MQTT_ALPR_Service_R2.py -- entry point. Implementation now lives in
# alpr_service/ (config, logging, cameras, snapshot, plate_text, image_ops,
# audit, mqtt_bus, worker, service), split so each concern can be edited
# and tested independently instead of one 570-line file. This filename is
# kept as-is so existing deployment/service configs that invoke
# `python 05_MQTT_ALPR_Service_R2.py` keep working unchanged.
#
# Changes vs R2 (see alpr_service/ for the actual diffs):
#   1. Logging switched from print() to the stdlib `logging` module, with
#      level selectable via config.json "logging.level" (DEBUG/INFO/
#      WARNING/ERROR) and an optional rotating log file -- no code change
#      needed to get verbose troubleshooting output in the field.
#   2. Capture is now HTTP snapshot polling (a single GET to each camera's
#      /snap.jpg per sample, HTTP Digest auth, config "snapshot") instead
#      of RTSP/Genetec entirely -- Genetec support has been removed. This
#      fits the trigger pattern here (trucks crawling in slowly, not fast
#      drive-bys) without needing RTSP's much higher frame rate, and
#      avoids the whole class of RTSP-stream-hang/leak failure modes.
#      Every camera uses one common username/password (config "snapshot.
#      username"/"password"); cameras.json only holds {ip, roi, enabled}.
#      A worker keeps one requests.Session + HTTPDigestAuth for its
#      lifetime so only the first fetch per camera pays for the digest
#      handshake, not every one.
#   3. A camera/config failure (snapshot endpoint unreachable, missing
#      credentials, or a wrong first-frame resolution) is now reported as
#      a distinct MQTT status (CAMERA_UNREACHABLE / CAMERA_CONFIG_ERROR /
#      FRAME_SIZE_ERROR) instead of being indistinguishable from a normal
#      NO_VALID_PLATE miss. An unhandled worker exception now also
#      publishes an ERROR result instead of silently dropping the job.
#   4. First fetched frame's resolution is checked against config
#      "alpr.expected_frame_width/height" (e.g. a 5MP camera); a mismatch
#      aborts the job as FRAME_SIZE_ERROR before wasting a collection
#      cycle on frames that could never hold a readable plate.
#   5. Initial MQTT connect now retries with exponential backoff instead
#      of crashing the process if the broker isn't reachable yet at boot;
#      SIGTERM (not just Ctrl+C/SIGINT) now triggers a clean shutdown.
#   6. config.json/cameras.json/best.pt are resolved relative to this
#      file's directory, not the process's current working directory.
#   7. Incoming MQTT trigger events are now filtered by their VCA
#      classification before anything else runs: only events with a
#      detection (Data.Object.Object[].Appearance.Class.Type[]) matching
#      config "event_filter.class_types" (default ["Vehicle"]) at or above
#      "event_filter.min_likelihood" are queued for ALPR. Everything else
#      (Person/Bicycle detections, low-confidence hits, events with no
#      classified object) is discarded before a single snapshot is
#      fetched -- R2 queued every line-crossing event regardless of what
#      triggered it.
#   8. Entering and leaving are now two independent triggers, confirmed
#      against real captured events for the same tracked object: enter
#      matches config "mqtt.enter_subscribe_topic" (LineDetector/Crossed
#      rules -- covers both "Crossing line N" and "Entering field N"),
#      leave matches "mqtt.leave_subscribe_topic" (ObjectTrack/Aggregation
#      rules -- "Leaving field N"). Each publishes a lean
#      {bay, direction, truck_number, confidence, status, event_time}
#      reply to its own topic (mqtt.enter_result_topic_prefix /
#      leave_result_topic_prefix + "/<bay>"); the full per-read detail
#      still goes to the audit folder's result.json regardless. Cooldown/
#      dedup is now tracked per (bay, direction), so an enter trigger's
#      cooldown never blocks a leave trigger for the same bay. The
#      Vehicle-class event_filter applies to both directions identically.
#   9. YOLO weights and all three PaddleOCR model folders (det/rec/cls)
#      are now resolved relative to the directory config.json actually
#      loaded from (config "model_path", "alpr.paddleocr_det_model_dir",
#      "alpr.paddleocr_rec_model_dir", "alpr.paddleocr_cls_model_dir"),
#      instead of PaddleOCR's own default of caching to ~/.paddleocr.
#      This includes the cls (angle classifier) model, which PaddleOCR
#      downloads unconditionally at construction time even with
#      use_angle_cls=False -- without cls_model_dir set explicitly it
#      was still going to ~/.paddleocr unnoticed. Model loading across
#      the worker threads (service.num_workers) is serialized with a shared lock so a
#      fresh install doesn't race multiple workers into downloading the
#      same files into the same folder at once. Also fixed BASE_DIR to
#      resolve from sys.executable (not Path(__file__)) when frozen by
#      PyInstaller, so a packaged .exe finds config.json/best.pt/
#      paddleocr_models sitting next to it rather than inside
#      PyInstaller's internal temp extraction folder.
#   10. Added a second, optional trigger source: config "http_trigger"
#       runs an HTTP server a camera's own alarm task can call directly
#       (e.g. a Bosch dome's built-in "HTTP notification" alarm task
#       GET-ing a URL with a numeric rule id) instead of going through
#       Genetec/MQTT for the trigger. It can run alongside the MQTT
#       trigger or replace it (config "mqtt.trigger_enabled"); MQTT is
#       still connected and used to publish results either way. The
#       calling camera is matched to a bay purely by its source IP against
#       cameras.json's "ip" field (same field the snapshot fetcher already
#       uses), and enter/exit is decided by "http_trigger.enter_rule_codes"
#       / "exit_rule_codes" -- everything downstream (collection, OCR,
#       audit, per-direction result topics) is identical to an MQTT-
#       triggered job; only the trigger's entry point differs. Served by
#       waitress (a production WSGI server, not Flask's own dev server);
#       adds a GET /healthz liveness endpoint and logs unhandled request
#       errors as JSON 500s instead of Flask's default HTML error page.
#       waitress's own log output is routed through the same logging
#       setup as the rest of the service (see logging_setup.py).
#   11. Added an entirely separate, optional feature: config "bay_monitor"
#       (alpr_service/bay_monitor.py) continuously round-robins every
#       enabled camera looking for presence, independent of the enter/
#       leave trigger pipeline -- it never touches JobBus, Worker, or the
#       MQTT/HTTP trigger sources above. Presence reuses the existing
#       ALPR plate model as a cheap signal (any detected box, no
#       filtering). Once a bay shows a detection it's "zoomed in": every
#       "bay_monitor.classify_interval_sec" (default 60s) a frame is sent
#       to a local Ollama-hosted vision model and the reply (empty /
#       occupied / unloading / loading / idle, or a custom set via
#       "bay_monitor.status_values") is published to
#       "mqtt.bay_status_topic_prefix" + "/<bay>". After
#       "bay_monitor.empty_debounce_count" consecutive "empty" replies it
#       reverts to baseline scanning. Off by default; fails fast at
#       startup if enabled without "bay_monitor.ollama_model" set to an
#       actual locally-pulled model tag. Also supports few-shot exemplar
#       images ("bay_monitor.reference_images") sent alongside every
#       classification call to calibrate ambiguous cases (e.g. a bay/
#       cargo door being open) that a text-only prompt undersells.
#   12. Added a second, optional, opt-in feature on top of #11: config
#       "bay_state_engine" (alpr_service/bay_state.py) fuses bay_monitor's
#       status stream with ALPR reads into one session per bay, and
#       becomes the authority for enter/leave INSTEAD of the MQTT/HTTP
#       trigger source. bay_monitor's own empty->occupied transition
#       enqueues an ALPR read via the normal Worker pool (Worker now
#       takes an optional "on_result" hook, called after every completed
#       job so this engine can capture a confirmed plate); unlike a
#       one-shot trigger it keeps retrying on every subsequent
#       reclassification of the bay as still occupied until a read
#       actually succeeds -- so one bad read (e.g. from a door being
#       open at that exact instant) doesn't cost the whole visit's
#       identity. occupied->empty publishes the leave result using
#       whatever plate got confirmed at any point during the stay, since
#       by the time a bay reads "empty" there's no truck left in frame
#       for a fresh read. Departure timing is driven directly by
#       bay_monitor's own empty_debounce_count decision (bay_monitor.py's
#       on_status hook now also passes its post-transition zoomed_in
#       flag) rather than a second, separately-configured debounce here,
#       since two independently-tuned copies of that threshold could
#       fall out of sync. Off by default; fails fast at startup if
#       enabled without bay_monitor also enabled, and warns (without
#       blocking) if the MQTT/HTTP triggers are also left on, since both
#       would then independently decide enter/leave for the same bays.
#       State is in-memory only -- a restart loses any truck mid-visit.
#   13. Every published result now carries a truck_number STRING, never
#       null: anything short of a valid read reports
#       "alpr.unknown_plate_value" ("UNKNOWN" by default) instead, so a
#       downstream consumer never has to null-check the field. "status"
#       remains what distinguishes a genuine read from the placeholder,
#       and the placeholder can never satisfy plate_text.is_valid() nor
#       be stored as a session's confirmed plate. The audit result.json
#       also gained "raw_vote" (whatever the OCR vote actually produced,
#       possibly null/garbage) so the placeholder hides no forensic detail.
#   14. bay_monitor's baseline presence check (config "bay_monitor.
#       presence_diff_enabled", on by default) now skips the YOLO pass
#       entirely on a bay whose ROI hasn't meaningfully changed since the
#       last real check -- a cheap thumbnail diff (~100us) stands in for
#       it, roughly three orders of magnitude cheaper than inference
#       (~100-300ms). Real detection still runs on any bay that changed
#       (an arrival moves far more pixels than "presence_diff_threshold"
#       tolerates, so this cannot miss one), and
#       "presence_diff_max_skip" forces a real check periodically
#       regardless, as a safety net against slow drift (e.g. gradually
#       shifting light) a single-frame diff would never trip. The cache
#       is cleared on both the arrival and departure transitions so it
#       never compares across a visit boundary.
#   15. Two fixes for a real field failure: a corporate proxy blocking
#       PaddleOCR's cls-model download took out all three worker
#       threads, one after another through the shared model_load_lock,
#       with nothing published anywhere and jobs left to queue forever
#       -- the process itself stayed up (MQTT connected, HTTP trigger
#       listening) while silently doing no work at all.
#       (a) Confirmed from paddleocr==2.7.3's own source: the classifier
#           object that actually READS the cls model files is only ever
#           instantiated when use_angle_cls=true, which this service
#           never passes -- so cls's file CONTENT was never the issue,
#           only PaddleOCR's unconditional existence check before it
#           even looks at use_angle_cls. worker.py's
#           ensure_cls_placeholder() now creates two empty placeholder
#           files there itself if nothing's present, so the download is
#           never attempted at all and no machine needs a real cls
#           model (a real one, if already present, is left untouched).
#       (b) Model loading (Worker._load_models_with_retry) now retries
#           on any failure ("alpr.model_load_max_retries", default 3,
#           "alpr.model_load_retry_backoff_sec" apart, default 5s), and
#           if every attempt still fails, that worker thread stays alive
#           rather than dying silently -- it publishes a
#           MODEL_LOAD_FAILED error result for every job it's handed
#           instead of processing them, so a downstream consumer gets a
#           signal on the result topic instead of the log file being the
#           only place the failure is visible.
#
# New dependencies: `requests` (HTTP snapshot fetch + digest auth),
# `flask` and `waitress` (only actually used if http_trigger.enabled is
# true) -- see requirements.txt for pinned versions. bay_monitor needs no
# new pip package (talks to Ollama over plain HTTP via `requests`), but
# does need a local Ollama install (https://ollama.com) with a
# vision-capable model pulled if bay_monitor.enabled is turned on.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from alpr_service.service import main

if __name__ == "__main__":
    main()
