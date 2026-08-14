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
#      the NUM_WORKERS threads is serialized with a shared lock so a
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
#       actual locally-pulled model tag.
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
