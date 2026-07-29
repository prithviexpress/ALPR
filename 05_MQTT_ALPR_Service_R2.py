#!/usr/bin/env python3
# 05_MQTT_ALPR_Service_R2.py -- entry point. Implementation now lives in
# alpr_service/ (config, logging, cameras, rtsp, plate_text, image_ops,
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
#   2. RTSP capture hardened: explicit connect/read socket timeout (R2 had
#      none, so a dead camera could hang a worker thread indefinitely);
#      cap.release() on open failure (R2 leaked the handle); read failures
#      now back off and abort after 30 consecutive misses instead of
#      busy-spinning for the full collection window.
#   3. A camera/stream failure (RTSP unreachable, config error, or a wrong
#      first-frame resolution) is now reported as a distinct MQTT status
#      (CAMERA_UNREACHABLE / CAMERA_CONFIG_ERROR / FRAME_SIZE_ERROR)
#      instead of being indistinguishable from a normal NO_VALID_PLATE
#      miss. An unhandled worker exception now also publishes an ERROR
#      result instead of silently dropping the job.
#   4. First captured frame's resolution is checked against config
#      "alpr.expected_frame_width/height" (e.g. a 5MP camera); a mismatch
#      aborts the job as FRAME_SIZE_ERROR -- catches a misconfigured RTSP
#      URL pointed at a low-res substream before wasting a collection
#      cycle on frames that could never hold a readable plate.
#   5. RTSP source is now selectable per camera (config "rtsp.mode",
#      per-camera override cameras.json "rtsp_mode"): "genetec" (GUID via
#      the Genetec media gateway, R1/R2 behavior) or "direct" (straight to
#      the camera's own RTSP server via IP + username/password). Both
#      modes always request the camera's primary/main stream (stream 1,
#      full resolution), never a substream.
#   6. Initial MQTT connect now retries with exponential backoff instead
#      of crashing the process if the broker isn't reachable yet at boot;
#      SIGTERM (not just Ctrl+C/SIGINT) now triggers a clean shutdown.
#   7. config.json/cameras.json/best.pt are resolved relative to this
#      file's directory, not the process's current working directory.
#   8. Incoming MQTT trigger events are now filtered by their VCA
#      classification before anything else runs: only events with a
#      detection (Data.Object.Object[].Appearance.Class.Type[]) matching
#      config "event_filter.class_types" (default ["Vehicle"]) at or above
#      "event_filter.min_likelihood" are queued for ALPR. Everything else
#      (Person/Bicycle detections, low-confidence hits, events with no
#      classified object) is discarded before it ever opens an RTSP stream
#      -- R2 queued every line-crossing event regardless of what triggered it.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from alpr_service.service import main

if __name__ == "__main__":
    main()
