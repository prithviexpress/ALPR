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
#
# New dependency: the `requests` package (HTTP snapshot fetch + digest
# auth) -- `pip install requests` on any machine that didn't already have
# it for other reasons.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from alpr_service.service import main

if __name__ == "__main__":
    main()
