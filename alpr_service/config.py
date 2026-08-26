"""Load and validate config.json, filling in defaults for optional keys.

Everything that used to be a module-level constant in the monolithic
script (cooldown, audit retention, snapshot timeouts, debug image toggle,
expected frame size, log level, ...) is a config key now, so ops can
retune the service without a code change/redeploy.
"""
import json
import sys
from pathlib import Path

# In a normal source checkout, this file lives in alpr_service/, so the
# repo root is one level up. But under a PyInstaller-frozen exe,
# Path(__file__) resolves inside PyInstaller's internal temp extraction
# folder, not the folder the .exe actually sits in -- sys.executable is
# the one thing that reliably points at the real exe location in both
# --onefile and --onedir builds. Getting this wrong means a frozen exe
# would look for config.json/best.pt/paddleocr_models in the wrong
# place even when they're sitting right next to it.
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"

REQUIRED_KEYS = {
    "mqtt": ["host", "port",
             "enter_subscribe_topic", "leave_subscribe_topic",
             "enter_result_topic_prefix", "leave_result_topic_prefix"],
    "alpr": ["collection_timeout", "max_ocr_attempts",
             "min_plate_width", "min_plate_height",
             "center_distance_limit"],
}


class ConfigError(RuntimeError):
    pass


def load_config(path: Path = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}. Copy config.example.json to "
            f"{path.name} and fill in your MQTT/snapshot/camera details.")
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"Config file {path} is not valid JSON: {e}") from e

    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            raise ConfigError(f"Config missing required section '{section}'")
        for k in keys:
            if k not in cfg[section]:
                raise ConfigError(f"Config missing required key '{section}.{k}'")

    # The directory config.json actually loaded from -- model_path and
    # the PaddleOCR model dirs below are resolved relative to THIS, not
    # BASE_DIR, so "next to config.json" holds even if config.json isn't
    # sitting next to the code (e.g. a future packaged .exe).
    cfg["_config_dir"] = str(path.resolve().parent)

    cfg.setdefault("model_path", "best.pt")

    # Entering and leaving are two independent triggers, confirmed against
    # real captured topics to have distinct, non-overlapping shapes:
    #   enter: "Camera_Events/<bay>/onvif-ej/RuleEngine/LineDetector/Crossed/&1/..."
    #   leave: "Camera_Events/<bay>/onvif-ej/RuleEngine/ObjectTrack/Aggregation/&1/..."
    # Each is its own MQTT subscription (mqtt.enter_subscribe_topic /
    # mqtt.leave_subscribe_topic) -- which rule/event types reach this
    # service at all is controlled by those topic filters (MQTT
    # wildcards), not by any code-side filtering. Results are published
    # to a direction-specific topic (mqtt.enter_result_topic_prefix /
    # mqtt.leave_result_topic_prefix + "/<bay>") so downstream consumers
    # can subscribe to entries and exits separately. Which topic segment
    # carries the bay name (default 1, "Camera_Events/<bay>/...") is
    # shared by both.
    mqtt = cfg["mqtt"]
    mqtt.setdefault("bay_segment_index", 1)
    # Whether the service subscribes to the VCA event topics at all --
    # false relies solely on http_trigger below for triggers. MQTT is
    # still connected and used to publish results either way; this only
    # controls the *subscribe* side.
    mqtt.setdefault("trigger_enabled", True)
    # Where bay_monitor publishes per-bay activity status ("/<bay>" is
    # appended). Lives here with the other topics rather than down in the
    # bay_monitor block, since it's part of the MQTT topic contract.
    mqtt.setdefault("bay_status_topic_prefix", "site/alpr/bay_status")
    # Where bay_state_engine publishes its LIVE per-bay session ("/<bay>"
    # appended) -- open/closed, direction, plate (if confirmed yet),
    # read_attempts -- on every state change (arrival, plate confirmed,
    # departure), not just the final enter/leave result. That result
    # (mqtt.enter_result_topic_prefix/leave_result_topic_prefix) only
    # ever fires once, at the end of a visit; this is the running view
    # of what the engine is doing right now, e.g. for a dashboard or for
    # confirming the engine is alive/reacting at all while troubleshooting.
    mqtt.setdefault("bay_state_topic_prefix", "site/alpr/bay_state")
    # Where bay_state_engine publishes a RICH notification ("/<bay>"
    # appended) whenever bay_monitor's activity classification actually
    # CHANGES (idle->loading, empty->occupied, etc, not on every
    # classification the way bay_status_topic_prefix above does) --
    # {bay, occupancy_status, activity, truck_number, comment,
    # image_base64, timestamp}. occupancy_status is "occupied"/"empty"
    # (bay_monitor's own occupied/departed interpretation); activity is
    # the raw classification word (may equal occupancy_status, or be a
    # finer one like "loading"/"unloading"/"idle"); truck_number is
    # whatever this session's plate is confirmed as so far (or
    # alpr.unknown_plate_value); comment is the vision model's own
    # free-text description of what it sees; image_base64 is the exact
    # frame it was shown. Requires bay_state_engine.enabled (only it
    # tracks truck_number) -- bay_monitor alone has no truck identity to
    # report.
    mqtt.setdefault("bay_notification_topic_prefix", "site/alpr/bay_notification")
    # The lean, consumer-facing topic ("/<bay>" appended) and the only
    # one published by default: exactly the events a dock system needs --
    # a truck came in, which truck it is, it left -- at roughly three
    # messages per visit.
    #   arrived     {truck_number (usually the UNKNOWN placeholder this
    #               early -- the ALPR read takes seconds), door_state,
    #               alert}. alert is "door_open_on_arrival" when the
    #               truck backed in with its doors ALREADY open, else
    #               null.
    #   identified  {truck_number, confidence, door_state} -- the plate
    #               resolving for a truck whose arrival already went out
    #               unidentified. Not published if the plate was already
    #               known at arrival, so there's never a redundant pair.
    #   departed    {truck_number, door_state, duration_sec}
    # Requires bay_state_engine.enabled (only it tracks truck identity).
    mqtt.setdefault("bay_event_topic_prefix", "site/alpr/bay_event")
    # Which of the four bay topics actually publish. The three older ones
    # default OFF: between them they emit dozens of messages per visit --
    # bay_status on EVERY classification, bay_state on every session
    # transition, bay_notification on every activity change -- which
    # buries the handful of events a consumer actually cares about.
    # bay_event says the same thing in ~3 messages, and the on-demand
    # webhook (bay_monitor.snapshot_webhook_*) answers everything else on
    # request rather than by broadcasting it. Turn the older ones back on
    # individually if something is already consuming them.
    mqtt.setdefault("publish_bay_event", True)
    mqtt.setdefault("publish_bay_status", False)
    mqtt.setdefault("publish_bay_state", False)
    mqtt.setdefault("publish_bay_notification", False)

    # An alternative trigger source to the MQTT VCA events above: some
    # cameras (e.g. a Bosch dome's built-in "HTTP notification" alarm
    # task) can be configured to call this service directly over HTTP on
    # a rule trigger, instead of going through Genetec/MQTT for the
    # trigger itself. Can run alongside mqtt.trigger_enabled, or be the
    # only trigger source (set mqtt.trigger_enabled to false). The
    # camera identifies itself only by source IP, matched against
    # cameras.json's "ip" field to resolve the bay -- so every enabled
    # camera needs a unique ip for this to work. enter_rule_codes /
    # exit_rule_codes map the camera's numeric alarm rule id to a
    # direction; a rule id in neither list is logged and ignored.
    http_trigger = cfg.setdefault("http_trigger", {})
    http_trigger.setdefault("enabled", False)
    http_trigger.setdefault("host", "0.0.0.0")
    http_trigger.setdefault("port", 8080)
    http_trigger.setdefault("path", "/alert")
    http_trigger.setdefault("rule_param", "rule")
    http_trigger.setdefault("enter_rule_codes", [2])
    http_trigger.setdefault("exit_rule_codes", [3])
    # Served by waitress (a production WSGI server, not Flask's own dev
    # server) -- this is its worker thread pool size. The default is
    # generous for what this endpoint actually does (a handful of alarm
    # hits per truck, each just doing a dict lookup + queue put), no need
    # to tune unless a very large number of bays fire in the same instant.
    http_trigger.setdefault("threads", 4)

    # Incoming MQTT trigger events carry a VCA classification per detected
    # object (Data.Object.Object[].Appearance.Class.Type[]: "#text" is the
    # class name, "@Likelihood" its confidence, e.g. {"#text": "Vehicle",
    # "@Likelihood": 0.91}). Only events with a detection matching one of
    # class_types at or above min_likelihood get queued for ALPR; anything
    # else (Person/Bicycle detections, low-confidence hits, events with no
    # classified object at all) is discarded before a single snapshot is
    # fetched. Comparison is case-insensitive.
    event_filter = cfg.setdefault("event_filter", {})
    class_types = event_filter.get("class_types", ["Vehicle"])
    if isinstance(class_types, str):
        class_types = [class_types]
    event_filter["class_types"] = class_types
    event_filter.setdefault("min_likelihood", 0.7)

    alpr = cfg["alpr"]
    alpr.setdefault("cooldown_sec", 90)
    alpr.setdefault("audit_retention_days", 14)
    # "basic": save the first fetched frame + its ROI crop, the selected
    #          plate crops, and the OCR-prep images (as before). Fine for
    #          normal operation.
    # "troubleshooting": all of the above, PLUS every fetched frame (full
    #          image and ROI, the ROI annotated with every box the model
    #          returned -- green=kept, red/orange=rejected with reason)
    #          and every kept candidate crop saved the moment it's found,
    #          not just the ones that got OCR'd. Also forces the log
    #          level to DEBUG regardless of "logging.level". Generates
    #          noticeably more files/log volume -- meant to be turned on
    #          only while actively diagnosing an issue (e.g. ocr_attempts
    #          staying at 0), then switched back to "basic".
    alpr.setdefault("diagnostics_mode", "basic")
    alpr.setdefault("min_ocr_conf", 0.35)
    # Published results always carry a truck_number STRING, never null --
    # anything short of a valid read reports this placeholder instead, so
    # a downstream consumer never has to null-check the field. The
    # "status" field (SUCCESS / NO_VALID_PLATE / CAMERA_* / ERROR) is what
    # distinguishes a genuine read from the placeholder. Keep this to
    # something no real plate could ever be (it must not satisfy
    # plate_text.is_valid(), or it could be mistaken for a real read).
    alpr.setdefault("unknown_plate_value", "UNKNOWN")
    # On every SUCCESS result, copies the crop that actually produced the
    # winning plate into a separate, flat audit/detected_plates/ folder
    # (named "<timestamp>_<bay>_<direction>_<plate>.jpg") -- so confirmed
    # reads can be browsed chronologically at a glance, without hunting
    # through each job's own per-event audit subfolder. The full per-job
    # detail (result.json, every candidate crop) is unaffected either way.
    alpr.setdefault("save_detected_plate_frames", True)
    # Every completed job's best-scoring attempted crop -- SUCCESS or
    # not -- into one flat audit/all_attempts/ folder (named
    # "<timestamp>_<bay>_<direction>_<status>_<plate-or-UNKNOWN>.jpg"),
    # so "what is the camera actually seeing, why is this bay coming
    # back NO_VALID_PLATE" can be answered by browsing one folder
    # chronologically instead of opening each job's own nested audit
    # subfolder one at a time. Separate from save_detected_plate_frames
    # above, which is SUCCESS-only and answers a different question
    # ("which trucks did we confirm").
    alpr.setdefault("save_all_attempt_frames", True)
    # Ceiling on how many ALPR reads bay_state_engine will queue for one
    # visit while the plate is still unconfirmed. Without a cap, a truck
    # parked for hours with an unreadable plate re-runs a full 8s
    # collection every cooldown window indefinitely -- each one occupying
    # one of only service.num_workers threads and re-photographing the
    # same obscured plate -- which starves genuinely new arrivals. Only
    # consulted by bay_state.py; the MQTT/HTTP trigger paths are
    # one-shot and unaffected.
    alpr.setdefault("max_read_attempts", 20)
    # Spend a RETRY read only when bay_monitor's truck model can actually
    # see a plate in frame (its Number_Plate class). A retry costs a full
    # collection window on one of only service.num_workers threads, and
    # the attempt budget above is per visit -- so a retry fired while the
    # plate is out of view isn't merely wasted, it makes it likelier the
    # budget runs out before the plate ever comes INTO view. Gating on
    # actual visibility is the point of the truck model reporting it.
    # Only has any effect with bay_monitor.classifier="yolo": the Ollama
    # backend reports no plate information at all, which is treated as
    # "go ahead" rather than "no plate", so this can't silently stop
    # retries for a deployment not running the truck model. Gates retries
    # ONLY -- the arrival read always fires, since entry is when a truck
    # is most likely facing the camera and the worker polls its own
    # frames across a whole collection window anyway.
    alpr.setdefault("retry_only_when_plate_visible", True)
    # How many times a Worker retries loading YOLO+PaddleOCR at startup
    # before giving up (backoff_sec apart) -- a transient failure (e.g. a
    # proxy blocking a one-time model download) shouldn't need a process
    # restart to recover from. If every attempt fails, that worker thread
    # stays alive rather than dying silently, and publishes
    # MODEL_LOAD_FAILED for every job it's handed instead of processing
    # them -- see Worker._load_models_with_retry.
    alpr.setdefault("model_load_max_retries", 3)
    alpr.setdefault("model_load_retry_backoff_sec", 5)
    # Minimum YOLO box confidence for a plate detection to be considered
    # at all (Ultralytics' own default is 0.25 if this isn't passed
    # explicitly -- surfaced here so it's tunable without a code change).
    alpr.setdefault("yolo_conf_threshold", 0.25)
    # Collection aborts after this many consecutive snapshot fetch
    # failures (reported as CAMERA_UNREACHABLE if no frame was ever
    # read), pausing this many seconds between retries.
    alpr.setdefault("max_consecutive_fetch_failures", 10)
    alpr.setdefault("fetch_failure_backoff_sec", 0.2)
    # Weights (should sum to ~1.0) for ranking multiple candidate crops
    # from the same collection window: yolo_conf (the model's own
    # confidence), area (bigger crop = more pixels for OCR, normalized by
    # score_area_norm), sharpness (Laplacian variance, normalized by
    # score_sharpness_norm), center (how close to the ROI's horizontal
    # center -- plates dead-center in frame tend to be the real target,
    # not a passing vehicle at the edge).
    # Merged key-by-key rather than setdefault'd as a whole dict: a
    # partial override like {"yolo_conf": 0.55} would otherwise leave the
    # other three keys absent, and worker.py subscripts all four -- every
    # job would die with a KeyError on the first kept box.
    score_weights = {"yolo_conf": 0.4, "area": 0.25, "sharpness": 0.2, "center": 0.15}
    score_weights.update(alpr.get("score_weights") or {})
    alpr["score_weights"] = score_weights
    alpr.setdefault("score_area_norm", 25000)
    alpr.setdefault("score_sharpness_norm", 600)
    # A candidate crop is discarded as a near-duplicate of one already
    # kept if, after both are resized to duplicate_resize_width x
    # _height, their mean pixel difference is below duplicate_diff_
    # threshold -- avoids voting on the same plate sighting twice just
    # because consecutive frames caught it almost identically.
    alpr.setdefault("duplicate_resize_width", 300)
    alpr.setdefault("duplicate_resize_height", 100)
    alpr.setdefault("duplicate_diff_threshold", 5)
    # Crop is upscaled to this height (proportionally) and padded by this
    # many pixels on each side before OCR -- PaddleOCR reads small/tight
    # crops less reliably than a properly-sized, bordered one.
    alpr.setdefault("ocr_prep_target_height", 220)
    alpr.setdefault("ocr_prep_padding", 24)
    # Where PaddleOCR looks for (and, if missing, downloads) its
    # detection/recognition/angle-classifier model files. Relative paths
    # are resolved against the config.json directory (see "_config_dir"
    # above), so by default everything -- config.json, model_path (YOLO),
    # and these -- lives in one folder instead of PaddleOCR silently
    # caching to the user's home directory (~/.paddleocr), which is what
    # it does if these aren't set. The cls (angle classifier) model is
    # downloaded unconditionally by PaddleOCR's constructor even though
    # use_angle_cls=False means it's never actually used for inference --
    # without cls_model_dir set explicitly it still goes to ~/.paddleocr,
    # so it's included here too for full offline/self-contained portability.
    alpr.setdefault("paddleocr_det_model_dir", "paddleocr_models/det")
    alpr.setdefault("paddleocr_rec_model_dir", "paddleocr_models/rec")
    alpr.setdefault("paddleocr_cls_model_dir", "paddleocr_models/cls")
    # Publish a result to MQTT even when no valid plate was found
    # (status NO_VALID_PLATE) -- a downstream consumer gets an update on
    # every trigger regardless of read success, not just successful
    # reads. The audit folder (result.json/event.json) is always written
    # either way; this only controls the MQTT side. Camera/system error
    # statuses (CAMERA_UNREACHABLE, CAMERA_CONFIG_ERROR, FRAME_SIZE_ERROR,
    # ERROR) are unaffected and always publish regardless of this
    # setting. Set to false to go back to only publishing actual reads.
    alpr.setdefault("publish_no_valid_plate", True)
    # Reject a detected box if its vertical center sits above this
    # fraction of the ROI's height (0.45 = top 45%) -- meant to filter
    # out false positives from cab signage/mounting structure above the
    # real plate. Tune per camera angle: a mounting position where the
    # plate legitimately sits high in the ROI needs this lowered (or set
    # to 0 to disable the check entirely) -- otherwise correctly detected
    # plates get silently discarded as "upper_half" rejections.
    alpr.setdefault("upper_half_fraction", 0.45)
    # Expand each detected box by this percent of its own width/height
    # (each side) before cropping for OCR, clamped to the ROI's bounds.
    # A YOLO box that's a little too narrow/short clips characters off
    # the edge of the plate -- e.g. two adjacent, overlapping detections
    # each only capturing half a plate ("HR47E" / "E4812" for what's
    # actually "HR47E4812"). Padding gives OCR a small margin so a
    # slightly-undersized box still captures the whole plate. Set to 0
    # to disable.
    alpr.setdefault("plate_crop_padding_pct", 15)
    # Expected sensor resolution (e.g. a 5MP camera -> 2592x1944). Leave
    # both null to disable the check. Used to catch a snapshot endpoint
    # unexpectedly serving a lower-res image than the camera's real sensor.
    alpr.setdefault("expected_frame_width", None)
    alpr.setdefault("expected_frame_height", None)
    alpr.setdefault("frame_size_tolerance_pct", 10)

    # Every camera is polled via its HTTP snapshot endpoint (e.g. a Bosch
    # dome's /snap.jpg) using one common username/password for all
    # cameras -- there is no per-camera override, and no RTSP/Genetec path
    # (removed entirely in favor of this).
    snapshot = cfg.setdefault("snapshot", {})
    snapshot.setdefault("url_template", "http://{ip}:{port}/snap.jpg")
    snapshot.setdefault("port", 80)
    snapshot.setdefault("username", None)
    snapshot.setdefault("password", None)
    snapshot.setdefault("connect_timeout_ms", 3000)
    snapshot.setdefault("read_timeout_ms", 3000)
    # Optional pacing between fetches within one collection window; 0 =
    # fetch as fast as the HTTP round trip allows.
    snapshot.setdefault("poll_interval_ms", 0)

    # Continuous, trigger-independent bay-activity monitor (bay_monitor.py)
    # -- a separate concern from the enter/leave ALPR pipeline above, off
    # by default. Round-robins every enabled camera looking for presence
    # (reusing the ALPR YOLO model as a cheap signal, no filtering), then
    # "zooms in" on a hit: every classify_interval_sec it sends a frame to
    # a local Ollama-hosted vision model and publishes the reply
    # (one of status_values) to mqtt.bay_status_topic_prefix + "/<bay>".
    # After empty_debounce_count consecutive "empty" reads it reverts to
    # baseline scanning. ollama_model has no working default -- it must
    # match a tag you've actually pulled locally (`ollama pull <tag>`),
    # so this fails fast at startup rather than silently misclassifying
    # everything as null if left unset.
    bay_monitor = cfg.setdefault("bay_monitor", {})
    bay_monitor.setdefault("enabled", False)
    bay_monitor.setdefault("baseline_scan_interval_ms", 2000)
    bay_monitor.setdefault("classify_interval_sec", 60)
    bay_monitor.setdefault("empty_debounce_count", 3)
    bay_monitor.setdefault("ollama_host", "http://localhost:11434")
    bay_monitor.setdefault("ollama_model", None)
    bay_monitor.setdefault("ollama_timeout_sec", 30)
    # Reasoning models (e.g. qwen3) spend several extra seconds emitting
    # a hidden "thinking" trace before the actual answer unless told not
    # to -- confirmed on this deployment to cut one classify call from
    # ~31.7s to ~6.4s. Harmless to send to non-reasoning models, which
    # simply ignore a field they don't understand.
    bay_monitor.setdefault("ollama_think", False)
    bay_monitor.setdefault("status_values",
                            ["empty", "occupied", "unloading", "loading", "idle"])
    # Which entry of status_values means "nothing is there". Drives both
    # this module's revert-to-baseline debounce and bay_state.py's
    # occupied set (everything that ISN'T this) -- neither hardcodes the
    # literal "empty" any more, so a custom status_values vocabulary
    # still works end to end.
    bay_monitor.setdefault("empty_status", "empty")
    # Which pixels the vision model is asked to judge:
    #   "full_frame" (default) -- the whole bay: cargo, forklifts, doors,
    #       which is what the activity vocabulary is really about. But a
    #       truck in an ADJACENT bay that's also in shot can drive this
    #       bay's status (and, with bay_state_engine on, open a session on
    #       the wrong bay).
    #   "roi" -- restrict it to the same cameras.json roi that presence
    #       detection uses. Removes that cross-talk, at the cost of a much
    #       narrower view (the roi is framed for plates, not for watching
    #       cargo movement). Use this if bays overlap in frame.
    bay_monitor.setdefault("classify_region", "full_frame")
    # Optional few-shot exemplars sent alongside every classification
    # call, e.g. [{"path": "reference_images/empty.jpg", "label": "empty"},
    # {"path": "reference_images/door_open.jpg", "label": "occupied (bay
    # door open)"}]. Paths are resolved relative to config.json's
    # directory, same as model_path. Loaded once at startup, not re-read
    # per call. Helps most on ambiguous real-world cases a text-only
    # prompt undersells -- e.g. an open bay/cargo door that can otherwise
    # get misread as empty or confuse the model.
    bay_monitor.setdefault("reference_images", [])
    # Overwrites audit/<bay>/latest_frame.jpg on every successful
    # snapshot fetch -- a live "what does this camera see right now"
    # view for remote troubleshooting (e.g. confirming camera framing/
    # focus/ROI without SSH/RDP access to actually look at the feed),
    # without the volume of diagnostics_mode='troubleshooting''s
    # per-event dumps. One file per bay, always the latest, never
    # accumulates.
    bay_monitor.setdefault("save_latest_frame", True)
    # Longest side (pixels) an image is downscaled to before being JPEG
    # encoded and sent to the vision model, and the JPEG quality used.
    # A raw 5MP snapshot is ~6x the pixels of a 1024-wide view, and a
    # vision model's prefill cost scales with pixel count -- on a CPU-only
    # box that difference routinely blows past ollama_timeout_sec, and a
    # timeout discards the fetch, the encode and the partial inference
    # entirely, leaving the bay with no status at all. Reference exemplars
    # are downscaled the same way, once at startup. Set to 0 to disable
    # downscaling and send frames at full resolution.
    bay_monitor.setdefault("classify_max_dimension", 1024)
    bay_monitor.setdefault("classify_jpeg_quality", 80)
    # The baseline presence check: a small thumbnail of the ROI is
    # compared against the one from the last time this bay was looked at
    # while empty -- a resize+diff (~100us), nothing else. Deliberately
    # NOT plate detection: a bay is "occupied" or not regardless of
    # whether a plate happens to be visible in frame (a reversed-in
    # trailer's plate usually faces away from the dock camera entirely),
    # so this has to be a general "did the scene change" test, not
    # "did the ALPR model find a plate-shaped box". A change big enough
    # to matter -- a truck arriving -- moves far more pixels than
    # presence_diff_threshold tolerates, so this cannot miss an arrival.
    # Set presence_diff_enabled to false to report presence
    # unconditionally on every round (bypasses the check entirely --
    # useful for tuning classify_interval_sec/the prompt without waiting
    # for an actual scene change).
    bay_monitor.setdefault("presence_diff_enabled", True)
    bay_monitor.setdefault("presence_diff_resize_width", 160)
    bay_monitor.setdefault("presence_diff_resize_height", 120)
    bay_monitor.setdefault("presence_diff_threshold", 3.0)
    # Same diff mechanism as presence_diff_*, but for a DIFFERENT
    # question: while a bay is already zoomed in, has the scene changed
    # enough since the last classify to be worth reclassifying right now
    # (e.g. idle -> unloading), rather than waiting out
    # classify_interval_sec. Runs alongside that timer, not instead of
    # it -- a subtle change too small to trip this still gets caught on
    # the next scheduled tick. Deliberately a separate threshold from
    # presence_diff_threshold: "is anything here at all" and "did an
    # established truck's activity change" don't need the same
    # sensitivity. Set classify_diff_enabled to false to go back to
    # pure timer-based reclassification only.
    bay_monitor.setdefault("classify_diff_enabled", True)
    bay_monitor.setdefault("classify_diff_threshold", 3.0)
    bay_monitor.setdefault("classification_prompt", (
        "You are monitoring a truck loading dock bay through a fixed "
        "security camera. Classify the current activity into EXACTLY ONE "
        "of these words: empty, occupied, unloading, loading, idle. "
        "'empty' = no truck present. 'occupied' = a truck is present but "
        "no cargo movement is visible. 'unloading' = cargo is visibly "
        "being removed from the truck. 'loading' = cargo is visibly being "
        "loaded onto the truck. 'idle' = a truck is present, not actively "
        "loading or unloading (e.g. waiting, doors closed, driver break). "
        "Respond in EXACTLY this two-line format, nothing else:\n"
        "STATUS: <the single classification word>\n"
        "COMMENT: <briefly explain WHY you chose that status -- the "
        "specific visual evidence you reasoned from (e.g. cargo doors "
        "open/closed, boxes or pallets visible on the ground or being "
        "carried, a forklift or workers present, the truck bed's "
        "visible contents), not just a restatement of the status word>"
    ))
    # On-demand snapshot server (snapshot_webhook.py) -- HTTP GET
    # http://<host>:<port>/snapshot/<bay> returns bay_monitor's most
    # recently saved frame (audit/<bay>/latest_frame.jpg, requires
    # save_latest_frame -- default on) plus this bay's last-known
    # occupancy_status/activity, as JSON with a base64 image. Pull, not
    # push: nothing is sent anywhere until requested. Off by default;
    # requires bay_monitor.enabled.
    bay_monitor.setdefault("snapshot_webhook_enabled", False)
    bay_monitor.setdefault("snapshot_webhook_host", "0.0.0.0")
    bay_monitor.setdefault("snapshot_webhook_port", 8081)
    bay_monitor.setdefault("snapshot_webhook_threads", 4)
    # How the served frame is resized before base64-encoding: longest
    # side capped at max_dimension, aspect ratio preserved (so a 4:3
    # camera yields 640x480, a 16:9 one 640x360). Full camera resolution
    # would make a multi-megabyte JSON body -- base64 alone inflates by
    # ~33% -- out of what is normally wanted as a thumbnail. 0 serves the
    # frame at its original size. Deliberately separate from
    # classify_max_dimension/classify_jpeg_quality (what the VISION MODEL
    # is shown): retuning what a dashboard pulls shouldn't silently
    # change what gets classified, or vice versa.
    bay_monitor.setdefault("snapshot_webhook_max_dimension", 640)
    bay_monitor.setdefault("snapshot_webhook_jpeg_quality", 80)

    # Which backend answers "what is happening at this bay".
    #   "yolo"    a purpose-trained truck/door detection model
    #             (truck_model_path). ~50ms per frame, deterministic, and
    #             it reports door state, which the VLM never did reliably.
    #             Worth understanding why the speed matters beyond the
    #             obvious: bay_monitor scans every bay from ONE thread,
    #             so a multi-second VLM call doesn't just delay that bay,
    #             it stalls the round-robin for every bay queued behind
    #             it. It also removes the failure mode where a timing-out
    #             VLM leaves a bay with no status at all indefinitely.
    #   "ollama"  the original vision-model prompt (classification_prompt
    #             + status_values + reference_images).
    # The VLM stays reachable either way for on-demand questions (the
    # webhook's POST /bay/<bay>/ask) -- routine status is a closed-
    # vocabulary question a detector answers better, while open questions
    # ("loading or unloading?") are what a VLM is actually good at.
    bay_monitor.setdefault("classifier", "ollama")
    # The trained truck/door weights, resolved relative to config.json
    # (like model_path and the PaddleOCR dirs). Only read when
    # classifier == "yolo"; startup fails loudly if it's missing then,
    # rather than silently reporting every bay as empty forever.
    bay_monitor.setdefault("truck_model_path", "Truck_model.pt")
    bay_monitor.setdefault("truck_conf_threshold", 0.4)
    # Trucks are large objects filling much of the frame, so detection
    # doesn't need the 640 a plate would -- 416 is 2-3x cheaper for the
    # same answer. Raise it if small/distant trucks are being missed.
    bay_monitor.setdefault("truck_imgsz", 416)
    # Detected and reported (plate_visible), never used to decide
    # presence: a bay is occupied whether or not a plate happens to be
    # readable from the dock camera's angle, which is exactly why
    # bay_monitor stopped using the ALPR plate model for presence.
    bay_monitor.setdefault("truck_plate_class", "Number_Plate")
    # Overrides/extends truck_detector.DEFAULT_CLASS_MAP, for a model
    # retrained with different class names:
    #   {"<class name>": {"status": "<one of status_values>",
    #                     "door_state": "open"|"closed"}}
    # Merged over the built-in map rather than replacing it, so a partial
    # override doesn't silently drop the classes it didn't mention.
    bay_monitor.setdefault("truck_class_map", {})
    # Consecutive agreeing frames before a door-state CHANGE is believed,
    # same reasoning as empty_debounce_count: a half-raised shutter or
    # someone walking through the doorway shouldn't flip the reported
    # state. Only affects the state the webhook reports -- the
    # door-open-on-arrival alert deliberately uses the raw reading from
    # the arrival frame itself, since that's a claim about one moment.
    bay_monitor.setdefault("door_state_debounce_count", 2)
    # Whether the truck-model backend also base64-encodes each classified
    # frame for downstream payloads. Pure overhead unless something
    # actually ships the image, so it follows the one topic that does.
    # (The Ollama backend always has an encoded frame regardless -- it
    # had to build one to make the call at all.)
    bay_monitor.setdefault("attach_frame_to_status",
                           mqtt["publish_bay_notification"])
    # Per-bay state engine (bay_state.py) -- fuses bay_monitor's
    # continuous status stream with ALPR plate reads into one session per
    # bay, and becomes the authority for enter/leave direction and
    # timing instead of the MQTT/HTTP trigger source: an empty->occupied
    # transition enqueues an ALPR read (retried across the whole stay,
    # not just once), and occupied->empty publishes "leave" using
    # whatever plate got confirmed at any point during the visit. Off by
    # default; depends entirely on bay_monitor also being enabled --
    # fails fast at startup otherwise (see check_bay_state_config() in
    # service.py). No config of its own beyond "enabled": its departure
    # timing directly follows bay_monitor.empty_debounce_count rather
    # than a separate setting, to guarantee the two can never fall out
    # of sync (see bay_state.py's BayStateEngine.on_status for why a
    # second, independently-configured debounce would be unsafe here).
    bay_state_engine = cfg.setdefault("bay_state_engine", {})
    bay_state_engine.setdefault("enabled", False)
    # Overwrites audit/bay_state.csv with a one-row-per-bay snapshot of
    # current session state (bay_status, session_open, direction, plate,
    # confidence, read_attempts, last_updated) on every classification
    # and every plate confirmation -- a file anyone can just open
    # (Excel, Notepad) to see current state at a glance, no MQTT client
    # or subscription required.
    bay_state_engine.setdefault("save_state_csv", True)

    # Worker pool sizing -- num_workers is roughly "max concurrent
    # dockings this instance can process at once" (each Worker owns its
    # own loaded YOLO+PaddleOCR models, ~2GB RAM apiece); queue_max caps
    # how many pending jobs can sit in JobBus before try_enqueue starts
    # dropping new ones rather than growing unbounded.
    service_cfg = cfg.setdefault("service", {})
    service_cfg.setdefault("num_workers", 3)
    service_cfg.setdefault("queue_max", 50)

    log_cfg = cfg.setdefault("logging", {})
    log_cfg.setdefault("level", "INFO")  # DEBUG / INFO / WARNING / ERROR
    log_cfg.setdefault("console", True)
    log_cfg.setdefault("file", None)     # e.g. "logs/alpr_service.log"
    log_cfg.setdefault("max_bytes", 10 * 1024 * 1024)
    log_cfg.setdefault("backup_count", 5)

    return cfg
