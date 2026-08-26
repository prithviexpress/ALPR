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
#   16. Two remote-troubleshooting aids, both on by default:
#       (a) "bay_monitor.save_latest_frame" overwrites
#           audit/<bay>/latest_frame.jpg on every successful snapshot
#           fetch -- a live "what does this camera see right now" view
#           for confirming framing/focus/ROI without direct access to the
#           feed, one file per bay, always the most recent, never
#           accumulating.
#       (b) "alpr.save_detected_plate_frames" copies the crop that
#           actually produced a SUCCESS result's winning plate into a
#           separate, flat audit/detected_plates/ folder (named
#           "<timestamp>_<bay>_<direction>_<plate>.jpg") -- so confirmed
#           reads can be browsed chronologically across every bay at a
#           glance, without hunting through each job's own per-event
#           audit subfolder (which still gets the full detail either way).
#   17. Fixed a real field failure where bay_monitor's Ollama calls
#       ("...actively refused it" / "connect timeout=30") failed while a
#       plain `curl` to the exact same host:port succeeded from the same
#       machine -- the signature of a corporate proxy that Python's
#       `requests` honors by default (via HTTP_PROXY/HTTPS_PROXY env vars
#       or Windows system proxy settings, trust_env=True) intercepting or
#       blocking traffic to an internal-only address that curl's shell
#       session was not routing through it. Cameras and the local Ollama
#       host are always on the internal network and should never go
#       through an outbound proxy, so both `BayMonitor.session` and
#       `Worker.session` now set `trust_env = False` right after
#       creation, and classify_frame() takes the caller's session instead
#       of using the bare `requests.post` module function (which had no
#       session to disable proxying on). No new config -- this is an
#       unconditional code-level fix.
#   18. "bay_monitor.ollama_think" (default false) is now sent as the
#       /api/generate "think" field. Confirmed on a real deployment
#       running a Qwen3 reasoning model that this alone cut one classify
#       call from ~31.7s to ~6.4s -- reasoning models emit a hidden
#       "thinking" trace before their actual answer unless told not to.
#       Harmless to leave on for non-reasoning models, which simply
#       ignore a field they don't recognize.
#   19. Supersedes the presence-check design from items #11 and #14:
#       bay_monitor no longer loads or calls the ALPR YOLO plate model at
#       all. That reuse was wrong for what presence detection needs to
#       answer -- a bay is occupied or not regardless of whether a plate
#       happens to be visible in frame, and a reversed-in trailer's plate
#       usually faces away from the dock camera entirely, so gating
#       "zoomed in" on a detected plate box left the monitor blind to
#       exactly the trucks it exists to track (it would never zoom in,
#       never classify, and never report anything for that bay's whole
#       visit). "bay_monitor._detect_presence" is now a pure frame-diff
#       against the ROI's last-seen-empty thumbnail -- any meaningful
#       visual change (truck, forklift, person, open door) counts,
#       independent of plate visibility or the ALPR pipeline entirely.
#       Since there's no expensive model pass left to gate, the old
#       skip-N-rounds/periodic-recheck machinery ("presence_diff_max_
#       skip", "presence_conf_threshold", "presence_imgsz") is gone too
#       -- every baseline round now runs the (~100us) diff directly, with
#       nothing to skip.
#   20. Two deliberate redesigns of how a plate gets read and how bay
#       activity gets reclassified, replacing collect-a-batch-then-vote
#       designs with immediate, event-driven ones:
#       (a) ALPR (worker.py): OCR is no longer deferred until the whole
#           collection window ends. The first detected plate box in ANY
#           frame is now the trigger to start OCRing immediately --
#           best-scoring candidate first within a frame, continuing
#           across subsequent frames -- and the ENTIRE collection stops
#           the instant one produces a plate that passes
#           plate_text.is_valid(), rather than waiting out
#           collection_timeout and voting across whatever showed up.
#           "alpr.max_ocr_attempts" caps total OCR calls per collection
#           window (replaces "max_raw_samples"/"best_samples", which no
#           longer mean anything now that collection and OCR are one
#           interleaved loop). There is no fallback vote if the budget
#           runs out with nothing valid -- plate_text.weighted_vote() is
#           removed entirely; the result is plain NO_VALID_PLATE, and
#           every individual attempt (valid or not) is still in the
#           audit result.json's "reads" array for forensics. collect()
#           is renamed collect_and_read() to match what it now does; the
#           audit field "raw_vote" is renamed "final_plate" (same value
#           as truck_number on SUCCESS, null otherwise -- there's no
#           more "best guess among garbage" left to preserve there).
#       (b) bay_monitor: while a bay is already zoomed in, reclassifying
#           no longer waits purely on the classify_interval_sec timer.
#           The same thumbnail-diff mechanism as the baseline presence
#           check now also runs on every round while occupied
#           ("bay_monitor.classify_diff_enabled"/"classify_diff_
#           threshold", separate threshold from presence_diff_*), and a
#           meaningful scene change (e.g. idle -> unloading) triggers an
#           immediate reclassify rather than waiting out the rest of the
#           interval. The timer still fires as a backstop for changes
#           too subtle to trip the diff -- this runs ALONGSIDE it, not
#           instead of it. A zoomed-in bay is now fetched every
#           baseline_scan_interval_ms round (not just when the timer is
#           due) to make this possible.
#   21. Fixed a real field failure: a bay already occupied by a truck
#       when the service starts was never classified at all, and stayed
#       that way indefinitely -- the presence-diff check (item #19) has
#       nothing to compare the very first frame against, so it silently
#       establishes THAT frame (truck already in it) as the "empty"
#       baseline and reports no presence, forever, until some LATER
#       visual change happens to trip the diff. No Ollama call, no
#       bay_state_engine session, no ALPR read, nothing -- while the
#       process looked completely healthy. bay_monitor now tracks
#       BayState.classified_once and forces exactly one real classify
#       per bay regardless of the diff if it's never been classified
#       before, to establish actual ground truth on startup (a failed
#       first attempt, e.g. Ollama unreachable, does NOT set the flag,
#       so it keeps retrying rather than giving up permanently). Also
#       added "mqtt.bay_state_topic_prefix" (default site/alpr/bay_state):
#       bay_state_engine now publishes its live per-bay session -- open,
#       direction, plate (if confirmed), read_attempts -- on every state
#       change (arrival, plate confirmed, departure), not just the final
#       enter/leave result on departure. Unlike that result, this topic
#       is NOT suppressed by alpr.publish_no_valid_plate=false, since its
#       whole purpose is showing the engine reacting at all -- exactly
#       what was invisible during this failure.
#   22. Fixed a real field complaint: too many "enter" MQTT publishes
#       with an invalid (NO_VALID_PLATE) result for what was really one
#       physical truck. Root cause: bay_state_engine retries a visit's
#       plate read every classification until it succeeds or
#       alpr.max_read_attempts is hit, and Worker.handle() was
#       publishing EVERY one of those attempts (default
#       alpr.publish_no_valid_plate=true), not just the eventual good
#       one -- confirmed these never continue PAST a successful read
#       (bay_state_engine stops enqueuing once session.plate is
#       confirmed), so the flood was always attempts BEFORE (or in place
#       of, if a visit never succeeds) a valid detection, never after.
#       mqtt_bus.make_job() now takes a "source" ("bay_state" for
#       bay_state_engine's own retries, unset for one-shot MQTT/HTTP
#       triggers); Worker.handle() publishes a bay_state-sourced
#       NO_VALID_PLATE result to disk and to audit/all_attempts/ (see
#       below) same as always, but no longer to MQTT -- SUCCESS still
#       publishes immediately as before, and a one-shot trigger's
#       NO_VALID_PLATE is completely unaffected. bay_state_engine's own
#       on_alpr_result hook still sees every attempt regardless, so
#       nothing about the retry/confirmation logic itself changed.
#
#       Also added "alpr.save_all_attempt_frames" (default true): every
#       completed job's best-scoring attempted crop -- SUCCESS or not --
#       now also lands in one flat audit/all_attempts/ folder (named
#       "<timestamp>_<bay>_<direction>_<status>_<plate-or-UNKNOWN>.jpg"),
#       separate from save_detected_plate_frames (SUCCESS-only). Directly
#       answers "what is the camera actually seeing, why is this bay
#       coming back NO_VALID_PLATE" by browsing one folder chronologically
#       instead of opening each job's own nested per-event audit
#       subfolder one at a time.
#   23. Added "bay_state_engine.save_state_csv" (default true):
#       overwrites audit/bay_state.csv with a one-row-per-bay snapshot
#       (bay_status, session_open, direction, plate, confidence,
#       read_attempts, last_updated) on every classification and every
#       plate confirmation. Requested directly: MQTT's bay_state topic
#       (item #20) needs a client and a subscription just to see "what's
#       happening right now" -- this is a plain file anyone can just
#       open (Excel, Notepad) and have it always show current state, no
#       tooling required. Always the full current table, not an
#       append-only log, written via temp-file-then-rename so a reader
#       never sees a half-written row.
#   24. Two more MQTT notification types, both requested directly:
#       (a) bay_monitor.classification_prompt's default now asks the
#           vision model for a two-line "STATUS: <word>\nCOMMENT:
#           <sentence>" reply instead of just the bare word.
#           bay_monitor.split_status_comment() isolates the status
#           before it's resolved, so a comment mentioning a DIFFERENT
#           status word (e.g. "STATUS: idle / COMMENT: looks mostly
#           empty of cargo") can't make the whole reply look ambiguous
#           the way it would if status and comment were resolved from
#           the same blob of text. classify_frame() now returns
#           (status, comment, image_b64) instead of a bare status, and
#           hands the comment and the exact base64 frame it sent to
#           Ollama through to bay_monitor's on_status hook.
#       (b) bay_state_engine now publishes a RICH notification to
#           "mqtt.bay_notification_topic_prefix" (default site/alpr/
#           bay_notification) whenever bay_monitor's classification
#           actually CHANGES for a bay (idle->loading, empty->occupied,
#           etc -- including the bay's very first-ever classification,
#           a "change" from unknown) -- {bay, occupancy_status, activity,
#           truck_number, comment, image_base64, timestamp}.
#           truck_number is whatever's confirmed so far (or
#           unknown_plate_value), captured before any session reset so
#           a departure notification still names the truck that just
#           left. Deliberately gated on an actual CHANGE, not fired on
#           every classification the way bay_status_topic_prefix is --
#           a truck sitting "idle" for an hour of periodic
#           classify_interval_sec rechecks shouldn't re-notify every
#           time, only when something's actually different.
#       (c) Separately, bay_monitor now also publishes a periodic,
#           classification-independent image-only heartbeat --
#           "bay_monitor.snapshot_publish_interval_sec" (default 300s),
#           to "mqtt.bay_snapshot_topic_prefix" (default site/alpr/
#           bay_snapshot) -- {bay, image_base64, timestamp}. Fires on
#           its own timer per bay regardless of activity, so a
#           downstream consumer always has a recent frame without
#           waiting for (or triggering) a status change. 0 disables it.
#   25. mqtt.bay_state_topic_prefix's payload gained "occupancy_status"
#       ("occupied"/"empty") and "activity" (bay_monitor's raw
#       classification word) fields -- a real user watching this topic
#       only saw "open": true/false and asked "what is this open, I
#       wanted occupancy". "open" is left in place (it's the session-
#       lifecycle boolean bay_state_engine itself reasons about, and is
#       functionally the same signal), but occupancy_status/activity are
#       now right there in the SAME message, not only on the separate
#       mqtt.bay_notification_topic_prefix (item #24) someone would have
#       had to discover and subscribe to separately.
#   26. Two more direct requests:
#       (a) bay_monitor.classification_prompt's COMMENT instruction now
#           explicitly asks for the vision model's REASONING -- the
#           specific visual evidence it based the status on (cargo
#           doors open/closed, boxes/pallets visible, a forklift or
#           workers present) -- not just a restatement of the status
#           word ("the bay is loading"). Makes mqtt.bay_notification_
#           topic_prefix's comment field actually useful for spot-
#           checking a classification without opening the frame.
#       (b) mqtt.bay_snapshot_topic_prefix's periodic heartbeat payload
#           gained "occupancy_status"/"activity", carrying this bay's
#           last-known classification (BayState.last_status) alongside
#           the image -- set whenever a classify actually runs, not
#           freshly computed by the heartbeat itself (which stays a
#           cheap JPEG-encode-only operation, no Ollama round trip). The
#           heartbeat still fires unconditionally on its own timer --
#           mqtt.bay_notification_topic_prefix (item #24) remains the
#           only channel gated on an actual status CHANGE, which is the
#           one meant to be treated as a "notification" to react to.
#   27. Reversed, on direct request ("don't publish snapshot
#       continuously, instead publish webhook, I will request when I
#       need"): the periodic MQTT image heartbeat introduced in item
#       #24(c) and enriched in #26(b) -- bay_monitor.
#       snapshot_publish_interval_sec / mqtt.bay_snapshot_topic_prefix
#       -- is REMOVED. Nothing about a bay's image is pushed to MQTT on
#       a timer anymore.
#       In its place: a small on-demand HTTP server, snapshot_webhook.py,
#       enabled via bay_monitor.snapshot_webhook_enabled (default
#       false, requires bay_monitor.enabled) and served by waitress on
#       its own configurable host/port (bay_monitor.
#       snapshot_webhook_host/_port/_threads, default 0.0.0.0:8081) --
#       the same pattern as http_trigger.py's server, but deliberately a
#       separate one: receiving camera-alert triggers and serving
#       snapshots on request are unrelated concerns, and a deployment
#       may want either without the other. GET /snapshot/<bay> returns
#       {bay, occupancy_status, activity, image_base64, timestamp}, 404
#       if the bay is unrecognized or nothing's been captured for it
#       yet. GET /healthz for a liveness check.
#       BayMonitor.get_snapshot(bay) serves audit/<bay>/latest_frame.jpg
#       (already written every scan round by save_latest_frame, default
#       on -- never more than one baseline_scan_interval_ms round
#       stale, 2s by default) plus the bay's last-known classification
#       (BayState.last_status), rather than fetching a brand-new frame
#       per request: that would mean sharing bay_monitor's own
#       HTTPDigestAuth session (stateful, not meant for concurrent
#       cross-thread reuse) with whatever waitress worker thread handles
#       the HTTP request. start_bay_monitor() now returns (monitor,
#       thread, stop_event) instead of (thread, stop_event) so
#       service.py can wire the webhook's route table up against the
#       running monitor's own get_snapshot method.
#       mqtt.bay_notification_topic_prefix (item #24(b)) is unaffected
#       -- it's the one channel still allowed to push anything, and only
#       because it's gated on an actual status CHANGE rather than a
#       timer.
#   28. A purpose-trained truck/door YOLO model becomes an alternative
#       classification backend, the MQTT topics collapse to one lean
#       event stream, and the snapshot webhook grows into a bay query
#       API. All four parts were requested together.
#       (a) bay_monitor.classifier picks the backend: "yolo" runs
#           truck_detector.py against bay_monitor.truck_model_path
#           (~50ms, deterministic), "ollama" keeps the original vision
#           prompt. The speed is not the only point: bay_monitor scans
#           every bay from ONE thread, so a multi-second VLM call
#           stalled the round-robin for every bay behind it, and a
#           timing-out call left a bay reporting no status at all
#           (activity: null) indefinitely. Class mapping is
#           Truck_Enter_Closed->arriving/closed, Truck_Enter_Open->
#           arriving/open, Truck_Docked_Closed->docked/closed,
#           Truck_Docked_Open->loading/open, overridable via
#           truck_class_map for a model retrained with other names. The
#           model's Number_Plate class is reported (plate_visible) but
#           NEVER counts as truck presence -- a bay is occupied whether
#           or not a plate is readable from the dock camera's angle,
#           which is exactly why bay_monitor stopped using the ALPR
#           plate model for presence back in item #19. The single
#           highest-confidence truck box decides the status, so two
#           overlapping detections of one truck as different classes
#           can't have box ORDER decide the answer.
#       (b) Door state, which the VLM never reported reliably. The
#           value bay_monitor reports (and GET /bay/<bay> serves) is
#           debounced over bay_monitor.door_state_debounce_count
#           agreeing frames, same reasoning as empty_debounce_count.
#           The requested "truck arrived with its doors already open"
#           alert deliberately uses the RAW reading from the arrival
#           frame instead, since that's a claim about one moment and
#           debouncing it would answer a different question. It rides
#           on the arrival event as alert="door_open_on_arrival"
#           rather than a separate topic -- one subscription was the
#           point.
#       (c) mqtt.bay_event_topic_prefix, the lean consumer-facing topic
#           and now the ONLY one published by default: 'arrived' /
#           'identified' / 'departed', roughly three messages per visit
#           against the dozens the older topics emitted between them.
#           arrived fires immediately with the UNKNOWN placeholder
#           rather than waiting on the ALPR read (a dock system needs
#           to know a truck is there NOW); identified follows when the
#           plate resolves, and is suppressed when the plate was
#           already known at arrival so there's never a redundant pair,
#           or when the truck has already departed. departed carries
#           the truck number and duration_sec. bay_status, bay_state
#           and bay_notification are each still available behind
#           mqtt.publish_bay_* but default to false now. The enter/
#           leave ALPR result topics are untouched -- different shape,
#           different consumer, its own publish_no_valid_plate gate.
#       (d) snapshot_webhook.py becomes a query API: GET /bays (every
#           bay in one call), GET /bay/<bay> (occupancy and activity
#           from bay_monitor merged with truck identity from
#           bay_state_engine -- neither can answer it alone, so
#           service.build_bay_state_query composes them rather than
#           making the two modules reference each other), the existing
#           GET /snapshot/<bay>, and POST /bay/<bay>/ask, which puts an
#           arbitrary question to the vision model about a bay's
#           current frame. That last route is what the VLM is FOR once
#           a detector handles routine status: "loading or unloading?"
#           is exactly what a fixed detector vocabulary cannot answer
#           (the truck model sees doors open at a dock, not which way
#           the cargo is moving). It passes session=None so the call
#           opens its own connection rather than borrowing the scan
#           loop's requests.Session from another thread. A route whose
#           handler isn't wired up answers 501, not 404 -- "this
#           service doesn't offer that" must not look like "no such
#           bay".
#       (e) The truck model's Number_Plate detection now GATES retry
#           plate reads (alpr.retry_only_when_plate_visible, default
#           true). Item (a) left plate_visible computed and then thrown
#           away, so retries stayed blind: bay_state_engine re-ran a
#           full collection on every classification while a bay was
#           occupied without a confirmed plate, whether or not a plate
#           was anywhere in view. Since max_read_attempts is a per-VISIT
#           budget, a retry fired while the plate is out of view isn't
#           merely wasted work on one of only service.num_workers
#           threads -- it makes it likelier the budget is exhausted
#           before the plate ever comes INTO view, which is precisely
#           how a truck ends up departing unidentified.
#           Gates RETRIES only: the arrival read always fires, since
#           entry is when a truck is most likely facing the camera and
#           the worker polls its own frames across a whole collection
#           window afterwards -- "no plate in this one classified
#           frame" doesn't mean none will appear during that window.
#           plate_visible=None (what the Ollama backend reports) means
#           NO INFORMATION, not "no plate", and never suppresses a
#           retry -- otherwise turning this on would silently stop all
#           retries for a deployment not running the truck model.
#           GET /bay/<bay> reports plate_visible too, so "no read yet
#           because no plate is visible" can be told apart from "a
#           plate is right there and OCR keeps failing" -- two very
#           different problems that present with the same symptom.
#           Door state deliberately gates nothing: a plate is readable
#           regardless of whether the CARGO doors are open.
#   29. Both models vote on truck entry, and an open door now ends the
#       plate read instead of prolonging it. Two requests.
#       (a) bay_monitor.plate_assist_enabled runs the dedicated
#           plate-only model (by default the same top-level model_path
#           weights the ALPR workers use) ALONGSIDE the truck model, as
#           a second opinion on presence. The two fail in different
#           places -- the truck model can miss a truck at an awkward
#           angle or in poor light, while the plate-only model is
#           trained on one thing and often still finds the plate in
#           exactly those frames -- so their UNION (a truck OR a plate
#           counts as presence) catches entries either alone would
#           miss. That union is the entire reason to pay for a second
#           inference per classification. It also feeds a better
#           plate_visible signal into item #28(e)'s retry gate, since
#           the dedicated model is the better judge of that.
#           When ONLY the plate model sees something, the bay reads as
#           plate_assist_only_status ("occupied") with door_state left
#           UNKNOWN rather than guessed: a plate box says nothing about
#           a cargo door, and inventing "closed" would feed a fabricated
#           reading straight into the door-open-on-arrival alert. When
#           the truck model DOES see a truck it keeps deciding status
#           and door state; the plate model only ever contributes plate
#           visibility, never overrides.
#       (b) alpr.abandon_read_when_doors_open stops trying to read a
#           plate once the truck's doors are reported OPEN, publishing a
#           "plate_unreadable" event (reason="doors_open") on the
#           bay_event topic instead. The physical reason: on a
#           reversed-in trailer the rear doors swing OUT and cover the
#           number plate, so once they're open the plate is not in the
#           frame at all -- no amount of retrying, better OCR or a
#           longer collection window recovers it, and every further
#           attempt photographs a door while spending the visit's
#           max_read_attempts budget. Published once per VISIT rather
#           than per classification, and cleared at both arrival and
#           departure so the next truck starts fresh.
#           The ARRIVAL read still fires even for a truck that backs in
#           with its doors already open: it costs one attempt, it lands
#           while the truck is still moving into position, and
#           abandoning before ever looking would mean never reading a
#           plate that was in fact briefly visible.
#           door_state is None on the ollama backend, which means no
#           information and never abandons -- same rule as plate_visible
#           in item #28(e).
#
# New dependencies: `requests` (HTTP snapshot fetch + digest auth),
# `flask` and `waitress` (used if http_trigger.enabled or bay_monitor.
# snapshot_webhook_enabled is true) -- see requirements.txt for pinned
# versions. bay_monitor needs no new pip package (talks to Ollama over
# plain HTTP via `requests`), but does need a local Ollama install
# (https://ollama.com) with a vision-capable model pulled if bay_monitor.
# enabled is turned on.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from alpr_service.service import main

if __name__ == "__main__":
    main()
