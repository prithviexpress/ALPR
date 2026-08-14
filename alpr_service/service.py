"""Process entry point: wires config, logging, cameras, MQTT, and workers
together. Import and call main() from the top-level entry script."""
import signal
import threading

import paho.mqtt.client as mqtt

from .audit import prune_audit
from .cameras import CamerasError, load_cameras
from .config import BASE_DIR, ConfigError, load_config
from .logging_setup import configure_logging, get_logger
from .mqtt_bus import (JobBus, connect_with_retry, extract_event, make_job,
                        matches_class_filter)
from .worker import Worker

AUDIT_DIR = BASE_DIR / "audit"
CAMERAS_FILE = BASE_DIR / "cameras.json"


def check_camera_ip_fields(cameras: dict, log):
    """Fail fast at startup if any camera is missing 'ip' -- every camera
    is reached via its HTTP snapshot endpoint now, so this is required
    for all of them, rather than discovering it one CAMERA_CONFIG_ERROR
    job at a time."""
    missing = [bay for bay, cam in cameras.items() if not cam.get("ip")]
    if missing:
        log.error(f"snapshot capture requires 'ip' on every camera, but "
                  f"it's missing for: {missing}")
        raise SystemExit(1)


def check_snapshot_credentials(config: dict, log):
    """Fail fast at startup if snapshot.username/password aren't set --
    every camera uses this one common credential pair."""
    snap_cfg = config["snapshot"]
    if snap_cfg.get("username") is None or snap_cfg.get("password") is None:
        log.error("snapshot.username and snapshot.password must be set in "
                  "config.json (one common set of credentials for every camera)")
        raise SystemExit(1)


def check_bay_monitor_config(config: dict, log):
    """Fail fast at startup if bay_monitor is enabled but ollama_model
    isn't set -- there's no sane default (it has to match a tag actually
    pulled into the local Ollama install), and running with it unset would
    mean every single classification call silently fails instead of
    surfacing the misconfiguration once, up front."""
    bm_cfg = config["bay_monitor"]
    if bm_cfg["enabled"] and not bm_cfg.get("ollama_model"):
        log.error("bay_monitor.enabled is true but bay_monitor.ollama_model "
                  "is not set -- set it to a vision model tag you've pulled "
                  "locally via `ollama pull <tag>` (e.g. a Qwen VL tag)")
        raise SystemExit(1)


def check_bay_state_config(config: dict, log):
    """Fail fast at startup if bay_state_engine is enabled without
    bay_monitor -- the state engine has no signal of its own, it entirely
    depends on bay_monitor's status classification stream to know when a
    bay's occupancy starts/ends. Also warns (doesn't block) if the
    original MQTT/HTTP triggers are also enabled alongside it, since both
    paths would then independently decide enter/leave for the same bay --
    redundant at best, duplicate/conflicting publishes at worst."""
    if not config["bay_state_engine"]["enabled"]:
        return
    if not config["bay_monitor"]["enabled"]:
        log.error("bay_state_engine.enabled is true but bay_monitor.enabled "
                  "is false -- the state engine has no signal without it, "
                  "enable bay_monitor too")
        raise SystemExit(1)
    if config["mqtt"]["trigger_enabled"] or config["http_trigger"]["enabled"]:
        log.warning("bay_state_engine is enabled AND an MQTT/HTTP trigger "
                    "is also enabled -- both will independently decide "
                    "enter/leave for the same bays, which usually means "
                    "redundant or conflicting result publishes. Consider "
                    "disabling mqtt.trigger_enabled/http_trigger.enabled "
                    "if bay_state_engine is meant to be the sole source "
                    "of enter/leave direction.")


def build_mqtt(cameras: dict, config: dict, bus: JobBus,
               subscribe_enabled: bool = True) -> mqtt.Client:
    log = get_logger("MQTT")
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):
        client = mqtt.Client()

    enter_topic = config["mqtt"]["enter_subscribe_topic"]
    leave_topic = config["mqtt"]["leave_subscribe_topic"]
    bay_segment_index = config["mqtt"]["bay_segment_index"]
    class_types = config["event_filter"]["class_types"]
    min_likelihood = config["event_filter"]["min_likelihood"]

    def on_connect(client, userdata, flags, rc, *args):
        log.info(f"connected rc={rc}")
        if subscribe_enabled:
            log.info(f"subscribing enter='{enter_topic}' "
                      f"leave='{leave_topic}'")
            client.subscribe(enter_topic, qos=1)
            client.subscribe(leave_topic, qos=1)
        else:
            log.info("mqtt.trigger_enabled=false -- not subscribing "
                      "(http_trigger is the active trigger source); "
                      "this connection is publish-only")

    def on_disconnect(client, userdata, *args):
        log.warning("disconnected -- paho will auto-reconnect")

    def make_on_message(direction):
        def on_message(client, userdata, msg):
            event = extract_event(msg.topic, msg.payload, bay_segment_index)
            if event is None:
                return
            bay = event["bay"]
            if bay not in cameras:
                log.warning(f"({direction}) event for unknown bay '{bay}' "
                            f"(topic {msg.topic})")
                return
            if not cameras[bay].get("enabled", True):
                return

            matched, cls_text, likelihood = matches_class_filter(
                event["data"], class_types, min_likelihood)
            if not matched:
                log.debug(f"({bay}/{direction}) event discarded: no detection "
                          f"matching class in {class_types} at "
                          f"likelihood>={min_likelihood} (topic {msg.topic})")
                return
            if likelihood is None:
                log.info(f"({bay}/{direction}) event_filter disabled (empty "
                         f"class_types), accepting event unconditionally")
            else:
                log.info(f"({bay}/{direction}) event matched class='{cls_text}' "
                         f"likelihood={likelihood:.2f}")

            if bus.try_enqueue(make_job(bay, direction, event["event_time"],
                                         cls_text, likelihood)):
                log.info(f"({bay}/{direction}) event queued for processing")
        return on_message

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    if subscribe_enabled:
        # Two independent triggers on two independent topic filters --
        # each gets its own callback rather than one handler branching on
        # topic shape, so enter/leave logic stays easy to read and change
        # separately.
        client.message_callback_add(enter_topic, make_on_message("enter"))
        client.message_callback_add(leave_topic, make_on_message("leave"))
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    if config["mqtt"].get("username"):
        client.username_pw_set(config["mqtt"]["username"],
                                config["mqtt"].get("password"))
    return client


def start_http_trigger(cameras: dict, config: dict, bus: JobBus, log) -> threading.Thread:
    """Starts the HTTP webhook trigger server (config "http_trigger") in a
    background thread, served by waitress rather than Flask's own app.run()
    -- Flask's built-in server is single-threaded-by-default and explicitly
    labels itself "not for production use"; waitress is a pure-Python
    production WSGI server (no C extension, so it survives PyInstaller
    freezing the same as everything else here) that handles concurrent
    requests properly. Imports are local so Flask/waitress are only
    required when this trigger source is actually turned on."""
    from waitress import serve
    from .http_trigger import build_http_trigger_app

    http_cfg = config["http_trigger"]
    app = build_http_trigger_app(cameras, config, bus)

    def _run():
        try:
            serve(app, host=http_cfg["host"], port=http_cfg["port"],
                  threads=http_cfg["threads"])
        except OSError as e:
            # e.g. port already in use -- surfaced from a background
            # thread, so log it explicitly rather than letting it vanish
            # into an unhandled-thread-exception with no context.
            log.error(f"HTTP trigger server failed to start on "
                      f"{http_cfg['host']}:{http_cfg['port']}: {e}")

    t = threading.Thread(target=_run, daemon=True, name="http-trigger")
    t.start()
    log.info(f"HTTP trigger listening on http://{http_cfg['host']}:"
              f"{http_cfg['port']}{http_cfg['path']} "
              f"(threads={http_cfg['threads']}, "
              f"enter rule codes={http_cfg['enter_rule_codes']}, "
              f"exit rule codes={http_cfg['exit_rule_codes']})")
    return t


def main():
    try:
        config = load_config()
    except ConfigError as e:
        print(f"[SETUP] FATAL: {e}", flush=True)
        raise SystemExit(1)

    alpr = config["alpr"]
    num_workers = config["service"]["num_workers"]
    queue_max = config["service"]["queue_max"]
    troubleshooting = alpr["diagnostics_mode"] == "troubleshooting"
    # diagnostics_mode is a one-variable shortcut: "troubleshooting" forces
    # verbose logging regardless of logging.level, so switching one setting
    # gets both the extra debug images (see worker.py) and the logs to
    # explain them, without also having to remember to bump logging.level.
    log_cfg = dict(config["logging"])
    if troubleshooting:
        log_cfg["level"] = "DEBUG"
    configure_logging(log_cfg)
    log = get_logger("SETUP")

    log.info("=" * 60)
    log.info("ALPR MQTT service starting")
    log.info(f"mqtt={config['mqtt']['host']}:{config['mqtt']['port']} "
              f"bay_segment_index={config['mqtt']['bay_segment_index']}")
    log.info(f"  enter: trigger='{config['mqtt']['enter_subscribe_topic']}' "
              f"results='{config['mqtt']['enter_result_topic_prefix']}/<bay>'")
    log.info(f"  leave: trigger='{config['mqtt']['leave_subscribe_topic']}' "
              f"results='{config['mqtt']['leave_result_topic_prefix']}/<bay>'")
    log.info(f"event_filter: class_types={config['event_filter']['class_types']} "
              f"min_likelihood={config['event_filter']['min_likelihood']} "
              f"(applies to mqtt triggers only -- http_trigger has no VCA "
              f"classification to filter on)")

    mqtt_trigger_enabled = config["mqtt"]["trigger_enabled"]
    http_trigger_enabled = config["http_trigger"]["enabled"]
    bay_state_enabled = config["bay_state_engine"]["enabled"]
    if not mqtt_trigger_enabled and not http_trigger_enabled and not bay_state_enabled:
        log.error("no trigger source enabled -- set mqtt.trigger_enabled, "
                  "http_trigger.enabled, and/or bay_state_engine.enabled "
                  "to true in config.json")
        raise SystemExit(1)
    log.info(f"trigger sources: mqtt={mqtt_trigger_enabled} "
              f"http={http_trigger_enabled} bay_state_engine={bay_state_enabled}")

    log.info(f"workers={num_workers} queue_max={queue_max} "
              f"cooldown={alpr['cooldown_sec']}s "
              f"collect_timeout={alpr['collection_timeout']}s")
    log.info(f"samples: raw<={alpr['max_raw_samples']} best={alpr['best_samples']} "
              f"min_plate={alpr['min_plate_width']}x{alpr['min_plate_height']} "
              f"center_limit={alpr['center_distance_limit']}")
    log.info(f"diagnostics_mode={alpr['diagnostics_mode']} "
              f"audit_retention={alpr['audit_retention_days']}d "
              f"log_level={log_cfg['level']}"
              + (" (forced DEBUG by diagnostics_mode)" if troubleshooting else ""))
    log.info(f"snapshot: url_template={config['snapshot']['url_template']} "
              f"connect_timeout={config['snapshot']['connect_timeout_ms']}ms "
              f"read_timeout={config['snapshot']['read_timeout_ms']}ms "
              f"expected_frame="
              f"{alpr['expected_frame_width']}x{alpr['expected_frame_height']}")
    check_snapshot_credentials(config, log)
    check_bay_monitor_config(config, log)
    check_bay_state_config(config, log)

    bay_monitor_cfg = config["bay_monitor"]
    log.info(f"bay_monitor: enabled={bay_monitor_cfg['enabled']}"
              + (f" ollama={bay_monitor_cfg['ollama_host']} "
                 f"model={bay_monitor_cfg['ollama_model']} "
                 f"classify_interval={bay_monitor_cfg['classify_interval_sec']}s"
                 if bay_monitor_cfg["enabled"] else ""))
    log.info(f"bay_state_engine: enabled={bay_state_enabled}")

    try:
        cameras = load_cameras(CAMERAS_FILE)
    except CamerasError as e:
        log.error(str(e))
        raise SystemExit(1)

    enabled = [b for b, c in cameras.items() if c.get('enabled', True)]
    log.info(f"Loaded {len(cameras)} cameras from {CAMERAS_FILE} "
              f"({len(enabled)} enabled: {enabled})")
    check_camera_ip_fields(cameras, log)
    for bay, cam in cameras.items():
        log.debug(f"camera '{bay}': ip={cam.get('ip')} "
                  f"roi={cam.get('roi')} enabled={cam.get('enabled', True)}")

    AUDIT_DIR.mkdir(exist_ok=True)
    prune_audit(AUDIT_DIR, alpr['audit_retention_days'])

    bus = JobBus(queue_max=queue_max, cooldown_sec=alpr['cooldown_sec'])
    # MQTT is always connected (results are always published over it);
    # subscribe_enabled only controls whether it also *subscribes* to the
    # VCA event topics as a trigger source.
    client = build_mqtt(cameras, config, bus, subscribe_enabled=mqtt_trigger_enabled)

    def publish(topic, payload):
        client.publish(topic, payload, qos=1)

    if http_trigger_enabled:
        try:
            start_http_trigger(cameras, config, bus, log)
        except ValueError as e:
            log.error(str(e))
            raise SystemExit(1)

    # Built before the monitor starts, so its on_status hook (below) and
    # every Worker's on_result hook can both be wired to the same engine.
    state_engine = None
    if bay_state_enabled:
        from .bay_state import BayStateEngine
        state_engine = BayStateEngine(cameras, config, bus, publish)

    # Every background component that needs telling to stop registers its
    # Event here, so shutdown iterates one list instead of accumulating a
    # per-component `if x is not None: x.set()` line each time one is added.
    stoppables = []
    if bay_monitor_cfg["enabled"]:
        from .bay_monitor import start_bay_monitor
        _, bay_monitor_stop = start_bay_monitor(
            cameras, config, publish,
            on_status=state_engine.on_status if state_engine else None,
            audit_dir=AUDIT_DIR)
        stoppables.append(bay_monitor_stop)

    # Shared by every worker so their first-run model loading (in
    # particular PaddleOCR's download-if-missing check against the one
    # shared local model folder) is serialized instead of racing --
    # see Worker.model_load_lock.
    model_load_lock = threading.Lock()
    workers = [
        Worker(i + 1, bus.jobs, cameras, config, publish, bus, AUDIT_DIR,
               model_load_lock,
               on_result=state_engine.on_alpr_result if state_engine else None)
        for i in range(num_workers)
    ]
    for w in workers:
        w.start()

    def _handle_signal(signum, frame):
        log.info(f"received signal {signum}, shutting down")
        # Stop the background components before dropping the MQTT
        # connection -- otherwise the bay monitor keeps fetching snapshots
        # and publishing to a client that's already gone. Each exits at
        # its next check point (nothing interrupts a request in flight).
        for stop_event in stoppables:
            stop_event.set()
        client.disconnect()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    connect_with_retry(client, config['mqtt']['host'], config['mqtt']['port'],
                        keepalive=60, log=get_logger("MQTT"))
    log.info("Service running. Send SIGINT/SIGTERM to stop.")
    client.loop_forever()
    log.info("Shut down cleanly")
