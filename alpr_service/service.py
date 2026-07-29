"""Process entry point: wires config, logging, cameras, MQTT, and workers
together. Import and call main() from the top-level entry script."""
import signal

import paho.mqtt.client as mqtt

from .audit import prune_audit
from .cameras import CamerasError, load_cameras
from .config import BASE_DIR, ConfigError, load_config
from .logging_setup import configure_logging, get_logger
from .mqtt_bus import JobBus, connect_with_retry, extract_event, matches_class_filter
from .worker import Worker

NUM_WORKERS = 3     # ~= max concurrent dockings; ~2GB RAM each
QUEUE_MAX = 50
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


def build_mqtt(cameras: dict, config: dict, bus: JobBus) -> mqtt.Client:
    log = get_logger("MQTT")
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):
        client = mqtt.Client()

    trigger_topic = config["mqtt"]["subscribe_topic"]
    class_types = config["event_filter"]["class_types"]
    min_likelihood = config["event_filter"]["min_likelihood"]

    def on_connect(client, userdata, flags, rc, *args):
        log.info(f"connected rc={rc}, subscribing {trigger_topic}")
        client.subscribe(trigger_topic, qos=1)

    def on_disconnect(client, userdata, *args):
        log.warning("disconnected -- paho will auto-reconnect")

    def on_message(client, userdata, msg):
        event = extract_event(msg.topic, msg.payload)
        if event is None:
            return
        bay = event["bay"]
        if bay not in cameras:
            log.warning(f"event for unknown bay '{bay}' (topic {msg.topic})")
            return
        if not cameras[bay].get("enabled", True):
            return

        matched, cls_text, likelihood = matches_class_filter(
            event["data"], class_types, min_likelihood)
        if not matched:
            log.debug(f"({bay}) event discarded: no detection matching "
                      f"class in {class_types} at likelihood>={min_likelihood} "
                      f"(topic {msg.topic})")
            return
        log.info(f"({bay}) event matched class='{cls_text}' "
                 f"likelihood={likelihood:.2f}")

        queued_event = {
            "bay": bay,
            "event_time": event["event_time"],
            "detected_class": cls_text,
            "detected_likelihood": likelihood,
        }
        if bus.try_enqueue(queued_event):
            log.info(f"({bay}) event queued for processing")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    if config["mqtt"].get("username"):
        client.username_pw_set(config["mqtt"]["username"],
                                config["mqtt"].get("password"))
    return client


def main():
    try:
        config = load_config()
    except ConfigError as e:
        print(f"[SETUP] FATAL: {e}", flush=True)
        raise SystemExit(1)

    configure_logging(config["logging"])
    log = get_logger("SETUP")

    log.info("=" * 60)
    log.info("ALPR MQTT service starting")
    log.info(f"mqtt={config['mqtt']['host']}:{config['mqtt']['port']} "
              f"trigger='{config['mqtt']['subscribe_topic']}' "
              f"results='{config['mqtt']['result_topic_prefix']}/<bay>'")
    log.info(f"event_filter: class_types={config['event_filter']['class_types']} "
              f"min_likelihood={config['event_filter']['min_likelihood']}")

    alpr = config["alpr"]
    log.info(f"workers={NUM_WORKERS} queue_max={QUEUE_MAX} "
              f"cooldown={alpr['cooldown_sec']}s "
              f"collect_timeout={alpr['collection_timeout']}s")
    log.info(f"samples: raw<={alpr['max_raw_samples']} best={alpr['best_samples']} "
              f"min_plate={alpr['min_plate_width']}x{alpr['min_plate_height']} "
              f"center_limit={alpr['center_distance_limit']}")
    log.info(f"debug_images={alpr['debug_save_images']} "
              f"audit_retention={alpr['audit_retention_days']}d "
              f"log_level={config['logging']['level']}")
    log.info(f"snapshot: url_template={config['snapshot']['url_template']} "
              f"connect_timeout={config['snapshot']['connect_timeout_ms']}ms "
              f"read_timeout={config['snapshot']['read_timeout_ms']}ms "
              f"expected_frame="
              f"{alpr['expected_frame_width']}x{alpr['expected_frame_height']}")
    check_snapshot_credentials(config, log)

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

    bus = JobBus(queue_max=QUEUE_MAX, cooldown_sec=alpr['cooldown_sec'])
    client = build_mqtt(cameras, config, bus)

    def publish(topic, payload):
        client.publish(topic, payload, qos=1)

    workers = [
        Worker(i + 1, bus.jobs, cameras, config, publish, bus, AUDIT_DIR)
        for i in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()

    def _handle_signal(signum, frame):
        log.info(f"received signal {signum}, shutting down")
        client.disconnect()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    connect_with_retry(client, config['mqtt']['host'], config['mqtt']['port'],
                        keepalive=60, log=get_logger("MQTT"))
    log.info("Service running. Send SIGINT/SIGTERM to stop.")
    client.loop_forever()
    log.info("Shut down cleanly")
