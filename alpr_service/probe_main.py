"""Entry point for the model effectiveness probe (06_Model_Probe.py).

Kept separate from service.py on purpose: that module wires up workers,
the job bus, trigger subscriptions and the bay state engine, none of
which the probe wants. Importing it would drag all of that in -- and
risk the probe accidentally acquiring the service's side effects, which
is exactly what a measuring instrument must not do.
"""
import signal
import threading

import paho.mqtt.client as mqtt

from .cameras import CamerasError, load_cameras
from .config import BASE_DIR, ConfigError, load_config
from .logging_setup import configure_logging, get_logger
from .mqtt_bus import connect_with_retry

AUDIT_DIR = BASE_DIR / "audit"
CAMERAS_FILE = BASE_DIR / "cameras.json"


def build_publisher(config: dict, log):
    """A publish-only MQTT client, or None if publishing is off or the
    broker can't be reached.

    A broker problem must NOT stop the probe: the JSONL record and the
    saved images are the measurement, and MQTT is a convenience on top.
    Losing the run because a broker was down would be the worst possible
    trade."""
    if not config["model_probe"]["publish_enabled"]:
        log.info("model_probe.publish_enabled=false -- recording to disk only")
        return None, None
    try:
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):
            client = mqtt.Client()
        mq = config["mqtt"]
        if mq.get("username"):
            client.username_pw_set(mq["username"], mq.get("password"))
        connect_with_retry(client, mq["host"], mq["port"], log=log,
                           max_attempts=config["model_probe"]["mqtt_max_connect_attempts"])
        client.loop_start()
        log.info(f"publishing probe results to "
                 f"{config['model_probe']['topic_prefix']}/<bay>")
        return client, client.publish
    except Exception:
        log.warning("could not connect to MQTT -- continuing without "
                    "publishing; results are still written to disk",
                    exc_info=True)
        return None, None


def main():
    try:
        config = load_config()
    except ConfigError as e:
        raise SystemExit(f"Config error: {e}")

    configure_logging(config)
    log = get_logger("PROBE")

    try:
        cameras = load_cameras(CAMERAS_FILE)
    except CamerasError as e:
        raise SystemExit(f"Cameras error: {e}")

    enabled = {b: c for b, c in cameras.items() if c.get("enabled", True)}
    if not enabled:
        raise SystemExit("No enabled cameras in cameras.json -- nothing to probe")
    log.info(f"probing {len(enabled)} enabled camera(s): "
             f"{', '.join(sorted(enabled))}")

    client, publish = build_publisher(config, log)

    from .model_probe import ModelProbe
    try:
        probe = ModelProbe(cameras, config, publish_fn=publish,
                           audit_dir=AUDIT_DIR)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    stop_event = threading.Event()

    def _handle_signal(signum, frame):
        # Ctrl-C must still produce the final summary -- a run whose
        # totals are lost on exit measured nothing.
        log.info(f"signal {signum} received, finishing the current frame "
                 f"and writing the final summary")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        probe.run(stop_event)
    finally:
        if client is not None:
            try:
                client.loop_stop()
                client.disconnect()
            except Exception:
                pass
        log.info(f"probe output written to {probe.out_dir}")
