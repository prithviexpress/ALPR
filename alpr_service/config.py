"""Load and validate config.json, filling in defaults for optional keys.

Everything that used to be a module-level constant in the monolithic
script (cooldown, audit retention, snapshot timeouts, debug image toggle,
expected frame size, log level, ...) is a config key now, so ops can
retune the service without a code change/redeploy.
"""
import json
from pathlib import Path

# Repo root: this file lives in alpr_service/, so the root is one level up.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"

REQUIRED_KEYS = {
    "mqtt": ["host", "port", "subscribe_topic", "result_topic_prefix"],
    "alpr": ["collection_timeout", "max_raw_samples",
             "best_samples", "min_plate_width", "min_plate_height",
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

    cfg.setdefault("model_path", "best.pt")

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
    alpr.setdefault("debug_save_images", True)
    alpr.setdefault("min_ocr_conf", 0.35)
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

    log_cfg = cfg.setdefault("logging", {})
    log_cfg.setdefault("level", "INFO")  # DEBUG / INFO / WARNING / ERROR
    log_cfg.setdefault("console", True)
    log_cfg.setdefault("file", None)     # e.g. "logs/alpr_service.log"
    log_cfg.setdefault("max_bytes", 10 * 1024 * 1024)
    log_cfg.setdefault("backup_count", 5)

    return cfg
