"""Load and validate config.json, filling in defaults for optional keys.

Everything that used to be a module-level constant in the monolithic
script (cooldown, audit retention, RTSP timeouts, debug image toggle,
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
    "alpr": ["collection_timeout", "frame_skip", "max_raw_samples",
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
            f"{path.name} and fill in your MQTT/RTSP/camera details.")
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

    alpr = cfg["alpr"]
    alpr.setdefault("cooldown_sec", 90)
    alpr.setdefault("audit_retention_days", 14)
    alpr.setdefault("debug_save_images", True)
    alpr.setdefault("min_ocr_conf", 0.35)
    # Expected sensor resolution (e.g. a 5MP camera -> 2592x1944). Leave
    # both null to disable the check. Used to catch a mis-selected RTSP
    # stream (e.g. accidentally pointed at a low-res substream).
    alpr.setdefault("expected_frame_width", None)
    alpr.setdefault("expected_frame_height", None)
    alpr.setdefault("frame_size_tolerance_pct", 10)

    rtsp = cfg.setdefault("rtsp", {})
    # "genetec": pull the feed through a Genetec media-gateway endpoint
    #            keyed by camera GUID (server/port/username/password below,
    #            same fields as R1/R2).
    # "direct":  connect straight to the camera's own RTSP server using its
    #            IP and credentials (see rtsp.direct below), bypassing
    #            Genetec entirely. Per-camera override via cameras.json
    #            "rtsp_mode".
    rtsp.setdefault("mode", "genetec")
    # Always the camera's primary/main stream (full resolution) -- OCR
    # needs the detail a substream throws away. Per-camera override via
    # cameras.json "stream".
    rtsp.setdefault("stream", 1)
    rtsp.setdefault("timeout_ms", 8000)
    # ffmpeg RTSP option name for the socket timeout. Some ffmpeg builds
    # use "stimeout" (older), others "timeout" (newer libavformat) --
    # override here if your build needs the other name.
    rtsp.setdefault("timeout_option_name", "stimeout")
    direct = rtsp.setdefault("direct", {})
    direct.setdefault("port", 554)
    direct.setdefault("username", None)
    direct.setdefault("password", None)
    # Bosch-style main-stream URL by default: single-sensor domes (the
    # 3000i/5000i FLEXIDOME line included) serve stream N at "/videoN" --
    # no NVR-style channel prefix, so {channel} is unused here but still
    # passed through for templates that do need it. Override per camera
    # vendor, e.g. Hikvision: ".../Streaming/Channels/{channel}0{stream}",
    # Dahua: ".../cam/realmonitor?channel={channel}&subtype=0"
    direct.setdefault(
        "url_template",
        "rtsp://{username}:{password}@{ip}:{port}/video{stream}")

    log_cfg = cfg.setdefault("logging", {})
    log_cfg.setdefault("level", "INFO")  # DEBUG / INFO / WARNING / ERROR
    log_cfg.setdefault("console", True)
    log_cfg.setdefault("file", None)     # e.g. "logs/alpr_service.log"
    log_cfg.setdefault("max_bytes", 10 * 1024 * 1024)
    log_cfg.setdefault("backup_count", 5)

    return cfg
