"""HTTP webhook trigger source -- an alternative to the MQTT-subscribed VCA
events in mqtt_bus.py, for cameras that call this service directly on a
rule trigger (e.g. a Bosch camera's built-in "HTTP notification" alarm
task) instead of going through Genetec/MQTT for the trigger itself.

Both trigger sources can be enabled at once (config "mqtt.trigger_enabled"
/ "http_trigger.enabled") -- whichever are on all feed the same JobBus, so
downstream detection, OCR, audit, and per-direction result publishing
(mqtt.enter_result_topic_prefix / leave_result_topic_prefix) are identical
no matter which door the trigger came in. MQTT is still used to publish
results either way; disabling mqtt.trigger_enabled only stops the service
from *subscribing* to the VCA event topics.

The camera identifies itself only by its source IP, so unlike an
event_filter.class_types match there's no VCA classification payload to
key off of here -- which bay fired is resolved purely from cameras.json's
ip field (the same file/field the snapshot fetcher already uses), and
enter/exit is resolved from a numeric rule code the camera's alarm task is
configured to send.
"""
from datetime import datetime, timezone

from flask import Flask, request, jsonify

from .logging_setup import get_logger


def build_camera_ip_map(cameras: dict) -> dict:
    """ip -> bay, the reverse of cameras.json's bay -> {ip, ...}, so an
    inbound webhook hit can be matched back to a bay by its source IP
    without the camera having to tell us its own name. Requires each
    enabled camera's ip to be unique -- if two bays share an ip, a webhook
    hit is ambiguous and this raises rather than guessing."""
    ip_map = {}
    for bay, cam in cameras.items():
        ip = cam.get("ip")
        if not ip:
            continue
        if ip in ip_map:
            raise ValueError(
                f"cameras.json has two bays ('{ip_map[ip]}' and '{bay}') "
                f"sharing ip {ip} -- http_trigger needs a unique ip per "
                f"bay to tell which bay a webhook hit came from")
        ip_map[ip] = bay
    return ip_map


def classify_rule(rule: int, enter_codes: set, exit_codes: set):
    if rule in enter_codes:
        return "enter"
    if rule in exit_codes:
        return "leave"
    return None


def build_http_trigger_app(cameras: dict, config: dict, bus) -> Flask:
    log = get_logger("HTTP_TRIGGER")
    http_cfg = config["http_trigger"]
    ip_map = build_camera_ip_map(cameras)
    rule_param = http_cfg["rule_param"]
    enter_codes = set(http_cfg["enter_rule_codes"])
    exit_codes = set(http_cfg["exit_rule_codes"])

    app = Flask(__name__)

    @app.route(http_cfg["path"], methods=["GET"])
    def alert():
        camera_ip = request.remote_addr
        rule_raw = request.args.get(rule_param)
        try:
            rule = int(rule_raw)
        except (TypeError, ValueError):
            log.warning(f"webhook hit from {camera_ip}: missing/non-numeric "
                        f"'{rule_param}' param ({rule_raw!r}), ignoring")
            return jsonify({"status": "ignored", "reason": "bad_rule"}), 400

        bay = ip_map.get(camera_ip)
        if bay is None:
            log.warning(f"webhook hit from unrecognized camera ip "
                        f"{camera_ip} (not in cameras.json), ignoring")
            return jsonify({"status": "ignored",
                             "reason": "unknown_camera"}), 404

        if not cameras[bay].get("enabled", True):
            return jsonify({"status": "ignored",
                             "reason": "camera_disabled"}), 200

        direction = classify_rule(rule, enter_codes, exit_codes)
        if direction is None:
            log.debug(f"({bay}) webhook rule={rule} not in "
                      f"enter={sorted(enter_codes)}/exit={sorted(exit_codes)}"
                      f", ignoring")
            return jsonify({"status": "ignored",
                             "reason": "unmapped_rule"}), 200

        queued_event = {
            "bay": bay,
            "direction": direction,
            "event_time": datetime.now(timezone.utc).isoformat(),
            "detected_class": None,
            "detected_likelihood": None,
        }
        queued = bus.try_enqueue(queued_event)
        log.info(f"({bay}/{direction}) webhook event "
                 f"{'queued' if queued else 'skipped (cooldown)'} "
                 f"(rule={rule} from {camera_ip})")
        return jsonify({"status": "queued" if queued else "cooldown_skip",
                         "bay": bay, "direction": direction}), 200

    return app
