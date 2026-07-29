"""MQTT event parsing and the enqueue/cooldown debounce shared by the
service and its workers."""
import json
import queue
import threading
import time

from .logging_setup import get_logger

log = get_logger("QUEUE")
mqtt_log = get_logger("MQTT")

DEFAULT_TRIGGER_RULES = [{"rule": "LineDetector", "event": "Crossed"}]


def extract_event(topic: str, payload: bytes, bay_segment_index=1,
                   rule_segment_index=4, event_segment_index=5,
                   trigger_rules=None):
    """Map a raw MQTT message to a bay + its parsed VCA payload -- but
    only for topics whose rule/event segments match one of
    `trigger_rules` (config.json "mqtt.trigger_rules", default just
    LineDetector/Crossed). Every other topic on the subscribed wildcard
    (e.g. other Genetec/Bosch rule types) is ignored, logged at DEBUG so
    it's visible rather than silently dropped -- set
    "logging.level": "DEBUG" (or "alpr.diagnostics_mode":
    "troubleshooting") to see what else is coming through and add it to
    trigger_rules if wanted.

    Segment positions are configurable too (config.json "mqtt.
    bay_segment_index"/"rule_segment_index"/"event_segment_index") in
    case your topic layout differs from
    "<prefix>/<bay>/.../<rule>/<event>".

    "data" is the full parsed JSON body (Genetec/Bosch-style, XML-derived:
    "@Attr" for attributes, "#text" for element text), e.g.:

        {"UtcTime": "...", "Data": {"Object": {"Object": {
            "Appearance": {"Class": {"Type": {
                "@Likelihood": 0.91, "#text": "Vehicle"}}}}}}}

    "data" is {} if the payload wasn't valid JSON -- matches_class_filter()
    treats that the same as "no detections", so it's discarded rather than
    raising further down the pipeline.
    """
    if trigger_rules is None:
        trigger_rules = DEFAULT_TRIGGER_RULES
    parts = topic.split('/')
    min_len = max(bay_segment_index, rule_segment_index, event_segment_index) + 1
    if len(parts) < min_len:
        mqtt_log.debug(f"ignored topic '{topic}': only {len(parts)} segments, "
                        f"need at least {min_len}")
        return None

    rule = parts[rule_segment_index]
    event_type = parts[event_segment_index]
    if not any(rule.lower() == r["rule"].lower()
               and event_type.lower() == r["event"].lower()
               for r in trigger_rules):
        mqtt_log.debug(f"ignored topic '{topic}': rule='{rule}' event='{event_type}' "
                        f"not in trigger_rules {trigger_rules}")
        return None

    bay = parts[bay_segment_index]
    try:
        data = json.loads(payload.decode())
    except Exception:
        data = {}
    event_time = data.get("UtcTime")
    return {"bay": bay, "event_time": event_time, "data": data}


def _iter_detections(data: dict):
    """Yield (class_text, likelihood) pairs out of a VCA event payload's
    Data.Object.Object[].Appearance.Class.Type[]. Both "Object" and "Type"
    may be a single dict or a list, depending on how many objects/
    classifications the camera reports in one event."""
    objects = data.get("Data", {}).get("Object", {}).get("Object", [])
    if isinstance(objects, dict):
        objects = [objects]
    elif not isinstance(objects, list):
        objects = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        types = obj.get("Appearance", {}).get("Class", {}).get("Type", [])
        if isinstance(types, dict):
            types = [types]
        elif not isinstance(types, list):
            types = []
        for t in types:
            if isinstance(t, dict):
                yield t.get("#text"), t.get("@Likelihood")


def matches_class_filter(data: dict, class_types, min_likelihood: float):
    """True if any detection in the payload is one of `class_types`
    (case-insensitive) at or above `min_likelihood`. Returns
    (matched: bool, class_text, likelihood) -- the latter two are the
    best-matching detection's values (or None) for logging/audit.

    An empty/falsy `class_types` (config "event_filter.class_types": [])
    disables the filter entirely -- every event matches, regardless of
    what (if anything) it was classified as. Without this special case,
    an empty list would mean the opposite of "no filtering": nothing
    would ever be in the wanted set, so every event would be rejected.
    """
    if not class_types:
        return True, None, None
    wanted = {c.strip().lower() for c in class_types}
    best = (False, None, None)
    for text, likelihood in _iter_detections(data):
        if text is None or likelihood is None:
            continue
        try:
            likelihood = float(likelihood)
        except (TypeError, ValueError):
            continue
        if text.strip().lower() in wanted and likelihood >= min_likelihood:
            if best[2] is None or likelihood > best[2]:
                best = (True, text, likelihood)
    return best


class JobBus:
    """Owns the job queue plus the per-bay active-set/cooldown debounce.

    This used to be three bare module globals (JOBS/ACTIVE/LAST_FIRED)
    plus a lock, with one mutation site (ACTIVE.discard in the worker's
    finally block) that didn't take the lock. Bundling the state here
    means every mutation goes through the same locked methods.
    """

    def __init__(self, queue_max: int, cooldown_sec: int):
        self.jobs: "queue.Queue[dict]" = queue.Queue(maxsize=queue_max)
        self.cooldown_sec = cooldown_sec
        self.active = set()
        self.last_fired = {}
        self._lock = threading.Lock()

    def try_enqueue(self, event: dict) -> bool:
        bay = event['bay']
        now = time.time()
        with self._lock:
            if bay in self.active:
                log.info(f"({bay}) rejected: job already active")
                return False
            remaining = self.cooldown_sec - (now - self.last_fired.get(bay, 0))
            if remaining > 0:
                log.info(f"({bay}) rejected: cooldown, {remaining:.0f}s left")
                return False
            try:
                self.jobs.put_nowait(event)
            except queue.Full:
                log.warning(f"({bay}) rejected: job queue FULL "
                            f"({self.jobs.qsize()}/{self.jobs.maxsize}), dropping event")
                return False
            self.active.add(bay)
            self.last_fired[bay] = now
            log.info(f"({bay}) accepted (queue depth {self.jobs.qsize()}, "
                      f"active {sorted(self.active)})")
            return True

    def release(self, bay: str):
        with self._lock:
            self.active.discard(bay)


def connect_with_retry(client, host, port, keepalive=60, log=log,
                        base_delay=2, max_delay=60, max_attempts=None):
    """Retry the initial MQTT connect with exponential backoff.

    Without this, a broker that's down or not-yet-DNS-resolvable at boot
    (e.g. right after a reboot, or a broker restart racing this service's
    start) raises out of client.connect() and kills the whole process
    before paho's own auto-reconnect (which only covers post-connect
    drops) ever gets a chance to run.
    """
    attempt = 0
    while True:
        try:
            client.connect(host, port, keepalive=keepalive)
            return
        except Exception as e:
            attempt += 1
            log.error(f"MQTT connect attempt {attempt} to {host}:{port} failed: {e}")
            if max_attempts and attempt >= max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            log.info(f"retrying in {delay}s")
            time.sleep(delay)
