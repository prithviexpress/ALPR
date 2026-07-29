"""MQTT event parsing and the enqueue/cooldown debounce shared by the
service and its workers."""
import json
import queue
import threading
import time

from .logging_setup import get_logger

log = get_logger("QUEUE")


def extract_event(topic: str, payload: bytes):
    parts = topic.split('/')
    if len(parts) < 6:
        return None
    if parts[4] != 'LineDetector':
        return None
    if parts[5] != 'Crossed':
        return None
    bay = parts[1]
    try:
        data = json.loads(payload.decode())
        event_time = data.get("UtcTime")
    except Exception:
        event_time = None
    return {"bay": bay, "event_time": event_time}


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
