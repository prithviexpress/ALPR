"""Per-bay state engine -- fuses bay_monitor's continuous activity
classification with ALPR plate reads into one authoritative session per
bay, and uses bay_monitor's own empty<->occupied transitions (not the
MQTT/HTTP trigger source) to decide enter vs leave.

Why this exists: an MQTT/HTTP trigger fires once, at one instant, and if
that instant's ALPR read is ruined -- e.g. by a bay/cargo door being open,
a known real failure mode -- there's no second chance; the whole session's
identity is lost. bay_monitor is already watching the bay continuously
(at least once a minute per classify_interval_sec), so this engine uses
that to retry ALPR reads across the entire stay and only closes a session
(publishes a leave result) once the truck is confirmed gone, using
whatever plate got confirmed at ANY point during the visit -- not just
whatever a single trigger instant happened to catch.

Wiring: bay_monitor.py calls on_status(bay, status, timestamp, zoomed_in)
after every classification; this engine reacts to empty<->occupied
transitions by enqueuing ALPR jobs into the existing JobBus/Worker pool
(reusing all of worker.py's collection/OCR/voting pipeline unchanged)
with direction always "enter" while a session is open. Worker calls
on_alpr_result(reply) after every completed job so this engine can
capture a confirmed plate. Departure is decided by bay_monitor's own
zoomed_in flag going False (its empty_debounce_count, not a second copy
of that threshold here -- see on_status below for why), at which point
this engine publishes the "leave" result itself, directly -- bypassing
JobBus/Worker entirely, since by the time a bay reads "empty" the truck
is no longer in frame for a fresh read; the leave result reports the
session's last confirmed plate instead.

Depends entirely on bay_monitor being enabled (its status stream is the
only thing driving this) -- see check_bay_state_config() in service.py.

State is in-memory only -- a restart loses any in-progress session.
"""
import json
import threading
from datetime import datetime, timezone

from .logging_setup import get_logger


class BaySession:
    def __init__(self, bay: str):
        self.bay = bay
        self.status = None
        self.direction = None       # "enter" while a session is open, else None
        self.plate = None
        self.confidence = 0.0
        self.confirmed = False      # True once a SUCCESS read has landed
        self.session_start = None
        self.read_attempts = 0


class BayStateEngine:
    def __init__(self, cameras: dict, config: dict, bus, publish_fn):
        self.log = get_logger("BAY_STATE")
        self.cameras = cameras
        self.config = config
        self.bus = bus
        self.publish = publish_fn
        self.lock = threading.Lock()
        self.sessions = {bay: BaySession(bay) for bay in cameras}
        alpr_cfg = config["alpr"]
        self.publish_no_valid_plate = alpr_cfg.get("publish_no_valid_plate", True)
        self.unknown_plate_value = alpr_cfg.get("unknown_plate_value", "UNKNOWN")
        self.leave_topic_prefix = config["mqtt"].get(
            "leave_result_topic_prefix", "alpr_result/leave")
        # "Occupied" is derived from bay_monitor's own configured
        # vocabulary (everything that isn't its empty_status) rather than
        # a hardcoded set here -- a custom bay_monitor.status_values would
        # otherwise never match, and no session would ever open or close.
        bm_cfg = config.get("bay_monitor", {})
        empty_status = bm_cfg.get("empty_status", "empty")
        status_values = bm_cfg.get(
            "status_values", ["empty", "occupied", "unloading", "loading", "idle"])
        self.occupied_statuses = {s for s in status_values if s != empty_status}

    def on_status(self, bay: str, status: str, timestamp: str, zoomed_in: bool):
        """Called by bay_monitor after every classification. zoomed_in is
        bay_monitor's OWN post-transition state -- True while it's still
        watching this bay closely, False the moment it's decided (via its
        own empty_debounce_count) to give up and revert to baseline
        scanning. That transition, not a separately re-counted debounce
        here, is what this engine treats as "the truck is confirmed
        gone" -- re-deriving a second copy of that threshold risks the
        two falling out of sync (e.g. this engine waiting for more
        consecutive empties than bay_monitor itself does, in which case
        bay_monitor would stop calling this at all once it reverts,
        leaving a session open forever)."""
        with self.lock:
            session = self.sessions.setdefault(bay, BaySession(bay))
            session.status = status

            if status in self.occupied_statuses:
                # Whether this is a NEW arrival is decided purely by
                # whether a session is already open -- NOT by comparing
                # against the previous classification. Keying off the
                # previous status was a real bug: a single transient
                # "empty" mid-visit (a VLM blip, or a door-open frame that
                # reads as empty) sits below bay_monitor's debounce, so no
                # departure fires and the session stays open -- but the
                # next "occupied" then looked like a fresh arrival and
                # wiped the plate already confirmed for this very truck,
                # publishing leave with no truck number. That is precisely
                # the failure this module exists to prevent. With a
                # session open, any occupied read is a continuation.
                if session.direction != "enter":
                    self._on_arrival(session, timestamp)
                elif not session.confirmed:
                    self._retry_read(session, timestamp)
            elif not zoomed_in and session.direction == "enter":
                self._on_departure(session, timestamp)

    def _on_arrival(self, session: BaySession, timestamp: str):
        session.direction = "enter"
        session.plate = None
        session.confidence = 0.0
        session.confirmed = False
        session.session_start = timestamp
        session.read_attempts = 0
        self.log.info(f"({session.bay}) arrival detected -> enter, "
                       f"enqueuing ALPR read")
        self._enqueue_read(session, timestamp)

    def _retry_read(self, session: BaySession, timestamp: str):
        self.log.debug(f"({session.bay}) still occupied, no confirmed plate "
                        f"yet -- retrying ALPR read (attempt "
                        f"{session.read_attempts + 1})")
        self._enqueue_read(session, timestamp)

    def _enqueue_read(self, session: BaySession, timestamp: str):
        queued_event = {
            "bay": session.bay,
            "direction": "enter",
            "event_time": timestamp,
            "detected_class": None,
            "detected_likelihood": None,
        }
        # JobBus's own (bay, direction) cooldown paces retries -- but it
        # also silently REFUSES them, so read_attempts only counts reads
        # that were actually accepted, and a refusal is logged rather than
        # looking like an attempt that came back empty. Worth knowing:
        # with the defaults (cooldown_sec=90 > classify_interval_sec=60)
        # roughly every other retry is refused this way, so effective
        # retry cadence is the cooldown, not the classify interval. Lower
        # alpr.cooldown_sec if you want a retry on every classification.
        if self.bus.try_enqueue(queued_event):
            session.read_attempts += 1
            return True
        self.log.debug(f"({session.bay}) ALPR read not enqueued (JobBus "
                        f"cooldown or queue full) -- will try again on the "
                        f"next classification")
        return False

    def _on_departure(self, session: BaySession, timestamp: str):
        self.log.info(f"({session.bay}) departure detected -> leave "
                       f"(plate={session.plate!r} confirmed={session.confirmed}, "
                       f"{session.read_attempts} read attempt(s) this session)")
        status = "SUCCESS" if session.confirmed else "NO_VALID_PLATE"
        reply = {
            "bay": session.bay,
            "direction": "leave",
            # Never null -- a downstream consumer always gets a string,
            # either the confirmed plate or alpr.unknown_plate_value. The
            # "status" field is what distinguishes a real read from a
            # placeholder.
            "truck_number": session.plate or self.unknown_plate_value,
            "confidence": session.confidence,
            "status": status,
            "event_time": timestamp,
            "ocr_time": datetime.now(timezone.utc).isoformat(),
        }

        if status == "NO_VALID_PLATE" and not self.publish_no_valid_plate:
            self.log.info(f"({session.bay}) no confirmed plate at departure, "
                           f"not publishing leave result "
                           f"(alpr.publish_no_valid_plate=false)")
        else:
            topic = f"{self.leave_topic_prefix}/{session.bay}"
            self.publish(topic, json.dumps(reply, default=str))
            self.log.info(f"({session.bay}) leave published to {topic}")

        # Reset for the next arrival.
        session.direction = None
        session.plate = None
        session.confidence = 0.0
        session.confirmed = False
        session.read_attempts = 0

    def on_alpr_result(self, reply: dict):
        """Called by Worker after every completed job. Only updates a
        session that is (a) known here, (b) currently in an open "enter"
        session, AND (c) the result itself is direction="enter" -- this
        engine only ever enqueues "enter" jobs (see _enqueue_read), so a
        "leave"-direction result can only have come from some other
        source (e.g. the original MQTT/HTTP trigger path, if still
        enabled alongside this engine) and must be ignored rather than
        misapplied to session state it didn't actually establish.
        Checking session.direction alone isn't enough for this -- it
        would still be "enter" while a session is open regardless of
        which direction the incoming result itself claims."""
        bay = reply.get("bay")
        with self.lock:
            session = self.sessions.get(bay)
            if (session is None or session.direction != "enter"
                    or reply.get("direction") != "enter"):
                return
            plate = reply.get("truck_number")
            # status=="SUCCESS" already excludes the placeholder, but
            # guard explicitly so alpr.unknown_plate_value can never be
            # recorded as this session's confirmed plate.
            if (reply.get("status") == "SUCCESS" and plate
                    and plate != self.unknown_plate_value):
                session.plate = plate
                session.confidence = reply.get("confidence", 0.0)
                session.confirmed = True
                self.log.info(f"({bay}) plate confirmed this session: "
                               f"{session.plate} (conf {session.confidence})")
            # NO_VALID_PLATE / ERROR: leave session.plate exactly as it
            # was -- a failed retry must never erase an earlier good read.
