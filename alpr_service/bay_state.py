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

Wiring: bay_monitor.py calls
on_status(bay, status, timestamp, occupied, departed) after every
classification -- it reports those two booleans already interpreted,
since it owns both the status vocabulary and the departure debounce they
derive from. On "occupied" with no session open this engine enqueues an
ALPR job through the normal JobBus/Worker pool (reusing worker.py's
collection/OCR/voting pipeline unchanged), retrying while the plate stays
unconfirmed up to alpr.max_read_attempts; Worker calls on_alpr_result()
after every completed job so a confirmed plate can be captured. On
"departed" it emits the "leave" result itself rather than queueing a job,
since by then the truck is out of frame and there is nothing left to
photograph -- the payload, topic and suppression rule all come from
results.py, the same contract Worker publishes under.

Depends entirely on bay_monitor being enabled (its status stream is the
only thing driving this) -- see check_bay_state_config() in service.py.

State is in-memory only -- a restart loses any in-progress session.
"""
import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .logging_setup import get_logger
from .mqtt_bus import make_job
from .results import build_reply, result_topic, should_publish

CSV_FIELDS = ["bay", "bay_status", "session_open", "direction", "plate",
              "confidence", "read_attempts", "last_updated"]


class BaySession:
    """One truck's visit. `plate` carries the whole story: None means no
    valid read has landed yet, non-None means confirmed. There is
    deliberately no separate `confirmed` flag -- it would be a second
    field meaning exactly `plate is not None`, hand-synced at three reset
    sites, and a missed update would publish SUCCESS with no plate (or a
    plate marked NO_VALID_PLATE), which is the precise corruption this
    module exists to prevent."""

    def __init__(self, bay: str):
        self.bay = bay
        self.direction = None       # "enter" while a session is open, else None
        self.plate = None
        self.confidence = 0.0
        self.read_attempts = 0

    @property
    def open(self) -> bool:
        return self.direction == "enter"

    def reset(self):
        self.direction = None
        self.plate = None
        self.confidence = 0.0
        self.read_attempts = 0


class BayStateEngine:
    def __init__(self, cameras: dict, config: dict, bus, publish_fn,
                 audit_dir: Path = None):
        self.log = get_logger("BAY_STATE")
        self.config = config
        self.bus = bus
        self.publish = publish_fn
        self.lock = threading.Lock()
        self.sessions = {bay: BaySession(bay) for bay in cameras}
        # bay_monitor's raw activity classification (empty/occupied/
        # unloading/...) per bay -- tracked here purely for the CSV
        # snapshot below, since on_status() doesn't otherwise persist it
        # anywhere this engine can read back.
        self.bay_status = {bay: "" for bay in cameras}
        self.unknown_plate_value = config["alpr"]["unknown_plate_value"]
        self.max_read_attempts = config["alpr"]["max_read_attempts"]
        self.state_topic_prefix = config["mqtt"]["bay_state_topic_prefix"]
        self.notification_topic_prefix = config["mqtt"]["bay_notification_topic_prefix"]
        # A one-row-per-bay CSV snapshot of current state, rewritten in
        # full on every on_status()/on_alpr_result() call -- MQTT
        # requires a client and a subscription just to see "what's
        # happening right now"; this is a file anyone can just open
        # (Excel, Notepad, `watch cat`) and have it always show the
        # latest state, no tooling required. None (no audit_dir passed,
        # or bay_state_engine.save_state_csv turned off) just skips it.
        self.csv_path = None
        if audit_dir is not None and config["bay_state_engine"]["save_state_csv"]:
            self.csv_path = audit_dir / "bay_state.csv"

    def _publish_state(self, session: BaySession, timestamp: str):
        """Live view of this engine's own state, separate from both
        bay_monitor's raw activity status (empty/occupied/...) and the
        final enter/leave result (which only ever fires once, at the end
        of a visit). Fired on every state change -- arrival, a plate
        getting confirmed, departure -- so a downstream consumer (or
        someone troubleshooting) can see the engine reacting in real
        time instead of waiting for a visit to conclude.

        occupancy_status/activity are included here too (not just on
        mqtt.bay_notification_topic_prefix) since this is the topic
        someone watching the engine is most likely already subscribed
        to -- "open" alone reads as session-lifecycle jargon, not the
        occupied/empty answer that's actually wanted. activity is
        bay_monitor's own raw word (e.g. "loading"/"idle"), which may be
        finer than the occupied/empty occupancy_status."""
        topic = f"{self.state_topic_prefix}/{session.bay}"
        payload = {
            "bay": session.bay,
            "occupancy_status": "occupied" if session.open else "empty",
            "activity": self.bay_status.get(session.bay, ""),
            "open": session.open,
            "direction": session.direction,
            "plate": session.plate,
            "confidence": session.confidence,
            "read_attempts": session.read_attempts,
            "timestamp": timestamp,
        }
        self.publish(topic, json.dumps(payload, default=str))

    def _publish_notification(self, bay: str, status: str, occupied: bool,
                               truck_number: str, comment, image_b64,
                               timestamp: str):
        """The rich, CHANGE-gated notification (see on_status -- only
        called when bay_monitor's classification actually differs from
        the previous one for this bay): occupancy + finer activity +
        whichever truck number is known so far + the vision model's own
        description of what it sees + the exact frame it was shown.
        Deliberately separate from bay_status_topic_prefix (bay_monitor's
        own topic, which fires on every classification regardless of
        change, and carries neither comment nor image to keep that
        high-frequency topic light) and from bay_state_topic_prefix
        (fires on session transitions, not activity changes, and has no
        comment/image either)."""
        topic = f"{self.notification_topic_prefix}/{bay}"
        payload = {
            "bay": bay,
            "occupancy_status": "occupied" if occupied else "empty",
            "activity": status,
            "truck_number": truck_number,
            "comment": comment,
            "image_base64": image_b64,
            "timestamp": timestamp,
        }
        self.publish(topic, json.dumps(payload, default=str))
        self.log.info(f"({bay}) activity changed -> '{status}' "
                       f"(truck_number={truck_number}), notification "
                       f"published to {topic}")

    def _write_csv(self):
        """Overwrite bay_state.csv with the current snapshot of every
        bay -- always the full table, not an append-only log, so the
        file stays small and opening it always shows the latest state
        rather than history to scroll through. Written to a .tmp file
        and atomically renamed into place so a reader (or Excel keeping
        the file open) never sees a half-written row. Best-effort --
        must never take the state engine down over a disk/permission
        problem."""
        if self.csv_path is None:
            return
        try:
            tmp_path = self.csv_path.with_suffix(".csv.tmp")
            now = datetime.now(timezone.utc).isoformat()
            with open(tmp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for bay in sorted(self.sessions):
                    session = self.sessions[bay]
                    writer.writerow({
                        "bay": bay,
                        "bay_status": self.bay_status.get(bay, ""),
                        "session_open": session.open,
                        "direction": session.direction or "",
                        "plate": session.plate or "",
                        "confidence": session.confidence,
                        "read_attempts": session.read_attempts,
                        "last_updated": now,
                    })
            tmp_path.replace(self.csv_path)
        except Exception:
            self.log.warning("failed to write bay_state.csv", exc_info=True)

    def on_status(self, bay: str, status: str, timestamp: str,
                  occupied: bool, departed: bool, comment=None,
                  image_b64=None):
        """Called by bay_monitor after every classification.

        `occupied` and `departed` are bay_monitor's OWN interpretation of
        the reading, not raw state for this engine to re-derive: it owns
        `empty_status`, so it decides what counts as occupied, and it owns
        `empty_debounce_count`, so it decides when a bay has truly emptied
        out. Re-deriving either here would mean two copies of the same
        threshold that can drift -- and if this engine ever waited for
        more consecutive empties than bay_monitor does, bay_monitor would
        stop reporting entirely once it reverted to baseline scanning and
        the session would stay open forever.

        `comment` and `image_b64` are the vision model's free-text
        description and the exact frame it was shown -- passed straight
        through from bay_monitor for the rich notification below,
        without this engine re-fetching or re-classifying anything."""
        with self.lock:
            status_changed = self.bay_status.get(bay) != status
            self.bay_status[bay] = status
            session = self.sessions.setdefault(bay, BaySession(bay))
            # The truck this activity change is ABOUT: on a departure,
            # session.reset() below wipes session.plate before we'd get
            # to read it, so grab whatever's confirmed so far -- possibly
            # still None/unconfirmed -- before any transition runs.
            truck_number = session.plate or self.unknown_plate_value

            if occupied:
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
                if not session.open:
                    self._on_arrival(session, timestamp)
                elif session.plate is None:
                    self._retry_read(session, timestamp)
            elif departed and session.open:
                self._on_departure(session, timestamp)

            if status_changed:
                self._publish_notification(bay, status, occupied,
                                            truck_number, comment,
                                            image_b64, timestamp)

            # Rewritten after every classification, not just at
            # transitions -- this is what makes the CSV a reliable "is
            # this thing even alive" check: its last_updated column
            # moves every time bay_monitor classifies ANY bay, whether
            # or not anything actually changed.
            self._write_csv()

    def _on_arrival(self, session: BaySession, timestamp: str):
        session.reset()
        session.direction = "enter"
        self.log.info(f"({session.bay}) arrival detected -> enter, "
                       f"enqueuing ALPR read")
        self._enqueue_read(session, timestamp)
        self._publish_state(session, timestamp)

    def _retry_read(self, session: BaySession, timestamp: str):
        # Capped: the truck is stationary and the camera fixed, so attempt
        # N photographs the same obscured plate attempt 1 did. Left
        # uncapped, a truck parked for hours with an unreadable plate
        # re-runs a full collection every cooldown window indefinitely,
        # each one holding one of only service.num_workers threads and
        # starving genuinely new arrivals.
        if session.read_attempts >= self.max_read_attempts:
            self.log.debug(f"({session.bay}) still occupied without a valid "
                            f"plate, but {session.read_attempts} read attempts "
                            f"reached (alpr.max_read_attempts) -- not retrying "
                            f"again this visit")
            return
        self.log.debug(f"({session.bay}) still occupied, no confirmed plate "
                        f"yet -- retrying ALPR read (attempt "
                        f"{session.read_attempts + 1}/{self.max_read_attempts})")
        self._enqueue_read(session, timestamp)

    def _enqueue_read(self, session: BaySession, timestamp: str):
        # JobBus's own (bay, direction) cooldown paces retries -- but it
        # also silently REFUSES them, so read_attempts only counts reads
        # that were actually accepted, and a refusal is logged rather than
        # looking like an attempt that came back empty. Worth knowing:
        # with the defaults (cooldown_sec=90 > classify_interval_sec=60)
        # roughly every other retry is refused this way, so effective
        # retry cadence is the cooldown, not the classify interval. Lower
        # alpr.cooldown_sec if you want a retry on every classification.
        if self.bus.try_enqueue(make_job(session.bay, "enter", timestamp,
                                          source="bay_state")):
            session.read_attempts += 1
        else:
            self.log.debug(f"({session.bay}) ALPR read not enqueued (JobBus "
                            f"cooldown or queue full) -- will try again on "
                            f"the next classification")

    def _on_departure(self, session: BaySession, timestamp: str):
        self.log.info(f"({session.bay}) departure detected -> leave "
                       f"(plate={session.plate!r}, {session.read_attempts} "
                       f"read attempt(s) this session)")
        status = "SUCCESS" if session.plate else "NO_VALID_PLATE"
        # Built and published through worker.py's shared result helpers,
        # not hand-rolled here -- this is the same contract Worker emits
        # for trigger-driven reads, on the same topic family, to the same
        # consumer. Two producers of one message shape is how they drift.
        reply = build_reply(self.config, session.bay, "leave", session.plate,
                             session.confidence, status, timestamp)

        if not should_publish(self.config, status):
            self.log.info(f"({session.bay}) no confirmed plate at departure, "
                           f"not publishing leave result "
                           f"(alpr.publish_no_valid_plate=false)")
        else:
            topic = result_topic(self.config, "leave", session.bay)
            self.publish(topic, json.dumps(reply, default=str))
            self.log.info(f"({session.bay}) leave published to {topic}")

        # Announce the session closing on the live-state topic too --
        # the same plate/confidence/read_attempts this visit concluded
        # with, but open=False now that direction is cleared, so it's
        # visually distinct from the "still open, unconfirmed" state.
        session.direction = None
        self._publish_state(session, timestamp)
        session.reset()

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
            if (session is None or not session.open
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
                self.log.info(f"({bay}) plate confirmed this session: "
                               f"{session.plate} (conf {session.confidence})")
                self._publish_state(session, reply.get("ocr_time", ""))
                self._write_csv()
            # NO_VALID_PLATE / ERROR: leave session.plate exactly as it
            # was -- a failed retry must never erase an earlier good read.
