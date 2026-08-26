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
import time
from datetime import datetime, timezone
from pathlib import Path

from .logging_setup import get_logger
from .mqtt_bus import make_job
from .results import build_reply, result_topic, should_publish

CSV_FIELDS = ["bay", "bay_status", "door_state", "session_open", "direction",
              "plate", "confidence", "read_attempts", "last_updated"]


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
        self.retry_only_when_plate_visible = \
            config["alpr"]["retry_only_when_plate_visible"]
        self.abandon_when_doors_open = \
            config["alpr"]["abandon_read_when_doors_open"]
        self.abandon_when_docked = config["alpr"]["abandon_read_when_docked"]
        self.read_startup_occupancy = config["alpr"]["read_startup_occupancy"]
        # Bays where the plate has been given up on for the CURRENT visit
        # -- the truck's doors opened before it could be read, and on a
        # reversed-in trailer those doors physically cover the plate, so
        # further attempts photograph a door. Cleared at arrival and at
        # departure, never carried between visits.
        self.read_abandoned = set()
        self.state_topic_prefix = config["mqtt"]["bay_state_topic_prefix"]
        self.notification_topic_prefix = config["mqtt"]["bay_notification_topic_prefix"]
        self.event_topic_prefix = config["mqtt"]["bay_event_topic_prefix"]
        # The three older topics all fire far more often than a consumer
        # who only wants "a truck came in / left, and which truck" needs
        # -- bay_status on every single classification, bay_state on every
        # session transition, bay_notification on every activity change.
        # bay_event (see _publish_event) answers exactly that question in
        # ~3 messages per visit, so it's the only one on by default; the
        # others stay available for anyone already consuming them.
        self.publish_state_topic = config["mqtt"]["publish_bay_state"]
        self.publish_notification_topic = config["mqtt"]["publish_bay_notification"]
        self.publish_event_topic = config["mqtt"]["publish_bay_event"]
        # Per-bay door state as last reported by bay_monitor's truck-model
        # backend, for the CSV and for enriching events. None-valued for
        # the Ollama backend, which doesn't report door state at all.
        self.bay_door_state = {bay: None for bay in cameras}
        # Bays whose "arrived" event went out with no confirmed plate yet
        # -- an "identified" event is published for these if the plate
        # later resolves (see on_alpr_result). Discarded at departure, so
        # a plate confirmed after the truck already left can't emit one.
        self.awaiting_identity = set()
        # Wall-clock arrival time per open session, so the departure event
        # can report how long the truck was at the bay -- the single most
        # asked-for number about a dock visit, and nothing else in this
        # service records it.
        self.arrival_time = {}
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
        if not self.publish_state_topic:
            return
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
        if not self.publish_notification_topic:
            return
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

    def get_session_info(self, bay: str) -> dict:
        """This bay's session state for the webhook's /bay/<bay> query --
        the identity half of the answer (which truck, how confident, how
        long it's been here), which bay_monitor can't know because it
        never sees a plate. Returns an empty dict for an unrecognized bay
        so a caller can merge it unconditionally.

        Takes the lock: a query arrives on the webhook's thread while
        on_status/on_alpr_result may be mutating these same fields from
        the monitor's and workers' threads."""
        with self.lock:
            session = self.sessions.get(bay)
            if session is None:
                return {}
            arrived_at = self.arrival_time.get(bay)
            return {
                "session_open": session.open,
                "truck_number": session.plate or self.unknown_plate_value,
                "plate_confirmed": session.plate is not None,
                "confidence": session.confidence,
                "read_attempts": session.read_attempts,
                "door_state": self.bay_door_state.get(bay),
                "duration_sec": (int(time.time() - arrived_at)
                                 if arrived_at is not None else None),
            }

    def _publish_event(self, bay: str, event: str, timestamp: str, **fields):
        """The lean, consumer-facing topic: exactly the events a dock
        system asked for -- a truck came in, which truck it is, it left --
        and nothing else. Roughly three messages per visit, against the
        dozens the older topics produce between them.

        Three event types:
          arrived     a truck was detected at an empty bay. Carries
                      whatever plate is known (usually none yet -- the
                      ALPR read takes seconds), plus door_state, and
                      alert="door_open_on_arrival" when it showed up with
                      its doors already open.
          identified  the plate resolved for a truck whose "arrived"
                      already went out unidentified. Not published when
                      the plate was already known at arrival, so a
                      consumer never sees a redundant pair.
          departed    the bay emptied out. Carries the truck number this
                      visit concluded with and how long it stayed.

        Every event carries bay/event/timestamp; the rest varies by type
        rather than padding all three with nulls."""
        if not self.publish_event_topic:
            return
        topic = f"{self.event_topic_prefix}/{bay}"
        payload = {"bay": bay, "event": event, "timestamp": timestamp}
        payload.update(fields)
        self.publish(topic, json.dumps(payload, default=str))
        detail = " ".join(f"{k}={v}" for k, v in fields.items()
                          if k != "image_base64")
        self.log.info(f"({bay}) event '{event}' -> {topic} ({detail})")

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
                        "door_state": self.bay_door_state.get(bay) or "",
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
                  image_b64=None, door_state=None, plate_visible=None,
                  phase=None):
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
        without this engine re-fetching or re-classifying anything.

        `door_state` ("open"/"closed", or None on the Ollama backend,
        which doesn't report it) is bay_monitor's RAW per-frame reading,
        not its debounced one -- deliberately, since the only decision
        made from it here is whether a truck arrived with its doors
        already open, which is a claim about the arrival frame itself.

        `plate_visible` is whether the truck model can see a plate right
        now -- used to decide whether a RETRY read is worth spending (see
        _retry_read). None means no information (the Ollama backend
        doesn't report it), which is treated as "go ahead", not as "no
        plate"."""
        with self.lock:
            # Captured BEFORE the update: an empty/absent previous status
            # means this is the first reading this process has ever taken
            # of the bay. If it's already occupied, the truck was parked
            # there before startup -- see _arrival_read_skip_reason. No
            # extra plumbing from bay_monitor needed, since it forces
            # exactly one classify per bay at startup, so its "first look"
            # and this coincide.
            first_ever_status = not self.bay_status.get(bay)
            status_changed = self.bay_status.get(bay) != status
            self.bay_status[bay] = status
            if door_state is not None:
                self.bay_door_state[bay] = door_state
            session = self.sessions.setdefault(bay, BaySession(bay))
            # The truck this activity change is ABOUT: on a departure,
            # session.reset() below wipes session.plate before we'd get
            # to read it, so grab whatever's confirmed so far -- possibly
            # still None/unconfirmed -- before any transition runs.
            truck_number = session.plate or self.unknown_plate_value

            # Passed to _on_arrival rather than re-read there: by the time
            # a departure/arrival transition runs, self.bay_door_state may
            # already describe a later frame.
            arrival_door_state = door_state
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
                    self._on_arrival(
                        session, timestamp, arrival_door_state, phase,
                        self._arrival_read_skip_reason(first_ever_status, phase))
                elif session.plate is None:
                    # Checked before retrying, not after: once the read
                    # window has closed there is nothing left to read, so
                    # the right move is to stop and say so rather than
                    # spend another attempt finding that out again.
                    if self._check_read_window(session, status, door_state,
                                                phase, timestamp):
                        self._retry_read(session, timestamp, plate_visible)
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

    def _on_arrival(self, session: BaySession, timestamp: str,
                     door_state=None, phase=None, skip_reason=None):
        session.reset()
        session.direction = "enter"
        self.arrival_time[session.bay] = time.time()
        # A fresh truck: whatever was given up on last visit says nothing
        # about this one.
        self.read_abandoned.discard(session.bay)
        detail = "".join([f" (doors {door_state})" if door_state else "",
                          f" [{phase}]" if phase else ""])

        if skip_reason:
            # The arrival is still real and still published -- only the
            # ALPR read is skipped, because the plate-readable window
            # closed before this service ever saw the truck. Marked
            # abandoned so the retry path doesn't quietly start reading
            # on the next classification either.
            self.read_abandoned.add(session.bay)
            self.log.info(f"({session.bay}) arrival detected -> enter{detail}, "
                           f"but NOT reading the plate: {skip_reason}")
        else:
            self.log.info(f"({session.bay}) arrival detected -> enter{detail}, "
                           f"enqueuing ALPR read")
            self._enqueue_read(session, timestamp)

        # The requested alert: a truck that backs in with its doors
        # ALREADY open, rather than being opened once docked. Raised on
        # the event itself rather than a separate topic -- one topic to
        # subscribe to was the point -- so a consumer filters on the
        # field instead of managing another subscription.
        alert = "door_open_on_arrival" if door_state == "open" else None
        if alert:
            self.log.warning(f"({session.bay}) truck arrived with its doors "
                              f"ALREADY OPEN")
        # No confirmed plate this early -- the ALPR read was only just
        # enqueued above and takes seconds. Publish the arrival now
        # anyway (a dock system needs to know a truck is there NOW) and
        # follow up with 'identified' when the plate resolves.
        # Only expect an "identified" follow-up if a read is actually
        # going to happen -- otherwise this bay would sit in
        # awaiting_identity for a plate nobody is looking for.
        if not skip_reason:
            self.awaiting_identity.add(session.bay)
        self._publish_event(session.bay, "arrived", timestamp,
                             truck_number=self.unknown_plate_value,
                             door_state=door_state, phase=phase, alert=alert,
                             read_skipped=skip_reason)
        self._publish_state(session, timestamp)

    ABANDON_REASONS = {
        "doors_open": ("its doors are open, so the plate is behind them now"),
        "docked": ("it has finished reversing in, so the plate now faces "
                   "away from the dock camera"),
    }

    def _check_read_window(self, session: BaySession, status: str,
                            door_state, phase, timestamp: str) -> bool:
        """Is there still any point trying to read this truck's plate?

        A dock camera can only read a plate during a narrow window: while
        the truck is still coming IN, with its doors still shut. Two
        things close that window, both permanently for this visit:

          docked      the truck has finished reversing into the bay, so
                      its plate faces away from the camera entirely
                      (alpr.abandon_read_when_docked)
          doors_open  the rear doors have swung OUT and now cover the
                      plate (alpr.abandon_read_when_doors_open)

        Neither is recoverable by retrying: the plate is not in the frame
        any more, and no amount of extra attempts, better OCR or a longer
        collection window changes that. Since max_read_attempts is a
        per-VISIT budget spent on threads shared with genuinely new
        arrivals, continuing past either point is pure waste -- so give
        up and publish the observation instead, which is the honest
        answer: a truck is here, and its plate cannot be read.

        Published exactly once per visit (read_abandoned), so a truck
        that sits docked for an hour of classifications doesn't
        re-announce it on every one. Cleared at arrival and departure so
        the next truck starts fresh.

        Both door_state and phase are None on the ollama backend, which
        reports neither -- that means NO INFORMATION and never abandons,
        the same rule plate_visible follows."""
        if session.bay in self.read_abandoned:
            return False

        reason = None
        if self.abandon_when_doors_open and door_state == "open":
            reason = "doors_open"
        elif self.abandon_when_docked and phase == "docked":
            reason = "docked"
        if reason is None:
            return True

        self.read_abandoned.add(session.bay)
        self.log.info(
            f"({session.bay}) no plate confirmed after "
            f"{session.read_attempts} attempt(s) and {self.ABANDON_REASONS[reason]} "
            f"-- closing the read window for this visit and publishing the "
            f"observation instead")
        self._publish_event(
            session.bay, "plate_unreadable", timestamp,
            truck_number=self.unknown_plate_value,
            reason=reason, door_state=door_state, phase=phase, activity=status,
            read_attempts=session.read_attempts)
        return False

    def _arrival_read_skip_reason(self, first_ever_status: bool, phase):
        """Why this arrival shouldn't trigger an ALPR read at all, or
        None to go ahead.

        Both cases mean the same thing: the entry was already missed, so
        the plate-readable window closed before this service ever looked.

          startup_occupancy  this is the FIRST status ever recorded for
                             the bay and it's already occupied -- the
                             truck was parked there before the process
                             started (bay_monitor forces one classify per
                             bay at startup precisely to notice this).
                             Reading it means an 8-second collection
                             against a truck that docked hours ago.
          already_docked     the very first sighting of this visit is a
                             truck that has already finished reversing in
                             -- entry happened between scans, or the
                             truck model only picked it up once parked.

        Governed by alpr.read_startup_occupancy / abandon_read_when_docked
        respectively."""
        if first_ever_status and not self.read_startup_occupancy:
            return "startup_occupancy"
        if self.abandon_when_docked and phase == "docked":
            return "already_docked"
        return None

    def _retry_read(self, session: BaySession, timestamp: str,
                     plate_visible=None):
        # Don't spend one of this visit's limited attempts on a moment
        # when the truck model can see there is nothing to read. A retry
        # costs a full collection window on one of only
        # service.num_workers threads, and the attempt budget is per
        # visit -- so a retry fired while the plate is out of view is not
        # merely wasted, it makes it likelier the budget runs out before
        # the plate ever comes into view.
        #
        # plate_visible is None on the Ollama backend, meaning NO
        # INFORMATION rather than "no plate" -- that must not be read as
        # a reason to skip, or enabling this would silently stop all
        # retries for anyone not running the truck model.
        #
        # Note this gates RETRIES only. The arrival read always fires
        # (see _on_arrival): entry is when a truck is most likely facing
        # the camera, and the worker fetches its own frames over a whole
        # collection window, so "no plate in this one classified frame"
        # doesn't mean none will be visible during that window.
        if (self.retry_only_when_plate_visible and plate_visible is False):
            self.log.debug(f"({session.bay}) still occupied without a plate, "
                            f"but the truck model sees no plate in frame -- "
                            f"not spending a read attempt "
                            f"({session.read_attempts}/{self.max_read_attempts} "
                            f"used so far)")
            return
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
        duration_sec = None
        arrived_at = self.arrival_time.pop(session.bay, None)
        if arrived_at is not None:
            duration_sec = int(time.time() - arrived_at)
        self._publish_event(session.bay, "departed", timestamp,
                             truck_number=session.plate or self.unknown_plate_value,
                             door_state=self.bay_door_state.get(session.bay),
                             duration_sec=duration_sec)
        # A plate confirmed after this point belongs to no visit -- drop
        # the pending flag so a late ALPR result can't emit an
        # 'identified' for a truck that has already left.
        self.awaiting_identity.discard(session.bay)
        self.read_abandoned.discard(session.bay)

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
                # Answers the "which truck" half of the arrival that
                # already went out unidentified. Discarded (not just
                # checked) so a later re-read of the same visit can't
                # publish a second one.
                if bay in self.awaiting_identity:
                    self.awaiting_identity.discard(bay)
                    self._publish_event(
                        bay, "identified", reply.get("ocr_time", ""),
                        truck_number=session.plate,
                        confidence=session.confidence,
                        door_state=self.bay_door_state.get(bay))
                self._publish_state(session, reply.get("ocr_time", ""))
                self._write_csv()
            # NO_VALID_PLATE / ERROR: leave session.plate exactly as it
            # was -- a failed retry must never erase an earlier good read.
