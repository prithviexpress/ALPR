"""The published-result contract: payload shape, topic, suppression rule.

Its own module rather than living in worker.py because Worker is no
longer the only producer of these messages -- bay_state.py's
BayStateEngine publishes a synthetic "leave" when it sees a bay empty
out. Both go to the same topic family and are read by the same
downstream system, so the shape lives in one place instead of being
reimplemented per producer.

Deliberately free of heavy imports (no cv2/ultralytics/paddleocr), so a
light consumer like bay_state.py can use it without pulling the whole
inference stack into memory just to format a message.
"""
from datetime import datetime, timezone


def result_topic(config: dict, direction: str, bay: str) -> str:
    prefix = config['mqtt'].get(f"{direction}_result_topic_prefix")
    if not prefix:
        # Shouldn't happen -- both are required config keys -- but fail
        # toward a topic that's at least identifiable rather than raising
        # and losing the result entirely.
        prefix = f"alpr_result/{direction or 'unknown'}"
    return f"{prefix}/{bay}"


def build_reply(config: dict, bay: str, direction: str, truck_number,
                confidence: float, status: str, event_time, ocr_time=None,
                **extra) -> dict:
    """The lean payload a downstream gate/dock system consumes.

    truck_number is never null: anything short of a valid read reports
    alpr.unknown_plate_value, and "status" is what tells the two apart.
    """
    reply = {
        'bay': bay,
        'direction': direction,
        'truck_number': truck_number or config['alpr']['unknown_plate_value'],
        'confidence': confidence,
        'status': status,
        'event_time': event_time,
        'ocr_time': ocr_time or datetime.now(timezone.utc).isoformat(),
    }
    reply.update(extra)
    return reply


def should_publish(config: dict, status: str) -> bool:
    """A NO_VALID_PLATE result is publishable only if the operator asked
    for an update on every trigger regardless of read success. Camera and
    system errors always publish."""
    return status != 'NO_VALID_PLATE' or config['alpr']['publish_no_valid_plate']
