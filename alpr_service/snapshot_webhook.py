"""On-demand bay query API -- serves bay state, snapshots, and one-off
vision-model questions over HTTP, GET on request, instead of the service
pushing any of it out on a timer. Pull, not push: nothing is sent
anywhere until it's actually asked for.

Routes:
    GET  /bays              every bay's state in one call (dashboards)
    GET  /bay/<bay>         one bay: occupancy, activity, door, truck
    GET  /snapshot/<bay>    that bay's most recent frame, base64 JPEG
    POST /bay/<bay>/ask     put a question to the vision model about
                            this bay's current frame
    GET  /healthz           liveness

Deliberately a separate server/config from http_trigger.py -- that one
RECEIVES camera alert triggers (a different concern entirely), and a
deployment may well want this query API without also running the ALPR
trigger webhook, or vice versa.

Every route is read-only with respect to service state: querying a bay
never triggers a camera fetch, a classification, or an MQTT publish. The
one exception is /ask, which by its nature spends a vision-model call --
which is exactly why it's a POST and on-demand rather than something the
service does on a schedule.
"""
from flask import Flask, jsonify, request

from .logging_setup import get_logger


def build_snapshot_webhook_app(get_snapshot_fn, get_state_fn=None,
                                list_states_fn=None, ask_fn=None) -> Flask:
    """Each handler is injected as a plain callable rather than importing
    BayMonitor/BayStateEngine here, so this module (and its Flask
    dependency) only gets pulled in when the webhook is actually turned
    on, and so the routes can be tested without constructing either.

    get_state_fn/list_states_fn/ask_fn are optional: a deployment running
    bay_monitor without the state engine still gets /snapshot and the
    occupancy half of /bay, and a route whose handler wasn't wired up
    answers 501 rather than 404 -- "this service doesn't offer that"
    is a different problem from "no such bay", and confusing the two
    sends someone hunting for a typo in a bay name that was fine."""
    log = get_logger("SNAPSHOT_WEBHOOK")
    app = Flask(__name__)

    @app.errorhandler(Exception)
    def _on_error(exc):
        log.error(f"unhandled error in snapshot webhook request: {exc}",
                  exc_info=True)
        return jsonify({"status": "error", "reason": "internal_error"}), 500

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify({"status": "ok"}), 200

    @app.route("/bays", methods=["GET"])
    def bays():
        if list_states_fn is None:
            return jsonify({"status": "not_available",
                            "reason": "bay state queries are not enabled"}), 501
        states = list_states_fn()
        return jsonify({"count": len(states), "bays": states}), 200

    @app.route("/bay/<bay>", methods=["GET"])
    def bay_state(bay):
        if get_state_fn is None:
            return jsonify({"status": "not_available",
                            "reason": "bay state queries are not enabled"}), 501
        result = get_state_fn(bay)
        if result is None:
            return jsonify({"status": "not_found", "bay": bay}), 404
        return jsonify(result), 200

    @app.route("/snapshot/<bay>", methods=["GET"])
    def snapshot(bay):
        result = get_snapshot_fn(bay)
        if result is None:
            return jsonify({"status": "not_found", "bay": bay}), 404
        return jsonify(result), 200

    @app.route("/bay/<bay>/ask", methods=["POST"])
    def ask(bay):
        """An open question about what's happening at a bay, answered by
        the vision model against that bay's current frame -- the thing a
        fixed detector vocabulary can't do ("is it being loaded or
        unloaded", "is anything blocking the bay").

        Costs a real model call, so it's a POST and never happens on its
        own. silent=True on get_json: a caller sending no body, or a bad
        content-type, should get the 400 below explaining what's needed,
        not Flask's own 415/400 with no hint about the "question" field."""
        if ask_fn is None:
            return jsonify({"status": "not_available",
                            "reason": "vision-model queries are not enabled"}), 501
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or body.get("q") or "").strip()
        if not question:
            return jsonify({
                "status": "bad_request",
                "reason": 'send a JSON body like {"question": "..."}',
            }), 400
        result = ask_fn(bay, question)
        if result is None:
            return jsonify({"status": "not_found", "bay": bay}), 404
        return jsonify(result), 200

    return app
