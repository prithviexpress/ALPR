"""On-demand snapshot webhook -- serves bay_monitor's most recently
fetched frame (plus its last-known occupancy/activity) over HTTP, GET on
request, instead of bay_monitor pushing images out on a timer. Pull, not
push: nothing is sent anywhere until this is actually asked for.

Deliberately a separate server/config from http_trigger.py -- that one
RECEIVES camera alert triggers (a different concern entirely), and a
deployment may well want snapshot-on-demand without also running the
ALPR trigger webhook, or vice versa.
"""
from flask import Flask, jsonify

from .logging_setup import get_logger


def build_snapshot_webhook_app(get_snapshot_fn) -> Flask:
    """get_snapshot_fn(bay) -> dict | None, e.g. BayMonitor.get_snapshot
    -- kept as a plain callable rather than importing BayMonitor here so
    this module (and its Flask/waitress dependency) only gets pulled in
    when the webhook is actually turned on."""
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

    @app.route("/snapshot/<bay>", methods=["GET"])
    def snapshot(bay):
        result = get_snapshot_fn(bay)
        if result is None:
            return jsonify({"status": "not_found", "bay": bay}), 404
        return jsonify(result), 200

    return app
