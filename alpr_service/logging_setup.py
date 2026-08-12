"""Configures the 'alpr' logger tree from config.json's "logging" section.

Call configure_logging() once at startup; use get_logger(stage) everywhere
else to get a stage-tagged logger (mirrors the old debug_log(stage, msg)
call style, but level-filtered and routable to a file).

Level is selectable purely via config ("logging.level": "DEBUG" / "INFO" /
"WARNING" / "ERROR") -- no code change needed to get verbose troubleshooting
output out of a worker in the field.
"""
import logging
import logging.handlers
import time
from pathlib import Path

_FORMAT = "%(asctime)s.%(msecs)03dZ [%(levelname)-7s] [%(stage)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_ROOT_NAME = "alpr"


def configure_logging(log_cfg: dict) -> logging.Logger:
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    formatter.converter = time.gmtime  # UTC, matching result.json's ocr_time

    if log_cfg.get("console", True):
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    log_file = log_cfg.get("file")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(path),
            maxBytes=log_cfg.get("max_bytes", 10 * 1024 * 1024),
            backupCount=log_cfg.get("backup_count", 5))
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # waitress (the http_trigger webhook server, when enabled) logs under
    # its own logger name with its own default formatting/destination --
    # without this it either goes nowhere (no handler attached) or prints
    # unformatted straight to stderr, inconsistent with everything else.
    # Reusing the same handlers/level keeps "server started on ..." and
    # any waitress-side errors in the same log stream as the rest of the
    # service, tagged with waitress's own logger name via %(name)s instead
    # of a "stage" (waitress doesn't go through get_logger/_StageAdapter).
    waitress_log = logging.getLogger("waitress")
    waitress_log.handlers.clear()
    waitress_log.setLevel(level)
    waitress_log.propagate = False
    waitress_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt=_DATEFMT)
    waitress_formatter.converter = time.gmtime
    # Fresh handler instances rather than reusing root's -- each handler
    # owns its own stream/lock, so sharing instances across loggers works
    # in CPython but is fragile; building console + optional rotating file
    # again here (same config, different formatter) is simple and correct.
    if log_cfg.get("console", True):
        wh = logging.StreamHandler()
        wh.setFormatter(waitress_formatter)
        waitress_log.addHandler(wh)
    if log_file:
        wfh = logging.handlers.RotatingFileHandler(
            str(Path(log_file)),
            maxBytes=log_cfg.get("max_bytes", 10 * 1024 * 1024),
            backupCount=log_cfg.get("backup_count", 5))
        wfh.setFormatter(waitress_formatter)
        waitress_log.addHandler(wfh)

    return root


class _StageAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})["stage"] = self.extra["stage"]
        return msg, kwargs


def get_logger(stage: str) -> logging.LoggerAdapter:
    """Stage-tagged logger, e.g. get_logger("RTSP").info("...")."""
    return _StageAdapter(logging.getLogger(_ROOT_NAME), {"stage": stage})
