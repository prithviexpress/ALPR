"""Audit folder retention: prune run folders older than N days at startup."""
import shutil
import time
from pathlib import Path

from .logging_setup import get_logger

log = get_logger("SETUP")


def prune_audit(audit_dir: Path, days: int):
    if not audit_dir.exists() or days <= 0:
        return
    cutoff = time.time() - days * 86400
    removed = 0
    for cam_dir in audit_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        for run_dir in cam_dir.iterdir():
            try:
                if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(run_dir)
                    removed += 1
            except Exception as e:
                log.warning(f"prune failed for {run_dir}: {e}")
    if removed:
        log.info(f"pruned {removed} audit folders older than {days}d")
