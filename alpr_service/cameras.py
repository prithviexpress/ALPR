"""cameras.json loading and validation (the per-bay registry).

Each entry is just {ip, roi, enabled} -- every camera is reached via its
HTTP snapshot endpoint (config.json "snapshot"), using one common
username/password for all cameras, so there's nothing camera-specific to
store here beyond where it is and what to crop. A leftover "guid" key
from an earlier Genetec-based setup is harmless and ignored.
"""
import json
from pathlib import Path

TEMPLATE = {
    "AR-FS": {
        "ip": "10.69.10.100",
        "roi": [850, 250, 1750, 1800],
        "enabled": True
    }
}


class CamerasError(RuntimeError):
    pass


def load_cameras(path: Path) -> dict:
    if not path.exists():
        path.write_text(json.dumps(TEMPLATE, indent=2))
        raise CamerasError(f"Wrote template {path} - fill it in and rerun")

    raw = json.loads(path.read_text())
    # Keys starting with "_" are treated as comments/metadata, not bays
    # (e.g. a top-level "_comment" documenting the file), and skipped.
    cameras = {bay: cam for bay, cam in raw.items() if not bay.startswith("_")}
    for bay, cam in cameras.items():
        roi = cam.get("roi")
        if not isinstance(roi, list) or len(roi) != 4:
            raise CamerasError(
                f"camera '{bay}' has an invalid/missing 'roi' "
                f"(expected [x1, y1, x2, y2])")
    return cameras
