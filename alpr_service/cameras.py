"""cameras.json loading and validation (the per-bay registry).

Each entry is just {guid, ip, roi, enabled} -- no RTSP mode or credentials
here. Whether guid (genetec) or ip (direct) is actually used, and what
credentials go with it, is decided globally by config.json "rtsp" (see
service.check_camera_rtsp_fields, which validates the active mode's
required field is present on every camera at startup).
"""
import json
from pathlib import Path

TEMPLATE = {
    "AR-FS": {
        "guid": "01000000001babe00c81f7ff9",
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
