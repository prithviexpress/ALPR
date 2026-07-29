"""cameras.json loading and validation (the per-bay registry)."""
import json
from pathlib import Path

TEMPLATE = {
    "AR-FS": {
        "guid": "01000000001babe00c81f7ff9",
        "ip": "10.69.10.100",
        "roi": [850, 250, 1750, 1800],
        "enabled": True,
        "rtsp_mode": "genetec"
    }
}


class CamerasError(RuntimeError):
    pass


def load_cameras(path: Path) -> dict:
    if not path.exists():
        path.write_text(json.dumps(TEMPLATE, indent=2))
        raise CamerasError(f"Wrote template {path} - fill it in and rerun")

    cameras = json.loads(path.read_text())
    for bay, cam in cameras.items():
        roi = cam.get("roi")
        if not isinstance(roi, list) or len(roi) != 4:
            raise CamerasError(
                f"camera '{bay}' has an invalid/missing 'roi' "
                f"(expected [x1, y1, x2, y2])")
        mode = cam.get("rtsp_mode", "genetec")
        if mode not in ("genetec", "direct"):
            raise CamerasError(
                f"camera '{bay}' has unknown rtsp_mode '{mode}' "
                f"(expected 'genetec' or 'direct')")
        if mode == "direct" and not cam.get("ip"):
            raise CamerasError(
                f"camera '{bay}' uses rtsp_mode 'direct' but has no 'ip'")
    return cameras
