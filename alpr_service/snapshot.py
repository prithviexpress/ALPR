"""HTTP snapshot capture: pull one full-resolution JPEG per request from a
camera's snapshot endpoint (e.g. a Bosch dome's /snap.jpg) using HTTP
Digest auth, instead of decoding a continuous RTSP video stream.

This is the sole capture path -- Genetec/RTSP support was removed
entirely in favor of this, since it's simpler (no video decode, no
stream-open/read timeout tuning) and, for cameras that are stationary or
slow-moving when triggered, doesn't need RTSP's higher frame rate to
catch a good shot.

One common username/password is used for every camera (config.json
"snapshot.username"/"password") -- there is no per-camera override.
"""
import time

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth


class SnapshotError(RuntimeError):
    pass


def build_snapshot_url(camera_cfg: dict, config: dict) -> str:
    ip = camera_cfg.get("ip")
    if not ip:
        raise SnapshotError("snapshot capture requires camera 'ip'")
    snap_cfg = config["snapshot"]
    template = snap_cfg.get("url_template", "http://{ip}:{port}/snap.jpg")
    port = snap_cfg.get("port", 80)
    return template.format(ip=ip, port=port)


def build_auth(config: dict) -> HTTPDigestAuth:
    snap_cfg = config["snapshot"]
    username = snap_cfg.get("username")
    password = snap_cfg.get("password")
    if username is None or password is None:
        raise SnapshotError(
            "snapshot capture requires config.json snapshot.username and "
            "snapshot.password (one common set of credentials for every "
            "camera)")
    return HTTPDigestAuth(username, password)


def fetch_snapshot(session: requests.Session, url: str, auth: HTTPDigestAuth,
                    connect_timeout_ms: int, read_timeout_ms: int):
    """Fetch and decode one JPEG snapshot.

    Returns (frame, elapsed_ms, size_bytes). Raises SnapshotError on any
    failure -- non-200 status, connection/read timeout, or an undecodable
    response -- so callers handle all of those the same way (count it as
    a failed sample and keep polling within the collection window).

    Reusing one Session/HTTPDigestAuth across many calls (see Worker,
    which creates one of each per worker thread) matters here: without
    it, every single request pays for a fresh TCP handshake plus the
    full digest challenge-response round trip; with it, only the first
    request per host does.
    """
    t0 = time.time()
    try:
        resp = session.get(
            url, auth=auth,
            timeout=(connect_timeout_ms / 1000, read_timeout_ms / 1000))
    except requests.RequestException as e:
        raise SnapshotError(f"request failed: {e}") from e
    elapsed_ms = round((time.time() - t0) * 1000)

    if resp.status_code != 200:
        raise SnapshotError(f"HTTP {resp.status_code}")

    data = np.frombuffer(resp.content, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise SnapshotError("could not decode JPEG response")
    return frame, elapsed_ms, len(resp.content)
