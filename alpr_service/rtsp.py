"""RTSP URL construction and stream capture, supporting two source modes
(config "rtsp.mode", overridable per-camera via cameras.json "rtsp_mode"):

  "genetec" -- pull the feed through a Genetec media-gateway endpoint keyed
               by camera GUID (server/port/username/password in config.json
               "rtsp", same as R1/R2's build_rtsp()).
  "direct"  -- connect straight to the camera's own RTSP server using its
               IP and credentials (config.json "rtsp.direct", or per-camera
               overrides in cameras.json), bypassing Genetec entirely.

Both modes always request the camera's primary/main stream (stream 1, full
resolution) -- OCR needs the detail a substream throws away:

  - "direct" mode builds the URL from a template with a {stream} placeholder
    that defaults to 1 (see config.json "rtsp.direct.url_template"; the
    stock template is Hikvision-style "Channels/101" = channel 1, stream 1
    -- adjust the template for other camera vendors).
  - "genetec" mode's gateway endpoint (".../<guid>/live") already serves a
    single, pre-selected stream per GUID in the deployments this was built
    against. If your Genetec gateway exposes multiple streams behind that
    GUID, add whatever selector it needs to config.json "rtsp.path_suffix"
    (appended verbatim to the URL).
"""
import os
import threading
import time
from urllib.parse import quote

import cv2

_open_lock = threading.Lock()


class RtspOpenError(RuntimeError):
    pass


def _stream_number(camera_cfg, rtsp_cfg):
    return int(camera_cfg.get("stream") or rtsp_cfg.get("stream", 1))


def _build_genetec_url(camera_cfg, rtsp_cfg):
    for key in ("server", "port", "username", "password"):
        if key not in rtsp_cfg:
            raise RtspOpenError(
                f"rtsp mode 'genetec' requires config.json rtsp.{key}")
    if "guid" not in camera_cfg:
        raise RtspOpenError("rtsp mode 'genetec' requires camera 'guid'")
    user = quote(str(rtsp_cfg["username"]), safe='')
    pw = quote(str(rtsp_cfg["password"]), safe='')
    suffix = rtsp_cfg.get("path_suffix", "")
    return (
        f"rtsp://{user}:{pw}@{rtsp_cfg['server']}:{rtsp_cfg['port']}/"
        f"{camera_cfg['guid']}/live{suffix}"
    )


def _build_direct_url(camera_cfg, rtsp_cfg):
    direct = rtsp_cfg.get("direct", {})
    ip = camera_cfg.get("ip")
    if not ip:
        raise RtspOpenError("rtsp mode 'direct' requires camera 'ip'")
    username = camera_cfg.get("username", direct.get("username"))
    password = camera_cfg.get("password", direct.get("password"))
    if username is None or password is None:
        raise RtspOpenError(
            "rtsp mode 'direct' requires 'username'/'password' on the "
            "camera entry or in config.json rtsp.direct")
    port = camera_cfg.get("port", direct.get("port", 554))
    channel = camera_cfg.get("channel", 1)
    stream = _stream_number(camera_cfg, rtsp_cfg)
    template = direct.get(
        "url_template",
        "rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}0{stream}")
    return template.format(
        username=quote(str(username), safe=''),
        password=quote(str(password), safe=''),
        ip=ip, port=port, channel=channel, stream=stream)


def build_rtsp_url(camera_cfg: dict, config: dict) -> str:
    rtsp_cfg = config["rtsp"]
    mode = camera_cfg.get("rtsp_mode", rtsp_cfg.get("mode", "genetec"))
    if mode == "direct":
        return _build_direct_url(camera_cfg, rtsp_cfg)
    if mode == "genetec":
        return _build_genetec_url(camera_cfg, rtsp_cfg)
    raise RtspOpenError(
        f"unknown rtsp mode '{mode}' (expected 'genetec' or 'direct')")


def redact(url: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://***:***@host/... for safe logging."""
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    _, host_and_path = rest.split("@", 1)
    return f"{scheme}://***:***@{host_and_path}"


def open_capture(url: str, timeout_ms: int, timeout_option_name: str = "stimeout"):
    """Open an RTSP stream over TCP with an explicit socket timeout.

    Without a timeout, an unreachable/black-holed camera can hang
    cv2.VideoCapture() (and later cap.read()) far longer than
    collection_timeout ever bounds, wedging a worker thread indefinitely.

    OpenCV's FFmpeg backend reads OPENCV_FFMPEG_CAPTURE_OPTIONS from the
    environment at VideoCapture() construction time (it's not cached), so
    it's set here, immediately before opening, under a lock -- it's
    process-wide state and multiple workers open streams concurrently.
    """
    options = f"rtsp_transport;tcp|{timeout_option_name};{timeout_ms * 1000}"
    with _open_lock:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
        t0 = time.time()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    open_ms = round((time.time() - t0) * 1000)
    return cap, open_ms
