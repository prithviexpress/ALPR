"""Crop preparation for OCR, quality heuristics, and frame-size QA."""
import cv2
import numpy as np

# Sizing/threshold values are required arguments, deliberately not
# defaulted here: config.py's load_config() is the single home for every
# tunable's default, and a second set of literals in this file would be a
# place for them to silently diverge from it. Callers pass
# alpr.ocr_prep_target_height / ocr_prep_padding / duplicate_resize_width
# / duplicate_resize_height / duplicate_diff_threshold.


def prep(img, target_h: int, pad: int):
    scale = target_h / img.shape[0]
    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    edge = np.concatenate([img[0].reshape(-1, 3), img[-1].reshape(-1, 3),
                            img[:, 0].reshape(-1, 3), img[:, -1].reshape(-1, 3)])
    fill = tuple(int(v) for v in np.median(edge, axis=0))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                               borderType=cv2.BORDER_CONSTANT, value=fill)


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var()


def thumbnail(img, resize):
    """The normalized form near-duplicate comparison works on. Computed
    once per candidate when it's kept, then reused for every later
    comparison -- collect() checks each new crop against every crop it
    already kept, so resizing the stored side per comparison meant the
    same crop was resized once per subsequent candidate instead of once
    ever (132 resizes instead of 24 to fill a 12-candidate window)."""
    try:
        return cv2.resize(img, resize)
    except Exception:
        return None


def duplicate_thumbs(a, b, diff_threshold) -> bool:
    """Near-duplicate test on two already-thumbnailed crops."""
    if a is None or b is None:
        return False
    try:
        return cv2.absdiff(a, b).mean() < diff_threshold
    except Exception:
        return False


def duplicate(a, b, resize, diff_threshold) -> bool:
    """Convenience wrapper for callers holding two raw crops."""
    return duplicate_thumbs(thumbnail(a, resize), thumbnail(b, resize),
                             diff_threshold)


def save_debug_image(folder, name, image, logger=None):
    if image is None:
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / name), image)
    except Exception as e:
        if logger:
            logger.warning(f"failed to save {name}: {e}")


def check_frame_size(frame, expected_width, expected_height, tolerance_pct):
    """Validate a captured frame's resolution against the expected sensor
    size (e.g. a 5MP camera -> 2592x1944).

    Returns (ok, actual_w, actual_h). If expected_width/height are not
    configured, the check is disabled and this always reports ok=True.

    This exists to catch the snapshot endpoint silently serving a lower
    resolution than the camera's real sensor -- before wasting a whole
    collection cycle running YOLO and OCR against frames that can never
    hold a readable plate.
    """
    if not expected_width or not expected_height:
        return True, frame.shape[1], frame.shape[0]
    actual_h, actual_w = frame.shape[:2]
    expected_mp = (expected_width * expected_height) / 1_000_000
    actual_mp = (actual_w * actual_h) / 1_000_000
    tolerance = expected_mp * (tolerance_pct / 100)
    ok = abs(actual_mp - expected_mp) <= tolerance
    return ok, actual_w, actual_h
