"""
Face-centred cropping for the pair composite.

Every extracted photo fills exactly one half of the 16:9 canvas.  The crop
window for that half is chosen so that

* the primary (largest) face sits as close as possible to the centre of the
  half,
* the face - plus sensible head-room and shoulders - is never cut off,
* the window keeps the half's aspect exactly, so nothing is ever stretched,

and it is a *single*, fixed window: the photo stays perfectly static for its
full 8 seconds (no zoom, no pan, no animation of any kind).

Detection uses the Haar cascades bundled with OpenCV: frontal first (two
strictness passes), then profile and mirrored profile, so faces looking to
either side are found as well.  The largest detection wins (never a random
pick).  When no face is found at all the window is centred on the strongest
image structure - a cheap "main subject" heuristic (gradient-energy
centroid of the top quartile) - and a completely flat photo just gets a
plain centre crop.  Detection problems never fail the job.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:  # OpenCV is a real dependency, but a broken install must not kill jobs
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only on broken installs
    cv2 = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# tuning constants
# --------------------------------------------------------------------------- #
_DETECT_MAX_DIM = 720            # detection runs on a downscaled copy (speed)
_FRONTAL_PASSES = (6, 4)         # minNeighbors: strict first, then looser
_PROFILE_NEIGHBORS = 5
_MIN_FACE_FRACTION = 0.05        # smallest face (fraction of the short side)
_BOX_ASPECT_RANGE = (0.55, 1.80)  # sane face boxes only (w/h)

_PAD_TOP = 0.65                  # head-room above the face box (x face height)
_PAD_BOTTOM = 0.30               # chin / shoulders below the box
_PAD_SIDE = 0.35                 # cheeks left and right of the box
_FACE_HEIGHT_FRACTION = 0.42     # face height / window height (framing)
_MAX_UPSCALE = 2.0               # window >= half/(this) -> resize <= this

_SUBJECT_SIZE = 96               # gradient-centroid grid for the no-face case
_ENERGY_EPS = 1e-6

_cascade_local = threading.local()


@dataclass(frozen=True)
class FaceBox:
    """A detected face in photo pixel coordinates (top-left origin)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def face_centered_crop(photo: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Crop ``photo`` to ``out_w`` x ``out_h`` around its main face (or subject).

    The returned image has exactly the requested size; the crop keeps the
    photo's natural proportions (the window has the output aspect ratio), so
    the picture is never distorted.
    """
    photo_w, photo_h = photo.size
    aspect = out_w / max(1, out_h)

    face = detect_primary_face(photo)
    if face is not None:
        box = _padded_box(face, photo_w, photo_h)
        max_h = min(float(photo_h), photo_w / aspect)
        want_h = face.h / _FACE_HEIGHT_FRACTION
        min_h = min(out_h / _MAX_UPSCALE, max_h)
        win_h = max(min_h, min(want_h, max_h))
        cx, cy = face.center
        # When the ideal window cannot be shifted far enough to centre the
        # face (face close to a photo edge), shrink it just enough to make
        # exact centring possible - but never below the framing / upscale
        # floors, never below what the face itself needs, and only when the
        # shrink actually achieves the exact centring.
        centerable = min(
            max_h,
            2.0 * min(cy, photo_h - cy),
            2.0 * min(cx, photo_w - cx) / aspect,
        )
        floor_h = max(min_h, face.h, face.w / aspect)
        if win_h > centerable >= floor_h:
            win_h = centerable
        win_w = win_h * aspect
        pref_x, pref_y = cx, cy
        x0, y0, x1, y1 = _window(pref_x, pref_y, win_w, win_h, photo_w, photo_h, box)
    else:
        subject = _subject_point(photo)
        max_h = min(float(photo_h), photo_w / aspect)
        win_w, win_h = max_h * aspect, max_h
        pref_x, pref_y = subject if subject is not None else (photo_w / 2.0, photo_h / 2.0)
        x0, y0, x1, y1 = _window(pref_x, pref_y, win_w, win_h, photo_w, photo_h, None)

    crop = photo.crop((x0, y0, x1, y1))  # (x1, y1) are exclusive
    if crop.size != (out_w, out_h):
        crop = crop.resize((out_w, out_h), Image.LANCZOS)
    return crop


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def detect_primary_face(photo: Image.Image) -> FaceBox | None:
    """Return the largest detected face, or ``None`` when no face is found.

    Frontal cascade with two strictness passes, then the profile cascade on
    the image and on its mirror image (faces looking left/right).  Any
    OpenCV problem degrades to ``None`` - the caller then uses the
    subject/centre fallback and the job continues.
    """
    if cv2 is None:
        return None
    try:
        frontal, profile = _cascades()
        if frontal is None and profile is None:
            return None

        gray = photo.convert("L")
        scale = min(1.0, _DETECT_MAX_DIM / max(photo.size))
        if scale < 1.0:
            gray = gray.resize(
                (max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
                Image.BILINEAR,
            )
        arr = np.asarray(gray, dtype=np.uint8)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        arr = cv2.equalizeHist(arr)

        min_side = max(24, int(round(_MIN_FACE_FRACTION * min(arr.shape))))
        min_size = (min_side, min_side)

        boxes: list[tuple[int, int, int, int]] = []
        if frontal is not None:
            for neighbors in _FRONTAL_PASSES:
                found = frontal.detectMultiScale(
                    arr, scaleFactor=1.1, minNeighbors=neighbors, minSize=min_size
                )
                if len(found):
                    boxes = [tuple(int(v) for v in box) for box in found]
                    break
        if not boxes and profile is not None:
            found = profile.detectMultiScale(
                arr, scaleFactor=1.1, minNeighbors=_PROFILE_NEIGHBORS, minSize=min_size
            )
            boxes = [tuple(int(v) for v in box) for box in found]
            mirrored = profile.detectMultiScale(
                cv2.flip(arr, 1),
                scaleFactor=1.1,
                minNeighbors=_PROFILE_NEIGHBORS,
                minSize=min_size,
            )
            width = arr.shape[1]
            boxes += [(width - x - w, y, w, h) for (x, y, w, h) in mirrored]

        boxes = [b for b in boxes if _sane_box(b)]
        if not boxes:
            return None

        # largest area wins; ties resolved deterministically (leftmost, topmost)
        x, y, w, h = max(boxes, key=lambda b: (b[2] * b[3], -b[0], -b[1]))
        inv = 1.0 / scale
        return FaceBox(x * inv, y * inv, w * inv, h * inv)
    except Exception as exc:  # noqa: BLE001 - detection must never fail the job
        log.warning("face detection failed (%s) - using the subject fallback", exc)
        return None


def _cascades():  # type: ignore[no-untyped-def]
    """Per-thread cascade classifiers (OpenCV cascades are not reentrant)."""
    cached = getattr(_cascade_local, "value", None)
    if cached is not None:
        return cached
    base = cv2.data.haarcascades  # type: ignore[union-attr]
    frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
    result = (
        frontal if not frontal.empty() else None,
        profile if not profile.empty() else None,
    )
    if result == (None, None):  # pragma: no cover - broken OpenCV install
        log.warning("OpenCV cascade files are missing - face detection is off")
    _cascade_local.value = result
    return result


def _sane_box(box: tuple[int, int, int, int]) -> bool:
    _x, _y, w, h = box
    if w <= 0 or h <= 0:
        return False
    ratio = w / h
    return _BOX_ASPECT_RANGE[0] <= ratio <= _BOX_ASPECT_RANGE[1]


# --------------------------------------------------------------------------- #
# window geometry (pure functions - unit-testable without OpenCV)
# --------------------------------------------------------------------------- #
def _padded_box(face: FaceBox, photo_w: int, photo_h: int) -> tuple[float, float, float, float]:
    """Head box: the face plus head-room / shoulders that must stay visible."""
    left = face.x - _PAD_SIDE * face.w
    right = face.x + face.w + _PAD_SIDE * face.w
    top = face.y - _PAD_TOP * face.h
    bottom = face.y + face.h + _PAD_BOTTOM * face.h
    # boxes may reach beyond the photo (face at the edge) - that is fine,
    # the window itself is kept inside the photo below
    return left, top, right, bottom


def _axis_center(
    preferred: float,
    box_lo: float,
    box_hi: float,
    win: float,
    total: float,
) -> float:
    """Window-centre coordinate on one axis.

    Prefers ``preferred`` (the face centre), slides the window as far as
    needed so the padded box stays inside it, and always keeps the window
    itself inside the photo.  When the box cannot fit (e.g. a face larger
    than the half can hold) the position closest to the ideal one wins.
    """
    half = win / 2.0
    lo_limit = half
    hi_limit = total - half
    if hi_limit < lo_limit:  # window bigger than the photo - centre it
        return total / 2.0

    if box_hi - box_lo <= win:
        lo = max(lo_limit, box_hi - half)
        hi = min(hi_limit, box_lo + half)
        if lo <= hi:
            return min(max(preferred, lo), hi)
    return min(max(preferred, lo_limit), hi_limit)


def _window(
    pref_x: float,
    pref_y: float,
    win_w: float,
    win_h: float,
    photo_w: int,
    photo_h: int,
    box: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int]:
    """Integer crop window (left, upper, right, lower), right/lower exclusive."""
    if box is not None:
        box_l, box_t, box_r, box_b = box
        cx = _axis_center(pref_x, box_l, box_r, win_w, photo_w)
        cy = _axis_center(pref_y, box_t, box_b, win_h, photo_h)
    else:
        cx = _axis_center(pref_x, pref_x, pref_x, win_w, photo_w)
        cy = _axis_center(pref_y, pref_y, pref_y, win_h, photo_h)

    w = max(2, min(int(round(win_w)), photo_w))
    h = max(2, min(int(round(win_h)), photo_h))
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x0 = max(0, min(x0, photo_w - w))
    y0 = max(0, min(y0, photo_h - h))
    return x0, y0, x0 + w, y0 + h


def _subject_point(photo: Image.Image) -> tuple[float, float] | None:
    """Gradient-energy centroid of the top quartile - the "main subject".

    Structured, high-contrast areas (faces, people, objects) carry far more
    gradient energy than smooth sky / ground / walls, so their centre of
    mass is a decent, deterministic stand-in for "the main visual subject".
    Returns ``None`` for essentially flat photos.
    """
    small = photo.convert("L").resize((_SUBJECT_SIZE, _SUBJECT_SIZE), Image.BILINEAR)
    gray = np.asarray(small, dtype=np.float32)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    if float(magnitude.max()) < _ENERGY_EPS:
        return None

    cut = float(np.percentile(magnitude, 75.0))
    weights = np.where(magnitude >= cut, magnitude, 0.0)
    total = float(weights.sum())
    if total < _ENERGY_EPS:
        return None

    index = np.arange(_SUBJECT_SIZE, dtype=np.float32)
    cx = float((weights.sum(axis=0) * index).sum() / total)
    cy = float((weights.sum(axis=1) * index).sum() / total)
    photo_w, photo_h = photo.size
    return (
        (cx + 0.5) * photo_w / _SUBJECT_SIZE,
        (cy + 0.5) * photo_h / _SUBJECT_SIZE,
    )
