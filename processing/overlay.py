"""
Overlay preparation.

The user supplied image (PNG / JPG / WebP - transparent PNG preferred) is
validated, scaled without distortion (aspect ratio preserved, alpha kept) and
positioned on the output canvas.

Two application styles exist:

* positioned (default) - the overlay keeps its aspect ratio, is scaled to a
  size preset and placed in one of the five positions with a margin;
* full screen (``OVERLAY_FULL_SCREEN_SIZE``) - the image covers the whole
  canvas edge-to-edge (aspect-preserving cover + centre crop).  Frame /
  watermark style overlays are usually opaque with a black middle, so when
  the source image carries no alpha channel its near-black pixels are keyed
  to transparent and the video shows through; images that already have an
  alpha channel are composited as-is.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings
from processing.errors import OverlayError

log = logging.getLogger(__name__)

# Black-keying band for full-screen overlays (luminance 0-255): at or below
# _KEY_LOW a pixel is fully transparent, at or above _KEY_HIGH fully opaque,
# with a linear ramp between so anti-aliased edges stay smooth.
_KEY_LOW = 16.0
_KEY_HIGH = 48.0


def validate_overlay(path: Path) -> None:
    """Raise :class:`OverlayError` when the file is not a usable image."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise OverlayError(f"Overlay file missing or empty: {path}")
    if path.suffix.lower() not in settings.OVERLAY_SUPPORTED_EXTENSIONS:
        raise OverlayError(
            f"Unsupported overlay format '{path.suffix}'. "
            f"Supported: {', '.join(settings.OVERLAY_SUPPORTED_EXTENSIONS)}"
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width < 2 or image.height < 2:
                raise OverlayError("Overlay image is too small.")
    except OverlayError:
        raise
    except Exception as exc:  # noqa: BLE001 - any Pillow failure means unusable file
        raise OverlayError(f"Overlay image cannot be read: {exc}") from exc


def normalize_position(position: str | None) -> str:
    position = (position or settings.OVERLAY_DEFAULT_POSITION).lower()
    if position not in settings.OVERLAY_POSITIONS:
        log.warning("Unknown overlay position %r - falling back to center", position)
        return "center"
    return position


def normalize_size_label(size_label: str | None) -> str:
    label = (size_label or settings.OVERLAY_DEFAULT_SIZE).lower()
    if label not in settings.OVERLAY_SIZE_PRESETS:
        log.warning("Unknown overlay size %r - falling back to original", label)
        return "original"
    return label


def prepare_overlay(
    overlay_path: Path,
    out_png: Path,
    canvas_width: int,
    canvas_height: int,
    *,
    position: str,
    size_label: str,
) -> tuple[Path, tuple[int, int]]:
    """
    Scale + position the overlay and save it as a transparent PNG.

    Returns ``(path, (x, y))`` where x/y is the top-left corner on the canvas.
    """
    validate_overlay(overlay_path)
    position = normalize_position(position)
    size_label = normalize_size_label(size_label)

    if size_label == settings.OVERLAY_FULL_SCREEN_SIZE:
        return _prepare_full_screen(overlay_path, out_png, canvas_width, canvas_height)

    margin = max(0, settings.OVERLAY_MARGIN)

    try:
        with Image.open(overlay_path) as image:
            overlay = image.convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        raise OverlayError(f"Overlay image cannot be processed: {exc}") from exc

    max_w = max(2, canvas_width - 2 * margin)
    max_h = max(2, canvas_height - 2 * margin)

    preset = settings.OVERLAY_SIZE_PRESETS[size_label]
    if preset is None:
        target_w = overlay.width
    else:
        target_w = max(2, int(round(canvas_width * preset)))

    scale = target_w / overlay.width
    # never overflow the canvas
    scale = min(scale, max_w / overlay.width, max_h / overlay.height)
    target_w = max(2, int(round(overlay.width * scale)))
    target_h = max(2, int(round(overlay.height * scale)))

    if (target_w, target_h) != overlay.size:
        overlay = overlay.resize((target_w, target_h), Image.LANCZOS)

    x, y = position_xy(position, canvas_width, canvas_height, target_w, target_h, margin)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_png, format="PNG", optimize=True)

    log.info(
        "overlay ready: %s %dx%d at (%d,%d) [%s, %s] -> %s",
        overlay_path.name,
        target_w,
        target_h,
        x,
        y,
        position,
        size_label,
        out_png.name,
    )
    return out_png, (x, y)


def position_xy(
    position: str,
    canvas_width: int,
    canvas_height: int,
    overlay_width: int,
    overlay_height: int,
    margin: int,
) -> tuple[int, int]:
    """Top-left overlay corner for the supported positions."""
    right = max(0, canvas_width - overlay_width - margin)
    bottom = max(0, canvas_height - overlay_height - margin)
    center_x = max(0, (canvas_width - overlay_width) // 2)
    center_y = max(0, (canvas_height - overlay_height) // 2)
    positions = {
        "top-left": (margin, margin),
        "top-right": (right, margin),
        "center": (center_x, center_y),
        "bottom-left": (margin, bottom),
        "bottom-right": (right, bottom),
    }
    x, y = positions.get(position, positions["center"])
    return max(0, min(x, canvas_width - 2)), max(0, min(y, canvas_height - 2))


def _prepare_full_screen(
    overlay_path: Path,
    out_png: Path,
    canvas_width: int,
    canvas_height: int,
) -> tuple[Path, tuple[int, int]]:
    """Scale the overlay to cover the whole canvas and key black to transparent.

    Returns ``(path, (0, 0))`` - a full-screen overlay always sits at the
    origin; the margin and position presets do not apply to it.
    """
    try:
        with Image.open(overlay_path) as image:
            has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
            overlay = image.convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        raise OverlayError(f"Overlay image cannot be processed: {exc}") from exc

    # cover: scale until both canvas dimensions are filled, then centre-crop
    scale = max(canvas_width / overlay.width, canvas_height / overlay.height)
    new_w = max(canvas_width, int(round(overlay.width * scale)))
    new_h = max(canvas_height, int(round(overlay.height * scale)))
    if (new_w, new_h) != overlay.size:
        overlay = overlay.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - canvas_width) // 2
    top = (new_h - canvas_height) // 2
    overlay = overlay.crop((left, top, left + canvas_width, top + canvas_height))

    if not has_alpha:
        overlay = _key_black_to_transparent(overlay)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_png, format="PNG", optimize=True)

    log.info(
        "overlay ready: full screen %dx%d (source alpha: %s) -> %s",
        canvas_width,
        canvas_height,
        has_alpha,
        out_png.name,
    )
    return out_png, (0, 0)


def _key_black_to_transparent(overlay: Image.Image) -> Image.Image:
    """Turn the near-black pixels of an opaque overlay transparent.

    Frame/watermark overlays use black as the see-through colour; the soft
    band keeps anti-aliased edges smooth instead of leaving a dark halo.
    """
    pixels = np.asarray(overlay, dtype=np.float32)
    luminance = (
        pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114
    )
    alpha = np.clip(
        (luminance - _KEY_LOW) / (_KEY_HIGH - _KEY_LOW), 0.0, 1.0
    ) * 255.0
    keyed = pixels.copy()
    keyed[..., 3] = alpha
    return Image.fromarray(keyed.astype(np.uint8), "RGBA")
