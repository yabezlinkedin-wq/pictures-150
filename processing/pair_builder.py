"""
Pair composites: photo 1 + photo 2 -> pair 1 ... photo 299 + 300 -> pair 150.

Every pair becomes a single 16:9 still that the slideshow shows for exactly
8 seconds (150 pairs x 8 s = 20:00).  The left half holds the face-centred
crop of the first photo, the right half the face-centred crop of the second
one; both halves are visible simultaneously the whole time.  The composite
is a fixed image - nothing moves, zooms or fades.

Face detection and crop geometry live in :mod:`processing.face_center`; this
module only stitches the two halves onto the canvas and writes the JPEG that
the slideshow encode consumes (bare file names - ffmpeg runs with the frames
directory as its working directory).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from config import settings
from processing.errors import JobCancelledError, SlideshowError
from processing.face_center import face_centered_crop
from processing.ffmpeg_utils import ProgressCallback

log = logging.getLogger(__name__)


def build_pairs(
    photos: list[Path],
    out_width: int,
    out_height: int,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_event=None,  # threading.Event | None
) -> list[Path]:
    """Compose the photos into ``PHOTO_COUNT / 2`` static 16:9 stills.

    Returns the pair paths in order (``pair_001.jpg`` ... ``pair_150.jpg``),
    written next to the photos so the slideshow concat file can list them by
    bare name.  Each pair shows photo ``2k-1`` in the left half and photo
    ``2k`` in the right half, both face-centred, for the full 8 seconds.
    """
    photo_count = settings.PHOTO_COUNT
    if len(photos) != photo_count:
        raise SlideshowError(f"Expected {photo_count} photos, got {len(photos)}.")
    if out_width % 2 != 0:
        raise SlideshowError(f"Canvas width {out_width} must be even for the two halves.")
    if len(photos) % 2 != 0:  # pragma: no cover - PHOTO_COUNT is even by construction
        raise SlideshowError("Photos must form pairs.")

    pair_count = photo_count // 2
    frames_dir = photos[0].parent
    half_w = out_width // 2
    half_h = out_height

    def build_one(index: int) -> Path:
        pair_no = index + 1
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelledError("Pair building cancelled")
        try:
            with Image.open(photos[2 * index]) as handle:
                left = face_centered_crop(handle.convert("RGB"), half_w, half_h)
            with Image.open(photos[2 * index + 1]) as handle:
                right = face_centered_crop(handle.convert("RGB"), half_w, half_h)
            canvas = Image.new("RGB", (out_width, out_height))
            canvas.paste(left, (0, 0))
            canvas.paste(right, (half_w, 0))
            target = frames_dir / f"pair_{pair_no:03d}.jpg"
            # 4:4:4 + quality 95: the composite is one generation away from
            # the final picture, so it must not add visible JPEG artefacts.
            canvas.save(
                target,
                format="JPEG",
                quality=settings.PAIR_JPEG_QUALITY,
                subsampling=0,
            )
            return target
        except JobCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report which pair failed
            raise SlideshowError(f"Pair {pair_no} could not be composed: {exc}") from exc

    workers = max(1, min(settings.FRAME_EXTRACT_WORKERS, pair_count))
    log.info(
        "composing %d pairs (%dx%d, %d photos) with %d workers",
        pair_count,
        out_width,
        out_height,
        photo_count,
        workers,
    )
    pairs: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Photo order matters: keep the futures list ordered so the returned
        # paths (and the progress figures) follow the source sequence.
        futures = [pool.submit(build_one, index) for index in range(pair_count)]
        for done, future in enumerate(futures, start=1):
            pairs.append(future.result())
            if progress_cb is not None:
                progress_cb(done / pair_count)
    return pairs
