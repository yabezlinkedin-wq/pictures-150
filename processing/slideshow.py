"""
Slideshow creation.

150 stills x 8 s = 1200 s = exactly 20 minutes.  Each still is one pair
composite (two photos side by side, faces centred - see ``pair_builder``),
so the picture is completely static for its full 8 seconds.

Implementation notes
--------------------
* A concat-demuxer file lists every still with an explicit ``duration``.
  The final still is listed twice (without a duration) so that the very last
  8 seconds are actually held on screen - a well known FFmpeg idiom.
* ``-t 1200`` caps the output at exactly 20:00 even if timestamps drift.
* ``-tune stillimage`` lets x264 spend its bitrate on the few changing frames,
  which keeps the file small and the encode fast.
* When an overlay is given it is drawn into this very encode.  The final step
  can then stream-copy the picture, which skips a second full encode and lets
  the overlay go through a single generation of compression.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from processing.errors import SlideshowError
from processing.ffmpeg_utils import ProgressCallback, ffprobe_json, run_ffmpeg

log = logging.getLogger(__name__)

_DURATION_TOLERANCE = 1.0  # seconds


def write_concat_file(frames: list[Path], concat_path: Path, duration_each: float) -> None:
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        lines.append(f"file '{frame.name}'")
        lines.append(f"duration {duration_each:.6f}")
    # hold the last frame for its full duration as well
    lines.append(f"file '{frames[-1].name}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_slideshow(
    frames: list[Path],
    out_path: Path,
    out_width: int,
    out_height: int,
    *,
    overlay_png: Path | None = None,
    overlay_xy: tuple[int, int] | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_event=None,  # threading.Event | None
) -> Path:
    if len(frames) != settings.FRAME_COUNT:
        raise SlideshowError(
            f"Expected {settings.FRAME_COUNT} stills, got {len(frames)}."
        )

    frames_dir = frames[0].parent
    # The concat file lists bare frame names, so it must live next to the
    # frames - ffmpeg runs with cwd=frames_dir (long concat inputs).
    concat_path = frames_dir / "slideshow_concat.txt"
    write_concat_file(frames, concat_path, settings.FRAME_DURATION_SECONDS)

    total_duration = settings.SLIDESHOW_DURATION_SECONDS
    # JPEG frames decode as full-range (yuvj420p) - converting to limited range
    # here is what makes the final stream report plain ``yuv420p`` instead of
    # ``yuvj420p`` (a plain ``format=yuv420p`` keeps the full-range flag).
    filter_chain = (
        f"fps={settings.SLIDESHOW_FPS},"
        f"scale={out_width}:{out_height}:force_original_aspect_ratio=decrease:"
        "flags=lanczos:in_range=auto:out_range=tv,"
        f"pad={out_width}:{out_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1,format=yuv420p"
    )
    args = [
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_path.name,  # relative to cwd below
    ]
    if overlay_png is not None:
        if overlay_xy is None:
            raise SlideshowError("Overlay image requires a position.")
        x, y = overlay_xy
        args += ["-i", str(overlay_png.resolve())]  # absolute: cwd is the frames dir
        # The overlay is composited right into the slideshow encode, so the
        # final step can copy this stream untouched.  out_range=tv keeps the
        # RGB overlay from flipping the frames to full range (see renderer).
        filter_chain = (
            f"[0:v]{filter_chain}[base];"
            f"[base][1:v]overlay={int(x)}:{int(y)}:format=auto,"
            "scale=out_range=tv,format=yuv420p[vout]"
        )
        args += ["-filter_complex", filter_chain, "-map", "[vout]"]
    else:
        args += ["-vf", filter_chain]

    args += [
        "-c:v",
        "libx264",
        "-preset",
        settings.SLIDESHOW_PRESET,
        "-crf",
        str(settings.SLIDESHOW_CRF),
        "-tune",
        "stillimage",
        "-t",
        f"{total_duration:.3f}",
        "-movflags",
        "+faststart",
        "-an",
        # absolute: ffmpeg runs with cwd=frames_dir, so a bare name would
        # end up inside the frames directory
        str(out_path.resolve()),
    ]
    # (-progress pipe:1 is injected by run_ffmpeg when a callback is given)

    log.info(
        "creating slideshow: %d stills x %gs -> %.0fs at %dx%d%s",
        len(frames),
        settings.FRAME_DURATION_SECONDS,
        total_duration,
        out_width,
        out_height,
        f" with overlay at {overlay_xy}" if overlay_png is not None else "",
    )
    run_ffmpeg(
        args,
        total_duration=total_duration,
        progress_cb=progress_cb,
        timeout=settings.JOB_TIMEOUT_SECONDS,
        cwd=frames_dir,
        error_cls=SlideshowError,
        error_message="Slideshow creation failed",
        cancel_event=cancel_event,
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SlideshowError("Slideshow file was not created.")

    duration = verify_video_duration(out_path, total_duration, tolerance=_DURATION_TOLERANCE)
    log.info("slideshow ready: %.2fs (%.1f MB)", duration, out_path.stat().st_size / 1024 / 1024)
    return out_path


def verify_video_duration(path: Path, expected: float, *, tolerance: float = _DURATION_TOLERANCE) -> float:
    data = ffprobe_json(path)
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if abs(duration - expected) > tolerance:
        raise SlideshowError(
            f"{path.name} has duration {duration:.2f}s, expected {expected:.2f}s "
            f"(tolerance {tolerance:.1f}s)."
        )
    return duration
