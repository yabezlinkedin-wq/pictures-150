"""
Final video composition.

    slideshow (20:00)  +  vocals.wav  +  overlay.png  ->  final_video.mp4

* the vocal track is padded with silence when it is shorter than 20 minutes
  and trimmed when it is longer (``apad`` + ``-t 1200``) - never synthesised;
* the overlay keeps its alpha channel and sits on top of the picture for the
  whole video;
* when no overlay is used the slideshow stream can be copied (``-c:v copy``),
  which makes the first attempt almost instant;
* if the produced file exceeds the upload limit (the Bot API mode default, or
  the per-job override the web interface passes) the video is re-encoded with
  a higher CRF (up to ``_MAX_SIZE_ATTEMPTS`` tries) before giving up with a
  friendly error.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from processing.errors import OutputTooLargeError, RenderError
from processing.ffmpeg_utils import (
    ProgressCallback,
    ffprobe_json,
    human_size,
    run_ffmpeg,
)
from processing.video_info import format_duration

log = logging.getLogger(__name__)

_DURATION_TOLERANCE = 1.0  # seconds around 20:00
_MAX_SIZE_ATTEMPTS = 3
_CRF_STEP = 6
_MAX_CRF = 34


def render_final(
    slideshow_path: Path,
    vocals_path: Path,
    out_path: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    overlay_png: Path | None = None,
    overlay_xy: tuple[int, int] | None = None,
    upload_limit: int | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_event=None,  # threading.Event | None
) -> Path:
    """Compose the final MP4 and guarantee its duration is exactly 20:00.

    ``upload_limit`` overrides the Bot-API-derived size guard (the web
    interface passes its own - Telegram's caps do not apply to a browser
    download).  ``None`` keeps the default limit of the current API mode.
    """
    total_duration = float(settings.SLIDESHOW_DURATION_SECONDS)
    if upload_limit is None:
        upload_limit = settings.effective_upload_limit()
    crf = settings.FINAL_CRF
    video_copy = overlay_png is None

    last_error: Exception | None = None
    for attempt in range(1, _MAX_SIZE_ATTEMPTS + 1):
        if out_path.exists():
            out_path.unlink()
        _render_once(
            slideshow_path,
            vocals_path,
            out_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            overlay_png=overlay_png,
            overlay_xy=overlay_xy,
            total_duration=total_duration,
            video_copy=video_copy,
            crf=crf,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )
        duration = _verify_output(out_path, total_duration)
        size = out_path.stat().st_size
        log.info(
            "final render attempt %d/%d: %s, %s, crf=%d, video_copy=%s",
            attempt,
            _MAX_SIZE_ATTEMPTS,
            format_duration(duration),
            human_size(size),
            crf,
            video_copy,
        )
        if size <= upload_limit:
            return out_path

        last_error = OutputTooLargeError(
            f"Final video is {human_size(size)} which exceeds the "
            f"{human_size(upload_limit)} upload limit after {attempt} attempt(s)."
        )
        if attempt == _MAX_SIZE_ATTEMPTS:
            break
        # encode harder / stop copying the stream
        video_copy = False
        crf = min(_MAX_CRF, crf + _CRF_STEP)
        log.warning(
            "final video %s exceeds the upload limit %s - retrying with CRF %d",
            human_size(size),
            human_size(upload_limit),
            crf,
        )

    assert last_error is not None  # loop always sets it before breaking
    raise last_error


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _render_once(
    slideshow_path: Path,
    vocals_path: Path,
    out_path: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    overlay_png: Path | None,
    overlay_xy: tuple[int, int] | None,
    total_duration: float,
    video_copy: bool,
    crf: int,
    progress_cb: ProgressCallback | None,
    cancel_event,
) -> None:
    args: list[str] = ["-y", "-i", str(slideshow_path), "-i", str(vocals_path)]
    video_filters: list[str] = []
    audio_filters = "[1:a]apad[aout]"

    if overlay_png is not None:
        if overlay_xy is None:
            raise RenderError("Overlay image requires a position.")
        x, y = overlay_xy
        args += ["-i", str(overlay_png)]
        # out_range=tv: the RGB overlay can flip the frames back to full range,
        # which would make the final stream report yuvj420p
        video_filters.append(
            f"[0:v][2:v]overlay={int(x)}:{int(y)}:format=auto,"
            "scale=out_range=tv,format=yuv420p[vout]"
        )
    elif video_copy:
        # re-use the already encoded slideshow stream
        pass
    else:
        video_filters.append("[0:v]scale=out_range=tv,format=yuv420p[vout]")

    filter_complex = ";".join(video_filters + [audio_filters])
    args += ["-filter_complex", filter_complex]
    args += ["-map", "[vout]" if video_filters else "0:v:0"]
    args += ["-map", "[aout]"]

    if video_copy:
        args += ["-c:v", "copy"]
    else:
        args += [
            "-c:v",
            "libx264",
            "-preset",
            settings.FINAL_PRESET,
            "-crf",
            str(crf),
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
        ]

    args += [
        "-c:a",
        "aac",
        "-b:a",
        settings.AUDIO_BITRATE,
        "-ar",
        str(settings.AUDIO_SAMPLE_RATE),
        "-ac",
        str(settings.AUDIO_CHANNELS),
        # pad shorter audio with silence, then cut everything at exactly 20:00
        "-t",
        f"{total_duration:.3f}",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    run_ffmpeg(
        args,
        total_duration=total_duration,
        progress_cb=progress_cb,
        timeout=settings.JOB_TIMEOUT_SECONDS,
        error_cls=RenderError,
        error_message="Final rendering failed",
        cancel_event=cancel_event,
    )


def _verify_output(path: Path, expected_duration: float) -> float:
    if not path.exists() or path.stat().st_size == 0:
        raise RenderError("The final video file was not created.")
    try:
        data = ffprobe_json(path)
    except Exception as exc:  # noqa: BLE001 - unreadable output is a render failure
        raise RenderError(f"The final video cannot be read back: {exc}") from exc

    streams = data.get("streams") or []
    if not any(s.get("codec_type") == "video" for s in streams):
        raise RenderError("The final video has no video stream.")
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise RenderError("The final video has no audio stream.")

    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if abs(duration - expected_duration) > _DURATION_TOLERANCE:
        raise RenderError(
            f"The final video has duration {duration:.2f}s instead of "
            f"{expected_duration:.2f}s."
        )
    return duration
