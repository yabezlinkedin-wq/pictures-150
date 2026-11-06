"""
Audio extraction and (optional) loudness normalisation.

The vocals stem of Demucs is usually quieter than the original mix, so a
single-pass EBU R128 loudness normalisation keeps the result at a comfortable
volume without distorting the voice.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from processing.errors import AudioExtractionError
from processing.ffmpeg_utils import ProgressCallback, run_ffmpeg

log = logging.getLogger(__name__)


def extract_audio(
    video_path: Path,
    out_wav: Path,
    *,
    duration: float | None = None,
    limit_seconds: float | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_event=None,  # threading.Event | None
) -> Path:
    """
    Extract the first audio stream as high quality PCM WAV (44.1 kHz, stereo).

    ``limit_seconds`` caps the extraction at the first N seconds.  The pipeline
    passes the slideshow length because audio beyond 20:00 never reaches the
    final video - skipping it up front makes the separation below faster for
    sources longer than 20 minutes without changing the result.
    """
    args = [
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-map",
        "0:a:0",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(settings.AUDIO_SAMPLE_RATE),
        "-ac",
        str(settings.AUDIO_CHANNELS),
    ]
    if limit_seconds is not None:
        args += ["-t", f"{limit_seconds:.3f}"]
    args.append(str(out_wav))
    run_ffmpeg(
        args,
        total_duration=duration,
        progress_cb=progress_cb,
        timeout=settings.JOB_TIMEOUT_SECONDS,
        error_cls=AudioExtractionError,
        error_message="Audio extraction failed",
        cancel_event=cancel_event,
    )

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise AudioExtractionError("Audio extraction produced no output (video has no audio?)")

    log.info("audio extracted: %s (%.1f MB)", out_wav.name, out_wav.stat().st_size / 1024 / 1024)
    return out_wav


def normalize_loudness(
    in_wav: Path,
    out_wav: Path,
    *,
    duration: float | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_event=None,  # threading.Event | None
) -> Path:
    """Single-pass EBU R128 normalisation (gentle - no dynamic compression)."""
    target = settings.AUDIO_LOUDNESS_TARGET
    true_peak = settings.AUDIO_TRUE_PEAK
    loudness_range = settings.AUDIO_LOUDNESS_RANGE
    args = [
        "-y",
        "-i",
        str(in_wav),
        "-af",
        f"loudnorm=I={target}:TP={true_peak}:LRA={loudness_range}",
        "-ar",
        str(settings.AUDIO_SAMPLE_RATE),
        "-ac",
        str(settings.AUDIO_CHANNELS),
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    run_ffmpeg(
        args,
        total_duration=duration,
        progress_cb=progress_cb,
        timeout=settings.JOB_TIMEOUT_SECONDS,
        error_cls=AudioExtractionError,
        error_message="Loudness normalisation failed",
        cancel_event=cancel_event,
    )
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise AudioExtractionError("Loudness normalisation produced no output")
    return out_wav
