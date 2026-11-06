"""
End-to-end job pipeline.

    source video
      -> 300 representative photos           (stage 1)
      -> 150 pair stills, overlay baked in   (stage 2, pairs + overlay first)
      -> original audio -> vocal separation  (stage 3)
      -> overlay checkout                    (stage 4, reported after stage 3)
      -> final MP4 (copy the picture, mux in the voice)  (stage 5)

The overlay is drawn during the slideshow encode, so the final step never
re-encodes the picture - it only muxes the separated vocals into the copy.

The pipeline is fully synchronous (it runs inside a worker thread) and reports
progress through ``progress_cb(overall, stage_index, stage_fraction, detail)``
where ``overall`` is a 0..1 fraction across all five stages.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import settings
from processing import (
    audio_extractor,
    frame_extractor,
    overlay as overlay_module,
    pair_builder,
    renderer,
    slideshow as slideshow_module,
    video_info,
    vocal_separator,
)
from processing.errors import (
    InsufficientDiskSpaceError,
    JobCancelledError,
    JobTimeoutError,
    NoAudioStreamError,
)
from processing.ffmpeg_utils import ensure_tools_available, human_size
from processing.video_info import VideoInfo, compute_output_dimensions, format_duration

log = logging.getLogger(__name__)

# Stage indices 1..5 (the five steps shown in Telegram) and their share of the
# overall progress bar.  Separation dominates because it is by far the slowest.
STAGE_FRAMES = 1
STAGE_SLIDESHOW = 2
STAGE_VOCALS = 3
STAGE_OVERLAY = 4
STAGE_RENDER = 5

STAGE_WEIGHTS: tuple[float, ...] = (0.25, 0.20, 0.35, 0.02, 0.18)
STAGE_COUNT = len(STAGE_WEIGHTS)

# If the audio track is shorter than the slideshow this fraction of the stage
# covers extracting the original audio; the rest is the separation model.
_AUDIO_EXTRACT_SHARE = 0.15
_NORMALISE_SHARE = 0.03

# Share of stage 2 spent composing the 150 pair stills; the slideshow encode
# itself takes the rest of the stage (it is far slower per unit of progress).
_PAIRS_SHARE = 0.20

ProgressReporter = Callable[[float, int, float, str], None]


@dataclass
class JobSpec:
    """Everything a worker needs to process one user request."""

    job_id: str
    user_id: int
    chat_id: int
    source_path: Path
    work_dir: Path
    overlay_path: Path | None = None
    overlay_position: str | None = None
    overlay_size_label: str | None = None


@dataclass
class PipelineResult:
    final_path: Path
    duration: float
    size_bytes: int
    video_info: VideoInfo
    frames_report: frame_extractor.FrameQualityReport
    overlay_applied: bool
    elapsed_seconds: float
    stage_seconds: dict[int, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def run_pipeline(
    spec: JobSpec,
    *,
    max_output_size: int | None = None,
    progress_cb: ProgressReporter | None = None,
    cancel_event: threading.Event | None = None,
) -> PipelineResult:
    """Run every processing stage and return the finished video.

    ``max_output_size`` overrides the Bot-API-derived size guard for this job.
    The Telegram bot leaves it unset (the upload goes through the Bot API);
    the web interface passes its own limit because the file is downloaded
    straight from the server, where Telegram's caps do not exist.
    """
    started = time.monotonic()
    job_dir = Path(spec.work_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    stage_seconds: dict[int, float] = {}

    last_overall = 0.0

    def emit(stage: int, fraction: float, detail: str = "") -> None:
        nonlocal last_overall
        if progress_cb is None:
            return
        fraction = max(0.0, min(1.0, fraction))
        overall = sum(STAGE_WEIGHTS[: stage - 1]) + STAGE_WEIGHTS[stage - 1] * fraction
        overall = max(last_overall, min(1.0, overall))
        last_overall = overall
        progress_cb(overall, stage, fraction, detail)

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelledError("Job cancelled by the user")

    def check_deadline() -> None:
        if time.monotonic() - started > settings.JOB_TIMEOUT_SECONDS:
            raise JobTimeoutError(
                f"Job exceeded the overall time budget of {settings.JOB_TIMEOUT_SECONDS}s"
            )

    class _StageTimer:
        """Measure how long one stage took (recorded in ``stage_seconds``)."""

        def __init__(self, stage: int) -> None:
            self.stage = stage
            self.t0 = 0.0

        def __enter__(self) -> "_StageTimer":
            self.t0 = time.monotonic()
            return self

        def __exit__(self, *_exc) -> bool:
            stage_seconds[self.stage] = time.monotonic() - self.t0
            return False

    # ------------------------------------------------------------------ #
    # pre-flight: tools, video, audio presence, overlay, disk space
    # ------------------------------------------------------------------ #
    ensure_tools_available()
    check_cancelled()

    info = video_info.probe_video(spec.source_path)
    if not info.has_audio:
        raise NoAudioStreamError(f"{spec.source_path.name} has no audio stream")

    if spec.overlay_path is not None:
        overlay_module.validate_overlay(spec.overlay_path)

    _check_disk_space(job_dir, spec.source_path.stat().st_size)

    out_width, out_height = compute_output_dimensions(info)
    log.info(
        "job %s: source %s, %s, %.1f fps, %dx%d -> canvas %dx%d, overlay=%s",
        spec.job_id,
        format_duration(info.duration),
        info.resolution_label,
        info.fps,
        info.display_width,
        info.display_height,
        out_width,
        out_height,
        spec.overlay_path.name if spec.overlay_path else "none",
    )

    # ------------------------------------------------------------------ #
    # stage 1 - extract 300 representative photos (150 pairs)
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    with _StageTimer(STAGE_FRAMES):
        emit(STAGE_FRAMES, 0.0, "Selecting 300 representative photos")
        selection = frame_extractor.extract_frames(
            info,
            job_dir,
            progress_cb=lambda fraction: emit(STAGE_FRAMES, fraction),
            cancel_event=cancel_event,
        )
        emit(STAGE_FRAMES, 1.0, selection.report.summary())
    log.info("job %s stage 1/5 done: %s", spec.job_id, selection.report.summary())

    # ------------------------------------------------------------------ #
    # stage 2 - 150 pair stills -> 150 x 8 s slideshow (exactly 20:00)
    #
    # The pairs are composed first (photo 1 left / photo 2 right, faces
    # centred, static).  When an overlay is requested it is prepared here and
    # drawn into the slideshow encode itself: the picture then goes through a
    # single generation of compression and the final step only stream-copies.
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    slideshow_path = job_dir / settings.SLIDESHOW_FILENAME
    overlay_png: Path | None = None
    overlay_xy: tuple[int, int] | None = None
    with _StageTimer(STAGE_SLIDESHOW):
        emit(STAGE_SLIDESHOW, 0.0, "Building 150 photo pairs")
        pairs = pair_builder.build_pairs(
            selection.frames,
            out_width,
            out_height,
            progress_cb=lambda fraction: emit(STAGE_SLIDESHOW, _PAIRS_SHARE * fraction),
            cancel_event=cancel_event,
        )
        if spec.overlay_path is not None:
            overlay_png, overlay_xy = overlay_module.prepare_overlay(
                spec.overlay_path,
                job_dir / "overlay_prepared.png",
                out_width,
                out_height,
                position=overlay_module.normalize_position(spec.overlay_position),
                size_label=overlay_module.normalize_size_label(spec.overlay_size_label),
            )
        emit(STAGE_SLIDESHOW, _PAIRS_SHARE, "150 pairs x 8 s = 20 minutes")
        slideshow_module.create_slideshow(
            pairs,
            slideshow_path,
            out_width,
            out_height,
            overlay_png=overlay_png,
            overlay_xy=overlay_xy,
            progress_cb=lambda fraction: emit(
                STAGE_SLIDESHOW, _PAIRS_SHARE + (1.0 - _PAIRS_SHARE) * fraction
            ),
            cancel_event=cancel_event,
        )
        emit(STAGE_SLIDESHOW, 1.0, "Slideshow rendered")
    log.info(
        "job %s stage 2/5 done: slideshow %s in %.1fs",
        spec.job_id,
        human_size(slideshow_path.stat().st_size),
        stage_seconds[STAGE_SLIDESHOW],
    )

    # ------------------------------------------------------------------ #
    # stage 3 - audio extraction + vocal separation (+ normalisation)
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    with _StageTimer(STAGE_VOCALS):
        emit(STAGE_VOCALS, 0.0, "Extracting the original audio")
        audio_path = job_dir / settings.AUDIO_FILENAME
        # Only the first 20 minutes ever reach the final video, so the
        # separation below never needs more audio than that.  Capping the
        # extraction up front makes sources longer than 20:00 proportionally
        # faster (the discarded part could not have affected the result).
        audio_seconds = min(info.duration, float(settings.SLIDESHOW_DURATION_SECONDS))
        audio_extractor.extract_audio(
            spec.source_path,
            audio_path,
            duration=audio_seconds,
            limit_seconds=audio_seconds,
            progress_cb=lambda fraction: emit(STAGE_VOCALS, _AUDIO_EXTRACT_SHARE * fraction),
            cancel_event=cancel_event,
        )

        emit(STAGE_VOCALS, _AUDIO_EXTRACT_SHARE, "Separating vocals from music")
        vocals_path = vocal_separator.separate_vocals(
            audio_path,
            job_dir,
            progress_cb=lambda fraction: emit(
                STAGE_VOCALS,
                _AUDIO_EXTRACT_SHARE + (1.0 - _AUDIO_EXTRACT_SHARE - _NORMALISE_SHARE) * fraction,
            ),
            cancel_event=cancel_event,
        )

        if settings.AUDIO_LOUDNESS_NORM:
            emit(STAGE_VOCALS, 1.0 - _NORMALISE_SHARE, "Normalising the vocal volume")
            normalised = job_dir / "vocals_normalised.wav"
            audio_extractor.normalize_loudness(
                vocals_path,
                normalised,
                duration=audio_seconds,
                cancel_event=cancel_event,
            )
            normalised.replace(vocals_path)

        emit(STAGE_VOCALS, 1.0, "Vocal track ready")
    log.info(
        "job %s stage 3/5 done: vocals %s in %.1fs",
        spec.job_id,
        human_size(vocals_path.stat().st_size),
        stage_seconds[STAGE_VOCALS],
    )
    _cleanup_intermediates(job_dir, (audio_path,))

    # ------------------------------------------------------------------ #
    # stage 4 - overlay preparation (already done inside stage 2; the
    # overlay is baked into the slideshow, so this only reports its place)
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    with _StageTimer(STAGE_OVERLAY):
        emit(STAGE_OVERLAY, 0.0, "Preparing the overlay")
        if overlay_png is not None:
            assert overlay_xy is not None  # together with overlay_png
            emit(STAGE_OVERLAY, 1.0, f"Overlay placed at {overlay_xy[0]},{overlay_xy[1]}")
        else:
            emit(STAGE_OVERLAY, 1.0, "No overlay provided")
    log.info("job %s stage 4/5 done: overlay=%s", spec.job_id, overlay_png or "none")

    # ------------------------------------------------------------------ #
    # stage 5 - final composition
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    final_path = job_dir / settings.FINAL_FILENAME
    with _StageTimer(STAGE_RENDER):
        emit(STAGE_RENDER, 0.0, "Rendering the final video")
        renderer.render_final(
            slideshow_path,
            vocals_path,
            final_path,
            canvas_width=out_width,
            canvas_height=out_height,
            # the overlay (when there is one) is already baked into the
            # slideshow, so the picture is stream-copied and only the audio
            # track is encoded here - that is what makes this step seconds long
            overlay_png=None,
            overlay_xy=None,
            upload_limit=max_output_size,
            progress_cb=lambda fraction: emit(STAGE_RENDER, fraction),
            cancel_event=cancel_event,
        )
        emit(STAGE_RENDER, 1.0, "Final video ready")
    log.info(
        "job %s stage 5/5 done: final %s in %.1fs",
        spec.job_id,
        human_size(final_path.stat().st_size),
        stage_seconds[STAGE_RENDER],
    )

    elapsed = time.monotonic() - started
    duration = float(settings.SLIDESHOW_DURATION_SECONDS)

    if settings.SAVE_OUTPUT_COPY:
        try:
            settings.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            destination = settings.OUTPUT_DIRECTORY / f"{spec.job_id}_{settings.FINAL_FILENAME}"
            shutil.copy2(final_path, destination)
            log.info("copy saved to %s", destination)
        except Exception:  # noqa: BLE001 - keeping a copy is best effort
            log.exception("could not save output copy for job %s", spec.job_id)

    _cleanup_intermediates(
        job_dir,
        (slideshow_path, vocals_path, job_dir / settings.FRAMES_DIRNAME)
        + ((overlay_png,) if overlay_png is not None else ()),
    )

    log.info(
        "job %s finished in %s (%s): %s",
        spec.job_id,
        format_duration(elapsed),
        {f"stage{s}": round(seconds, 1) for s, seconds in sorted(stage_seconds.items())},
        human_size(final_path.stat().st_size),
    )
    return PipelineResult(
        final_path=final_path,
        duration=duration,
        size_bytes=final_path.stat().st_size,
        video_info=info,
        frames_report=selection.report,
        overlay_applied=overlay_png is not None,
        elapsed_seconds=elapsed,
        stage_seconds=stage_seconds,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _check_disk_space(job_dir: Path, source_size: int) -> None:
    """Fail early when the disk cannot hold the job's working files."""
    try:
        usage = shutil.disk_usage(job_dir)
    except OSError as exc:  # pragma: no cover - exotic filesystem
        log.warning("could not check free disk space: %s", exc)
        return
    floor = int(settings.MIN_FREE_DISK_GB * (1024 ** 3))
    # source + wav + vocals + slideshow + final (+ headroom)
    estimated = max(floor, int(source_size * 4))
    if usage.free < estimated:
        raise InsufficientDiskSpaceError(
            f"Only {human_size(usage.free)} free on {job_dir.drive or job_dir}, "
            f"about {human_size(estimated)} required."
        )
    log.debug(
        "disk space ok: %s free, %s estimated for this job",
        human_size(usage.free),
        human_size(estimated),
    )


def _cleanup_intermediates(job_dir: Path, paths: tuple[Path, ...]) -> None:
    """Delete working files that are no longer needed (keeps disk usage low)."""
    if not settings.DELETE_TEMP_AFTER_SEND:
        return
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
        except OSError as exc:  # noqa: BLE001 - cleanup must never fail the job
            log.debug("could not remove intermediate %s: %s", path, exc)
    log.debug("job %s: intermediates cleaned up", job_dir.name)
