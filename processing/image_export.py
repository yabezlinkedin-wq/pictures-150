"""
Picture-only job pipeline.

    source video
      -> 300 representative photos                 (stage 1)
      -> 150 side-by-side pictures                 (stage 2)
      -> one ZIP with the 150 JPEGs                (stage 3)

There is no slideshow, no vocal separation, no overlay and no final MP4.
The 150 pictures are exactly what the old 20-minute video showed: photo 1
and photo 2 side by side form picture 1, photo 299 and photo 300 form
picture 150.  The web interface downloads the ZIP.

The pipeline is synchronous (it runs inside a worker thread) and reports
progress through ``progress_cb(overall, stage_index, stage_fraction, detail)``
where ``overall`` is a 0..1 fraction across all three stages.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import settings
from processing import frame_extractor, pair_builder, video_info
from processing.errors import (
    InsufficientDiskSpaceError,
    JobCancelledError,
    JobTimeoutError,
    PictureExportError,
)
from processing.ffmpeg_utils import ensure_tools_available, human_size
from processing.video_info import VideoInfo, compute_output_dimensions, format_duration

log = logging.getLogger(__name__)

STAGE_PHOTOS = 1
STAGE_PAIRS = 2
STAGE_ZIP = 3

# Selection needs one ffmpeg seek per photo, pair building is pure CPU, and
# zipping stored JPEGs is nearly instant - hence the small last share.
STAGE_WEIGHTS: tuple[float, ...] = (0.50, 0.45, 0.05)
STAGE_COUNT = len(STAGE_WEIGHTS)

ProgressReporter = Callable[[float, int, float, str], None]


@dataclass
class PicturesJobSpec:
    """Everything the picture pipeline needs to process one upload."""

    job_id: str
    source_path: Path
    work_dir: Path


@dataclass
class PicturesResult:
    zip_path: Path
    picture_count: int
    size_bytes: int
    video_info: VideoInfo
    frames_report: frame_extractor.FrameQualityReport
    elapsed_seconds: float
    stage_seconds: dict[int, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def run_pictures_pipeline(
    spec: PicturesJobSpec,
    *,
    progress_cb: ProgressReporter | None = None,
    cancel_event: threading.Event | None = None,
) -> PicturesResult:
    """Build the 150 pictures and return the finished ZIP."""
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
    # pre-flight: tools, video, disk space
    # ------------------------------------------------------------------ #
    ensure_tools_available()
    check_cancelled()

    info = video_info.probe_video(spec.source_path)
    _check_disk_space(job_dir, spec.source_path.stat().st_size)

    out_width, out_height = compute_output_dimensions(info)
    log.info(
        "job %s: source %s, %s, %dx%d -> %d pictures at %dx%d",
        spec.job_id,
        format_duration(info.duration),
        info.resolution_label,
        info.display_width,
        info.display_height,
        settings.FRAME_COUNT,
        out_width,
        out_height,
    )

    # ------------------------------------------------------------------ #
    # stage 1 - 300 representative photos (150 pairs)
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    with _StageTimer(STAGE_PHOTOS):
        emit(STAGE_PHOTOS, 0.0, "Selecting 300 representative photos")
        selection = frame_extractor.extract_frames(
            info,
            job_dir,
            progress_cb=lambda fraction: emit(STAGE_PHOTOS, fraction),
            cancel_event=cancel_event,
        )
        emit(STAGE_PHOTOS, 1.0, selection.report.summary())
    log.info("job %s stage 1/3 done: %s", spec.job_id, selection.report.summary())

    # ------------------------------------------------------------------ #
    # stage 2 - 150 side-by-side pictures
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    with _StageTimer(STAGE_PAIRS):
        emit(STAGE_PAIRS, 0.0, "Building 150 side-by-side pictures")
        pairs = pair_builder.build_pairs(
            selection.frames,
            out_width,
            out_height,
            progress_cb=lambda fraction: emit(STAGE_PAIRS, fraction),
            cancel_event=cancel_event,
        )
        emit(STAGE_PAIRS, 1.0, "150 pictures ready")
    log.info(
        "job %s stage 2/3 done: %d pictures in %.1fs",
        spec.job_id,
        len(pairs),
        stage_seconds[STAGE_PAIRS],
    )

    # ------------------------------------------------------------------ #
    # stage 3 - one ZIP with the 150 JPEGs
    # ------------------------------------------------------------------ #
    check_deadline()
    check_cancelled()
    zip_path = job_dir / settings.PICTURES_ZIP_FILENAME
    with _StageTimer(STAGE_ZIP):
        emit(STAGE_ZIP, 0.0, "Zipping the 150 pictures")
        _write_zip(
            pairs,
            zip_path,
            progress_cb=lambda fraction: emit(STAGE_ZIP, fraction),
            cancel_event=cancel_event,
        )
        emit(STAGE_ZIP, 1.0, "ZIP ready")
    log.info(
        "job %s stage 3/3 done: %s in %.1fs",
        spec.job_id,
        human_size(zip_path.stat().st_size),
        stage_seconds[STAGE_ZIP],
    )

    # The individual photos and composites are no longer needed once they are
    # inside the ZIP - delete them to keep the job directory small.
    _cleanup_intermediates(job_dir, (job_dir / settings.FRAMES_DIRNAME,))

    elapsed = time.monotonic() - started
    log.info(
        "job %s finished in %s: %s (%d pictures)",
        spec.job_id,
        format_duration(elapsed),
        human_size(zip_path.stat().st_size),
        len(pairs),
    )
    return PicturesResult(
        zip_path=zip_path,
        picture_count=len(pairs),
        size_bytes=zip_path.stat().st_size,
        video_info=info,
        frames_report=selection.report,
        elapsed_seconds=elapsed,
        stage_seconds=stage_seconds,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_zip(
    pictures: list[Path],
    zip_path: Path,
    *,
    progress_cb: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Store the JPEGs in order as ``picture_001.jpg`` ... ``picture_150.jpg``.

    JPEGs do not compress any further, so ``ZIP_STORED`` keeps the ZIP as
    small as the pictures themselves and needs no extra CPU.
    """
    total = len(pictures)
    if total == 0:
        raise PictureExportError("No pictures were produced for the ZIP.")
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
            for done, picture in enumerate(pictures, start=1):
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelledError("ZIP creation cancelled")
                bundle.write(picture, arcname=f"picture_{done:03d}.jpg")
                if progress_cb is not None:
                    progress_cb(done / total)
    except JobCancelledError:
        zip_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - report the ZIP step clearly
        zip_path.unlink(missing_ok=True)
        raise PictureExportError(f"The picture ZIP could not be written: {exc}") from exc


def _check_disk_space(job_dir: Path, source_size: int) -> None:
    """Fail early when the disk cannot hold the job's working files."""
    try:
        usage = shutil.disk_usage(job_dir)
    except OSError as exc:  # pragma: no cover - exotic filesystem
        log.warning("could not check free disk space: %s", exc)
        return
    floor = int(settings.MIN_FREE_DISK_GB * (1024 ** 3))
    # source + photos + pairs + zip (+ headroom)
    estimated = max(floor, int(source_size * 3))
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
