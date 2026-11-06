"""
Photo selection.

The source video is divided into ``PHOTO_COUNT`` (300) equal segments and one
representative photo is extracted from each segment - never 300 consecutive
frames.  Every candidate photo is checked for

* readability (corrupted JPEGs are rejected by FFmpeg/Pillow),
* near-black / near-white / flat (single colour) content,
* near-duplicates of the previously selected photo.

When a candidate is unusable, another position inside the same segment is
tried (see ``FRAME_CANDIDATE_OFFSETS``).  The function always returns exactly
300 photos: in the (very unlikely) worst case a neighbouring photo is reused
so that the pair count (and thus the slideshow length) stays exact.

The 300 photos are later combined into 150 pairs - photo 1 (left half) +
photo 2 (right half) form pair 1, and so on (see ``processing/pair_builder``).
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings
from processing.errors import FrameExtractionError, JobCancelledError
from processing.ffmpeg_utils import ProgressCallback, run_ffmpeg
from processing.video_info import (
    VideoInfo,
    compute_output_dimensions,
    compute_photo_dimensions,
)

log = logging.getLogger(__name__)

_MAX_ATTEMPTS_PER_SEGMENT = 3          # hard cap, keeps worst-case cost bounded
_SETTLE_AFTER_ATTEMPTS = 2             # accept the best duplicate after N tries
_SIGNATURE_SIZE = (32, 32)
_EXTRACT_TIMEOUT = 180.0


@dataclass
class FrameQualityReport:
    total: int = 0
    rejected_black: int = 0
    rejected_flat: int = 0
    rejected_corrupt: int = 0
    rejected_duplicate: int = 0
    segments_retried: int = 0
    segments_forced: int = 0
    frames_copied: int = 0
    extractions: int = 0

    def summary(self) -> str:
        parts = [
            f"{self.total} photos",
            f"{self.extractions} candidates extracted",
            f"rejected: {self.rejected_black} black, {self.rejected_flat} flat, "
            f"{self.rejected_corrupt} corrupt, {self.rejected_duplicate} duplicate",
            f"retried segments: {self.segments_retried}",
            f"forced (no clean candidate): {self.segments_forced}",
            f"copied from neighbour: {self.frames_copied}",
        ]
        return "; ".join(parts)


@dataclass
class _FrameAnalysis:
    path: Path
    readable: bool
    clean: bool
    reason: str
    score: float
    signature: np.ndarray | None = None
    mean: float = 0.0
    std: float = 0.0
    mae_prev: float = float("inf")


@dataclass
class _SegmentState:
    index: int
    attempts: int = 0
    chosen: _FrameAnalysis | None = None
    fallback: _FrameAnalysis | None = None  # best readable candidate (may be a duplicate)
    settled: bool = False

    @property
    def finished(self) -> bool:
        return self.chosen is not None or self.settled


@dataclass
class FrameSelectionResult:
    frames: list[Path] = field(default_factory=list)
    report: FrameQualityReport = field(default_factory=FrameQualityReport)


def extract_frames(
    info: VideoInfo,
    job_dir: Path,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_event=None,  # threading.Event | None
) -> FrameSelectionResult:
    """Select exactly ``PHOTO_COUNT`` representative photos and write them to disk.

    Photos are extracted at the size the pair composite wants (see
    ``compute_photo_dimensions``): big enough to fill half of the 16:9 canvas
    with a face-centred crop without upscaling, never bigger than the source.
    """
    photo_count = settings.PHOTO_COUNT
    frames_dir = job_dir / settings.FRAMES_DIRNAME
    candidates_dir = job_dir / ".candidates"
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    canvas_w, canvas_h = compute_output_dimensions(info)
    out_w, out_h = compute_photo_dimensions(info, canvas_w, canvas_h)
    segment_length = info.duration / photo_count
    offsets = settings.FRAME_CANDIDATE_OFFSETS
    report = FrameQualityReport(total=photo_count)

    states = [_SegmentState(index=i) for i in range(photo_count)]
    previous_signature: np.ndarray | None = None
    last_reported = 0.0

    def report_progress(fraction: float) -> None:
        nonlocal last_reported
        if progress_cb is None:
            return
        clamped = max(last_reported, max(0.0, min(1.0, fraction)))
        if clamped > last_reported:
            last_reported = clamped
            progress_cb(clamped)

    # ------------------------------------------------------------------ #
    # round 1: one candidate per segment, extracted in parallel
    # ------------------------------------------------------------------ #
    def extract_one(segment_index: int, attempt: int) -> Path | None:
        offset = offsets[min(attempt, len(offsets) - 1)]
        timestamp = segment_start(info, segment_index, photo_count) + offset * segment_length
        timestamp = max(0.0, min(timestamp, max(0.0, info.duration - 0.05)))
        out_path = candidates_dir / f"seg_{segment_index:03d}_a{attempt}.jpg"
        try:
            _extract_single_frame(info.path, timestamp, out_path, out_w, out_h, cancel_event)
            return out_path
        except JobCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad segment must not kill the job
            log.warning("Frame extraction failed (segment %d, attempt %d): %s", segment_index, attempt, exc)
            return None

    def run_extraction_batch(pending: list[_SegmentState], attempt: int) -> dict[int, Path | None]:
        results: dict[int, Path | None] = {}
        if not pending:
            return results
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelledError("Frame extraction cancelled")
        workers = max(1, min(settings.FRAME_EXTRACT_WORKERS, len(pending)))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(extract_one, state.index, attempt): state for state in pending
            }
            for future in futures:
                state = futures[future]
                results[state.index] = future.result()
                done += 1
                report_progress(0.65 * done / photo_count)
        report.extractions += len(pending)
        return results

    pending = list(states)
    for attempt in range(_MAX_ATTEMPTS_PER_SEGMENT):
        extracted = run_extraction_batch(pending, attempt)
        for state in pending:
            state.attempts += 1
            path = extracted.get(state.index)
            if path is None:
                report.rejected_corrupt += 1
                continue
            analysis = _analyse_frame(path, previous_signature)
            if not analysis.readable:
                report.rejected_corrupt += 1
                continue
            if analysis.clean:
                state.chosen = analysis
                previous_signature = analysis.signature
            else:
                if analysis.reason == "duplicate":
                    report.rejected_duplicate += 1
                elif analysis.reason == "flat":
                    report.rejected_flat += 1
                else:
                    report.rejected_black += 1

                if state.fallback is None or analysis.score > state.fallback.score:
                    state.fallback = analysis

                if analysis.reason == "duplicate" and state.attempts >= _SETTLE_AFTER_ATTEMPTS:
                    # The frame itself is fine, it only looks like the previous
                    # one - do not burn more extractions on this segment.
                    state.chosen = state.fallback
                    state.settled = True
                    previous_signature = state.chosen.signature
                    report.segments_forced += 1

        # progress: analysis of this round finished
        finished = sum(1 for s in states if s.finished or s.chosen is not None)
        report_progress(0.65 + 0.35 * finished / photo_count)

        pending = [s for s in states if not s.finished]
        if not pending:
            break
        if attempt == 0:
            report.segments_retried = len(pending)

    # ------------------------------------------------------------------ #
    # finalise: accept the best fallback, or copy a neighbour (last resort)
    # ------------------------------------------------------------------ #
    for state in states:
        if state.chosen is None and state.fallback is not None:
            state.chosen = state.fallback
            state.settled = True
            report.segments_forced += 1

    accepted: list[Path | None] = [s.chosen.path if s.chosen else None for s in states]

    last_good: Path | None = None
    for index, path in enumerate(accepted):
        if path is None:
            # copy forward from the previous frame, or backward from the next one
            neighbour = last_good or next((p for p in accepted[index + 1 :] if p), None)
            if neighbour is None:
                raise FrameExtractionError(
                    "No frame could be extracted from this video at all."
                )
            target = candidates_dir / f"seg_{index:03d}_copy.jpg"
            shutil.copyfile(neighbour, target)
            accepted[index] = target
            report.frames_copied += 1
        else:
            last_good = path

    # ------------------------------------------------------------------ #
    # move into frames/photo_001.jpg ... photo_300.jpg
    # ------------------------------------------------------------------ #
    frames: list[Path] = []
    for index, path in enumerate(accepted):
        assert path is not None  # guaranteed by the fallback logic above
        target = frames_dir / f"photo_{index + 1:03d}.jpg"
        shutil.move(str(path), str(target))
        frames.append(target)

    shutil.rmtree(candidates_dir, ignore_errors=True)

    if len(frames) != photo_count:  # pragma: no cover - defensive
        raise FrameExtractionError(
            f"Expected {photo_count} photos but produced {len(frames)}."
        )

    log.info("photo selection complete: %s", report.summary())
    report_progress(1.0)
    return FrameSelectionResult(frames=frames, report=report)


def segment_start(info: VideoInfo, segment_index: int, frame_count: int) -> float:
    return segment_index * (info.duration / frame_count)


def _extract_single_frame(
    source: Path,
    timestamp: float,
    out_path: Path,
    out_w: int,
    out_h: int,
    cancel_event=None,  # threading.Event | None
) -> None:
    """Accurate single-frame seek (input seek is fast; FFmpeg decodes to the exact time)."""
    if out_path.exists():
        out_path.unlink()
    args = [
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-vf",
        f"scale={out_w}:{out_h}:flags=lanczos",
        "-q:v",
        str(settings.JPEG_QUALITY),
        "-f",
        "image2",
        str(out_path),
    ]
    run_ffmpeg(
        args,
        timeout=_EXTRACT_TIMEOUT,
        error_cls=FrameExtractionError,
        error_message=f"Could not extract a frame at {timestamp:.3f}s",
        cancel_event=cancel_event,
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise FrameExtractionError(f"FFmpeg produced no frame at {timestamp:.3f}s")


def _analyse_frame(path: Path, previous_signature: np.ndarray | None) -> _FrameAnalysis:
    try:
        with Image.open(path) as image:
            image.load()
            gray = image.convert("L").resize(_SIGNATURE_SIZE, Image.BILINEAR)
    except Exception as exc:  # noqa: BLE001 - corrupted image
        log.debug("unreadable frame %s: %s", path, exc)
        return _FrameAnalysis(path, readable=False, clean=False, reason="corrupt", score=-1.0)

    signature = np.asarray(gray, dtype=np.float32)
    mean = float(signature.mean())
    std = float(signature.std())

    mae_prev = float("inf")
    if previous_signature is not None:
        mae_prev = float(np.abs(signature - previous_signature).mean())

    # meaningful content score: structure (std) + dissimilarity to the previous frame
    score = std + min(mae_prev if np.isfinite(mae_prev) else 20.0, 30.0) * 0.5

    # Black / flat frames are heavily penalised so that any "real" candidate
    # (even a near duplicate) is preferred as fallback.
    if mean < settings.BLACK_FRAME_MEAN_THRESHOLD:
        return _FrameAnalysis(path, True, False, "black", score - 100.0, signature, mean, std, mae_prev)
    if std < settings.FLAT_FRAME_STD_THRESHOLD:
        return _FrameAnalysis(path, True, False, "flat", score - 100.0, signature, mean, std, mae_prev)
    if mean > settings.BRIGHT_FRAME_MEAN_THRESHOLD and std < 2 * settings.FLAT_FRAME_STD_THRESHOLD:
        return _FrameAnalysis(path, True, False, "flat", score - 100.0, signature, mean, std, mae_prev)
    if previous_signature is not None and mae_prev < settings.DUPLICATE_FRAME_THRESHOLD:
        return _FrameAnalysis(path, True, False, "duplicate", score, signature, mean, std, mae_prev)

    return _FrameAnalysis(path, True, True, "", score, signature, mean, std, mae_prev)
