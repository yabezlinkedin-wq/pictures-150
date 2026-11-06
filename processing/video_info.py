"""Video inspection (ffprobe) and output-geometry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import settings
from processing.errors import VideoInfoError, VideoTooShortError
from processing.ffmpeg_utils import ffprobe_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    rotation: int
    video_codec: str
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    size_bytes: int

    # -- derived ----------------------------------------------------------- #
    @property
    def display_width(self) -> int:
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation in (90, 270) else self.height

    @property
    def resolution_label(self) -> str:
        return f"{self.display_width}x{self.display_height}"

    @property
    def is_portrait(self) -> bool:
        return self.display_height > self.display_width


def _parse_fraction(value: str | None) -> float:
    """``"30000/1001"`` -> 29.97; tolerate plain numbers and garbage."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_hms(value: str | None) -> float:
    """``"00:25:12.345000000"`` -> seconds."""
    if not value:
        return 0.0
    try:
        parts = [float(p) for p in value.split(":")]
    except ValueError:
        return 0.0
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds


def _stream_rotation(stream: dict) -> int:
    rotation = 0
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            try:
                rotation = int(round(float(side_data["rotation"])))
            except (TypeError, ValueError):
                rotation = 0
    if not rotation:
        try:
            rotation = int(round(float((stream.get("tags") or {}).get("rotate", 0))))
        except (TypeError, ValueError):
            rotation = 0
    return rotation % 360


def probe_video(path: Path | str, *, validate_duration: bool = True) -> VideoInfo:
    """Inspect a media file and return a :class:`VideoInfo` (raises on garbage)."""
    path = Path(path)
    if not path.is_file():
        raise VideoInfoError(f"File does not exist: {path}")
    if path.stat().st_size == 0:
        raise VideoInfoError(f"File is empty: {path}")

    data = ffprobe_json(path)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) != 1),
        None,
    )
    if video_stream is None:
        raise VideoInfoError(f"No video stream found in {path}")

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for candidate in (
        fmt.get("duration"),
        video_stream.get("duration"),
        (video_stream.get("tags") or {}).get("DURATION"),
        (fmt.get("tags") or {}).get("DURATION"),
    ):
        try:
            duration = float(candidate) if candidate is not None else 0.0
        except (TypeError, ValueError):
            duration = _parse_hms(str(candidate)) if candidate else 0.0
        if duration > 0:
            break

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoInfoError(f"Invalid video dimensions in {path}")

    fps = _parse_fraction(video_stream.get("avg_frame_rate")) or _parse_fraction(
        video_stream.get("r_frame_rate")
    )
    if fps <= 0:
        fps = 25.0

    if duration <= 0:
        raise VideoInfoError(f"Could not determine the duration of {path}")

    if validate_duration and duration < settings.MIN_SOURCE_DURATION_SECONDS:
        raise VideoTooShortError(
            f"Video is too short ({duration:.1f}s < {settings.MIN_SOURCE_DURATION_SECONDS:.0f}s)"
        )

    info = VideoInfo(
        path=path,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        rotation=_stream_rotation(video_stream),
        video_codec=str(video_stream.get("codec_name") or "unknown"),
        has_audio=audio_stream is not None,
        audio_codec=str(audio_stream.get("codec_name")) if audio_stream else None,
        audio_sample_rate=_safe_int(audio_stream.get("sample_rate")) if audio_stream else None,
        audio_channels=_safe_int(audio_stream.get("channels")) if audio_stream else None,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
    )
    log.info(
        "probe: %s | %.1fs | %s | %.2f fps | rotation %d | audio=%s(%s) | %.1f MB",
        path.name,
        info.duration,
        info.resolution_label,
        info.fps,
        info.rotation,
        info.has_audio,
        info.audio_codec,
        info.size_bytes / (1024 * 1024),
    )
    return info


def _safe_int(value) -> int | None:  # type: ignore[no-untyped-def]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_output_dimensions(info: VideoInfo) -> tuple[int, int]:
    """
    Output canvas for the slideshow / final video - always 16:9.

    Every job renders at 16:9 regardless of the source aspect ratio: each
    still holds two side-by-side photos (one per half), so the canvas keeps
    the source's useful vertical resolution where possible and derives the
    width at 16:9, capped by ``MAX_OUTPUT_WIDTH`` / ``MAX_OUTPUT_HEIGHT``.
    Both dimensions stay even (yuv420p friendly).
    """
    max_w, max_h = settings.MAX_OUTPUT_WIDTH, settings.MAX_OUTPUT_HEIGHT
    if max_w < max_h:
        # 16:9 is landscape by definition - keep a sane cap even if an
        # operator swapped the two values.
        max_w, max_h = max_h, max_w

    height = max(2, min(int(info.display_height), max_h))
    width = int(round(height * 16 / 9))
    if width > max_w:
        width = max_w
        height = int(round(width * 9 / 16))
    width -= width % 2
    height -= height % 2
    return max(2, width), max(2, height)


def compute_photo_dimensions(
    info: VideoInfo, out_width: int, out_height: int
) -> tuple[int, int]:
    """
    Extraction size for the 300 representative photos.

    Every photo later fills one half of the 16:9 canvas (``out_width/2`` x
    ``out_height``) with a crop window around its main face.  Photos are
    extracted as large as useful and never enlarged here:

    * at least the resolution that already covers a whole half without any
      upscaling (``cover scale``), so the crop stays sharp even when it has
      room to breathe;
    * at most 2x that requirement - a density that still serves the deepest
      zoom-in a face may trigger - so 4K+ sources do not flood the job
      directory with needless megabytes;
    * never above the source's own resolution (no fake detail).
    """
    cover_scale = max(
        (out_width / 2.0) / info.display_width,
        out_height / info.display_height,
    )
    # twice the cover density (leaves zoom headroom for face-centred crops),
    # never above the source's own resolution
    scale = min(1.0, 2.0 * cover_scale)
    width = max(2, int(round(info.display_width * scale)))
    height = max(2, int(round(info.display_height * scale)))
    width -= width % 2
    height -= height % 2
    return max(2, width), max(2, height)


def format_duration(seconds: float) -> str:
    """``1512.4`` -> ``"25:12"`` (``"1:02:03"`` once past an hour)."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
