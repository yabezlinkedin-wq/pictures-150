"""
Central configuration for the video processing bot.

Every value is read from environment variables (optionally supplied through a
``.env`` file in the project root).  No credentials are hard-coded here.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is an optional (but recommended) dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - tiny fallback so scripts never crash
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _int(name: str, default: int) -> int:
    raw = _str(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = _str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
TELEGRAM_BOT_TOKEN = _str("TELEGRAM_BOT_TOKEN")

# Only needed by the *local* Bot API server (docker-compose) and for reference.
TELEGRAM_API_ID = _str("TELEGRAM_API_ID")
TELEGRAM_API_HASH = _str("TELEGRAM_API_HASH")

# Empty -> official https://api.telegram.org (20 MB download / 50 MB upload
# limits).  Point this at a local Bot API server to lift the limits, e.g.
# ``http://localhost:8081/bot``.
TELEGRAM_API_BASE_URL = _str("TELEGRAM_API_BASE_URL").rstrip("/")
TELEGRAM_BASE_FILE_URL = _str("TELEGRAM_BASE_FILE_URL").rstrip("/")
TELEGRAM_LOCAL_MODE = _bool("TELEGRAM_LOCAL_MODE", False)

if TELEGRAM_API_BASE_URL:
    TELEGRAM_LOCAL_MODE = True
    if not TELEGRAM_BASE_FILE_URL:
        # http://host:8081/bot  ->  http://host:8081/file/bot
        base = TELEGRAM_API_BASE_URL
        TELEGRAM_BASE_FILE_URL = (
            base[: -len("/bot")] + "/file/bot" if base.endswith("/bot") else base + "/file/bot"
        )

TELEGRAM_CONNECT_TIMEOUT = _float("TELEGRAM_CONNECT_TIMEOUT", 30.0)
TELEGRAM_READ_TIMEOUT = _float("TELEGRAM_READ_TIMEOUT", 120.0)
TELEGRAM_WRITE_TIMEOUT = _float("TELEGRAM_WRITE_TIMEOUT", 900.0)
TELEGRAM_POOL_TIMEOUT = _float("TELEGRAM_POOL_TIMEOUT", 60.0)

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
MB = 1024 * 1024

# Hard upload limit of the official Bot API.  A local Bot API server allows up
# to 2000 MB.
OFFICIAL_DOWNLOAD_LIMIT = 20 * MB
OFFICIAL_UPLOAD_LIMIT = 50 * MB
LOCAL_SERVER_LIMIT = 2000 * MB

MAX_VIDEO_SIZE = _int("MAX_VIDEO_SIZE", LOCAL_SERVER_LIMIT)
MAX_OUTPUT_SIZE = _int("MAX_OUTPUT_SIZE", 0)  # 0 -> derived from the API mode

_EFFECTIVE_DOWNLOAD_LIMIT = (
    min(MAX_VIDEO_SIZE, LOCAL_SERVER_LIMIT)
    if TELEGRAM_LOCAL_MODE
    else min(MAX_VIDEO_SIZE, OFFICIAL_DOWNLOAD_LIMIT)
)
_EFFECTIVE_UPLOAD_LIMIT = (
    min(MAX_OUTPUT_SIZE or (LOCAL_SERVER_LIMIT - 100 * MB), LOCAL_SERVER_LIMIT)
    if TELEGRAM_LOCAL_MODE
    else min(MAX_OUTPUT_SIZE or OFFICIAL_UPLOAD_LIMIT, OFFICIAL_UPLOAD_LIMIT)
)


# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #
TEMP_DIRECTORY = Path(_str("TEMP_DIRECTORY", str(PROJECT_ROOT / "temp")))
OUTPUT_DIRECTORY = Path(_str("OUTPUT_DIRECTORY", str(PROJECT_ROOT / "output")))
MODELS_DIRECTORY = Path(_str("MODELS_DIRECTORY", str(PROJECT_ROOT / "models")))
LOG_DIRECTORY = Path(_str("LOG_DIRECTORY", str(PROJECT_ROOT / "logs")))

# Keep the vocal-separation weights inside the project (models/) instead of
# the user home directory, which makes Docker volumes predictable.
# Demucs 4.x fetches htdemucs through the HuggingFace hub (HF_HOME) and some
# checkpoints through the torch hub (TORCH_HOME) - cover both.
os.environ.setdefault("TORCH_HOME", str(MODELS_DIRECTORY))
os.environ.setdefault("HF_HOME", str(MODELS_DIRECTORY / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Keep a copy of every finished video in OUTPUT_DIRECTORY (besides sending it).
SAVE_OUTPUT_COPY = _bool("SAVE_OUTPUT_COPY", False)

# --------------------------------------------------------------------------- #
# Job handling
# --------------------------------------------------------------------------- #
MAX_CONCURRENT_JOBS = max(1, _int("MAX_CONCURRENT_JOBS", 1))
MAX_JOBS_PER_USER = max(1, _int("MAX_JOBS_PER_USER", 1))
JOB_TIMEOUT_SECONDS = _int("JOB_TIMEOUT_SECONDS", 3 * 60 * 60)  # 3 hours per step
MIN_FREE_DISK_GB = _float("MIN_FREE_DISK_GB", 5.0)
DELETE_TEMP_AFTER_SEND = _bool("DELETE_TEMP_AFTER_SEND", True)
KEEP_TEMP_ON_ERROR = _bool("KEEP_TEMP_ON_ERROR", False)

PROGRESS_UPDATE_INTERVAL = _float("PROGRESS_UPDATE_INTERVAL", 2.5)

# --------------------------------------------------------------------------- #
# Web interface (browser upload -> process -> download)
# --------------------------------------------------------------------------- #
# Started by ``python -m web.app``; docker-compose.yml does it for the ``web``
# service.  The browser talks HTTP directly, so there is no Telegram size cap
# - only MAX_VIDEO_SIZE above applies.
WEB_HOST = _str("WEB_HOST", "0.0.0.0")
# Render (and similar hosts) injects PORT; WEB_PORT still wins when set.
WEB_PORT = _int("WEB_PORT", _int("PORT", 8080))
# HTTP Basic auth for the whole interface (download links included).
# WEB_PASSWORD empty -> no password prompt (safe only on localhost!).
WEB_USERNAME = _str("WEB_USERNAME", "admin")
WEB_PASSWORD = _str("WEB_PASSWORD")
# How long (seconds) a finished job stays downloadable, then it is removed.
WEB_JOB_TTL_SECONDS = _int("WEB_JOB_TTL_SECONDS", 3600)
# Filename the browser suggests when saving the finished picture ZIP.
WEB_DOWNLOAD_NAME = _str("WEB_DOWNLOAD_NAME", "150-pictures")

# --------------------------------------------------------------------------- #
# Slideshow / video
# --------------------------------------------------------------------------- #
FRAME_COUNT = _int("FRAME_COUNT", 150)
# The source video yields 300 photos which are shown as 150 pairs: photo 1
# (left half) + photo 2 (right half) form pair 1, and so on.
PHOTO_COUNT = FRAME_COUNT * 2  # 300
FRAME_DURATION_SECONDS = _float("FRAME_DURATION_SECONDS", 8.0)
# The video shows 150 stills, so a high frame rate buys nothing visually but
# costs encode time and file size: 10 fps is indistinguishable from 30 fps for
# static pictures and cuts the H.264 frame count (and render time) by ~3x.
SLIDESHOW_FPS = _int("SLIDESHOW_FPS", 10)
SLIDESHOW_DURATION_SECONDS = FRAME_COUNT * FRAME_DURATION_SECONDS  # 1200 s = 20 min

# Output canvas cap.  Default 1280x720 ("720p"): photos stay sharp, and the
# encode is still cheap because the video holds only 150 stills at 10 fps.
# The canvas is always 16:9 (any 16:9 resolution works - 1280x720, 1920x1080,
# ...); every still fills it with two side-by-side halves, each cropped around
# the main face of one photo, so photos are never stretched and only ever
# downscaled from the source.  Use a smaller cap (e.g. 640x360) to stay under
# the 50 MB official Bot API upload.
MAX_OUTPUT_WIDTH = _int("MAX_OUTPUT_WIDTH", 1280)
MAX_OUTPUT_HEIGHT = _int("MAX_OUTPUT_HEIGHT", 720)

SLIDESHOW_PRESET = _str("SLIDESHOW_PRESET", "veryfast")
SLIDESHOW_CRF = _int("SLIDESHOW_CRF", 20)
FINAL_PRESET = _str("FINAL_PRESET", "veryfast")
FINAL_CRF = _int("FINAL_CRF", 20)
JPEG_QUALITY = _int("JPEG_QUALITY", 2)  # ffmpeg -q:v for photo extraction
# Pair composites are written with Pillow at quality 95 / 4:4:4 subsampling:
# they are the input of the slideshow encode, so this is the one lossy image
# step before H.264.
PAIR_JPEG_QUALITY = _int("PAIR_JPEG_QUALITY", 95)

# Frame quality heuristics
BLACK_FRAME_MEAN_THRESHOLD = _float("BLACK_FRAME_MEAN_THRESHOLD", 12.0)
BRIGHT_FRAME_MEAN_THRESHOLD = _float("BRIGHT_FRAME_MEAN_THRESHOLD", 245.0)
FLAT_FRAME_STD_THRESHOLD = _float("FLAT_FRAME_STD_THRESHOLD", 4.0)
DUPLICATE_FRAME_THRESHOLD = _float("DUPLICATE_FRAME_THRESHOLD", 2.5)
FRAME_EXTRACT_WORKERS = max(1, _int("FRAME_EXTRACT_WORKERS", min(8, (os.cpu_count() or 4))))
# fallback positions inside a segment (fraction of the segment) when the
# preferred frame is unusable (black / duplicate / corrupted)
FRAME_CANDIDATE_OFFSETS = (0.50, 0.30, 0.70, 0.15, 0.85, 0.42, 0.62)
MIN_SOURCE_DURATION_SECONDS = _float("MIN_SOURCE_DURATION_SECONDS", 5.0)

# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
AUDIO_SAMPLE_RATE = _int("AUDIO_SAMPLE_RATE", 44100)
AUDIO_CHANNELS = _int("AUDIO_CHANNELS", 2)
AUDIO_BITRATE = _str("AUDIO_BITRATE", "192k")
AUDIO_LOUDNESS_NORM = _bool("AUDIO_LOUDNESS_NORM", True)
AUDIO_LOUDNESS_TARGET = _float("AUDIO_LOUDNESS_TARGET", -16.0)
AUDIO_TRUE_PEAK = _float("AUDIO_TRUE_PEAK", -1.5)
AUDIO_LOUDNESS_RANGE = _float("AUDIO_LOUDNESS_RANGE", 11.0)

# --------------------------------------------------------------------------- #
# Vocal separation
# --------------------------------------------------------------------------- #
MODEL_BACKEND = _str("MODEL_BACKEND", "demucs").lower()
MODEL_NAME = _str("MODEL_NAME", "htdemucs")
DEMUCS_DEVICE = _str("DEMUCS_DEVICE", "auto").lower()  # auto | cpu | cuda
# Segment workers inside the separator (demucs -j).  On CPU the workers run in
# parallel threads, which is the single biggest speed lever: measured ~1.6x
# faster separation with 4 workers vs 1 (benchmark: 41 s vs 69 s of compute for
# a 60 s clip).  0 = auto (half of the logical cores, at most 8; 8 workers is
# what brings a 20-minute separation down to ~5 minutes on a 16-vCPU server);
# 1 = single worker (lowest memory).
DEMUCS_JOBS = max(0, _int("DEMUCS_JOBS", 0))
# Crossfade between the separator's internal segments (demucs --overlap).
# Demucs' own default is 0.25; 0.1 runs the very same model at the very same
# precision but takes ~15% less time (measured 1.45x realtime vs 1.25x on the
# reference laptop).  Raise to 0.25 if segment seams ever become audible.
DEMUCS_OVERLAP = max(0.0, min(0.9, _float("DEMUCS_OVERLAP", 0.1)))
DEMUCS_EXTRA_ARGS = _str("DEMUCS_EXTRA_ARGS")

# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
OVERLAY_DEFAULT_POSITION = _str("OVERLAY_DEFAULT_POSITION", "center")
OVERLAY_MARGIN = _int("OVERLAY_MARGIN", 24)
OVERLAY_SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
OVERLAY_POSITIONS = ("top-left", "top-right", "center", "bottom-left", "bottom-right")
OVERLAY_SIZE_PRESETS: dict[str, float | None] = {
    "original": None,   # keep the original pixel size (down-scaled if too big)
    "25%": 0.25,
    "50%": 0.50,
    "75%": 0.75,
    "100%": 1.00,
    # edge-to-edge special case (handled by OVERLAY_FULL_SCREEN_SIZE in
    # overlay.py): the image covers the whole canvas and its near-black
    # pixels turn transparent, so frame/watermark style overlays let the
    # video show through the middle
    "full screen": None,
}
OVERLAY_FULL_SCREEN_SIZE = "full screen"
OVERLAY_DEFAULT_SIZE = _str("OVERLAY_DEFAULT_SIZE", "original")

# --------------------------------------------------------------------------- #
# Files produced inside each job directory
# --------------------------------------------------------------------------- #
SOURCE_FILENAME = "source_video"
PICTURES_ZIP_FILENAME = _str("PICTURES_ZIP_FILENAME", "150-pictures.zip")
AUDIO_FILENAME = "original_audio.wav"
VOCALS_FILENAME = "vocals.wav"
SLIDESHOW_FILENAME = "slideshow.mp4"
FINAL_FILENAME = "final_video.mp4"
FRAMES_DIRNAME = "frames"

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = _str("LOG_LEVEL", "INFO").upper()
LOG_FILE = _str("LOG_FILE")  # empty -> console only


def ensure_directories() -> None:
    """Create all runtime directories (safe to call multiple times)."""
    for path in (TEMP_DIRECTORY, OUTPUT_DIRECTORY, MODELS_DIRECTORY, LOG_DIRECTORY):
        path.mkdir(parents=True, exist_ok=True)


def validate() -> list[str]:
    """Return a list of human readable configuration problems (empty = ok)."""
    problems: list[str] = []
    if not TELEGRAM_BOT_TOKEN:
        problems.append(
            "TELEGRAM_BOT_TOKEN is not set - copy .env.example to .env and fill it in."
        )
    if FRAME_COUNT != 150:
        problems.append(
            f"FRAME_COUNT is {FRAME_COUNT} (the specification requires exactly 150)."
        )
    if abs(FRAME_DURATION_SECONDS - 8.0) > 1e-6:
        problems.append(
            f"FRAME_DURATION_SECONDS is {FRAME_DURATION_SECONDS} "
            "(the specification requires exactly 8 seconds)."
        )
    if abs(SLIDESHOW_DURATION_SECONDS - 1200.0) > 1e-6:
        problems.append(
            f"Slideshow duration would be {SLIDESHOW_DURATION_SECONDS}s instead of 1200s."
        )
    return problems


def summary() -> str:
    """Multi-line summary used in the startup log."""
    mode = "local Bot API server" if TELEGRAM_LOCAL_MODE else "official Bot API"
    return "\n".join(
        [
            f"  API mode                : {mode}",
            f"  API base URL            : {TELEGRAM_API_BASE_URL or 'https://api.telegram.org'}",
            f"  Max download (effective): {_EFFECTIVE_DOWNLOAD_LIMIT / MB:.0f} MB",
            f"  Max upload   (effective): {_EFFECTIVE_UPLOAD_LIMIT / MB:.0f} MB",
            f"  Photos                  : {PHOTO_COUNT} -> {FRAME_COUNT} pairs x "
            f"{FRAME_DURATION_SECONDS:g}s = {SLIDESHOW_DURATION_SECONDS / 60:g} min",
            f"  Output resolution       : 16:9, up to {MAX_OUTPUT_WIDTH}x{MAX_OUTPUT_HEIGHT} @ "
            f"{SLIDESHOW_FPS} fps",
            f"  Vocal separation        : {MODEL_BACKEND}/{MODEL_NAME} (device={DEMUCS_DEVICE}, "
            f"jobs={DEMUCS_JOBS or 'auto'}, overlap={DEMUCS_OVERLAP:g})",
            f"  Concurrent jobs         : {MAX_CONCURRENT_JOBS} (max {MAX_JOBS_PER_USER} per user)",
            f"  Temp directory          : {TEMP_DIRECTORY}",
            f"  Output directory        : {OUTPUT_DIRECTORY}",
        ]
    )


def effective_download_limit() -> int:
    return _EFFECTIVE_DOWNLOAD_LIMIT


def effective_upload_limit() -> int:
    return _EFFECTIVE_UPLOAD_LIMIT


def web_output_limit() -> int:
    """Output size cap for the web interface (browser download).

    Telegram's 50 MB upload cap only exists because the bot has to push the
    file through the Bot API - a download straight from this server has no
    such limit, so only the operator's ``MAX_OUTPUT_SIZE`` applies (and the
    2000 MB local-server bound).  Without it a 1280x720 render would be
    silently compressed by the renderer's size fallback even though nothing
    stops the browser from saving the larger file.
    """
    return min(MAX_OUTPUT_SIZE or (LOCAL_SERVER_LIMIT - 100 * MB), LOCAL_SERVER_LIMIT)
