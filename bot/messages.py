"""
User-facing text.

Every string the bot sends lives here so wording stays consistent and easy to
translate.  Progress rendering follows the specification:

    Step 1/5 — Extracting photos...
    [████████░░] 80%
"""

from __future__ import annotations

from config import settings

# --------------------------------------------------------------------------- #
# Static texts
# --------------------------------------------------------------------------- #
WELCOME = (
    "👋 Welcome!\n\n"
    "Send me a video and I will turn it into a 20-minute video that shows a "
    "pair of photos every 8 seconds and plays only the voice extracted from "
    "the original audio track.\n\n"
    "Please upload your video."
)

HELP = (
    "<b>What I do</b>\n"
    "1. Pick 300 photos spread across your video (two for every 8 seconds of "
    "the final movie).\n"
    "2. Arrange them into 150 side-by-side pairs and build a 20:00 slideshow "
    "from them (8 seconds per pair).\n"
    "3. Separate the voice from the music and keep only the voice.\n"
    "4. Optionally place your overlay image on top.\n"
    "5. Render the final MP4 and send it back.\n\n"
    "<b>How to use me</b>\n"
    "• Send a video (or a video file / a photo as overlay).\n"
    "• Limits: up to {max_mb:.0f} MB.\n"
    "• /start – start over\n"
    "• /cancel – cancel the current job\n"
    "• /skip – skip the overlay step\n"
    "• /status – show the current job state\n"
    "• /help – this message"
)

ASK_VIDEO = "Please upload your video."

ASK_OVERLAY = (
    "Do you want to add an overlay image?\n"
    "Send me a PNG / JPG / WebP image, or press <b>Skip</b>."
)

WORKING_HINT = "Got it — starting to process your video."

# --------------------------------------------------------------------------- #
# Stage titles
# --------------------------------------------------------------------------- #
STAGE_TITLES = {
    0: "Preparing...",
    1: "Extracting photos...",
    2: "Creating 20-minute slideshow...",
    3: "Separating vocals from music...",
    4: "Adding overlay...",
    5: "Rendering final video...",
}

STEP_WORDS = {
    1: "Extracting photos",
    2: "Creating 20-minute slideshow",
    3: "Separating vocals from music",
    4: "Adding overlay",
    5: "Rendering final video",
}

STAGE_COUNT = 5

# --------------------------------------------------------------------------- #
# Final texts
# --------------------------------------------------------------------------- #
PROCESSING_COMPLETE = "Processing complete."
READY_CAPTION = "Your 20-minute voice-only video is ready."


# --------------------------------------------------------------------------- #
# Progress rendering
# --------------------------------------------------------------------------- #
def progress_bar(fraction: float, width: int = 10) -> str:
    """``0.8`` -> ``[████████░░] 80%``"""
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    filled = min(width, max(0, filled))
    if fraction > 0 and filled == 0:
        filled = 1
    if fraction < 1 and filled == width:
        filled = width - 1
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fraction * 100:.0f}%"


def progress_text(percent: float, stage: int, detail: str = "") -> str:
    """Build the full progress message for one edit."""
    text = f"<b>{status_line(stage)}</b>\n{progress_bar(percent / 100.0)}"
    if detail:
        text += f"\n\n{detail}"
    return text


def status_line(stage: int) -> str:
    if stage in STEP_WORDS:
        return f"Step {stage}/{STAGE_COUNT} — {STEP_WORDS[stage]}..."
    return STAGE_TITLES.get(stage, STAGE_TITLES[0])


def stage_detail(stage: int, fraction: float) -> str:
    """Numeric sub-line for a stage, e.g. ``extracted 42 of 300 photos``."""
    fraction = min(1.0, max(0.0, fraction))
    if stage == 1:
        done = int(round(fraction * settings.PHOTO_COUNT))
        return f"extracted {done} of {settings.PHOTO_COUNT} photos"
    if stage == 2:
        total = settings.SLIDESHOW_DURATION_SECONDS
        return f"built {_hhmmss(fraction * total)} of {_hhmmss(total)}"
    if stage == 3:
        return f"analysed {fraction * 100:.0f}% of the audio track"
    if stage == 5:
        total = settings.SLIDESHOW_DURATION_SECONDS
        return f"encoded {_hhmmss(fraction * total)} of {_hhmmss(total)}"
    return ""


def transfer_text(label: str, fraction: float, detail: str = "") -> str:
    """Progress text for a download/upload (not part of the 5 processing steps)."""
    text = f"<b>{label}</b>\n{progress_bar(fraction)}"
    if detail:
        text += f"\n\n{detail}"
    return text


def _hhmmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


# public alias (same formatting, used outside this module)
format_clock = _hhmmss


STAGE_TIMING_LABELS = {1: "photos", 2: "slideshow", 3: "vocals", 4: "overlay", 5: "render"}


def final_summary(*, duration: float, size_bytes: int, frames_line: str = "", timing_line: str = "") -> str:
    lines = [
        "🎬 <b>Processing complete.</b>",
        f"Duration {_hhmmss(duration)} • Size {size_bytes / 1024 / 1024:.1f} MB",
    ]
    if frames_line:
        lines.append(frames_line)
    if timing_line:
        lines.append(timing_line)
    return "\n".join(lines)


def frames_line(report) -> str:  # noqa: ANN001 - FrameQualityReport (avoids an import cycle)
    if report is None:
        return ""
    retried = f" • {report.segments_retried} segments retried" if report.segments_retried else ""
    return (
        f"🎞️ {report.total} photos • {report.extractions} candidates checked{retried}"
    )


def timing_line(stage_seconds: dict) -> str:
    if not stage_seconds:
        return ""
    parts = [
        f"{STAGE_TIMING_LABELS.get(stage, str(stage))} {seconds:.0f}s"
        for stage, seconds in sorted(stage_seconds.items())
        if STAGE_TIMING_LABELS.get(stage)
    ]
    return "⏱️ " + " • ".join(parts) if parts else ""


def video_received_text(info) -> str:  # noqa: ANN001 - VideoInfo (avoids an import cycle)
    audio = info.audio_codec or "unknown"
    if info.audio_sample_rate:
        audio += f" {info.audio_sample_rate / 1000:g} kHz"
    if info.audio_channels:
        audio += f" • {'mono' if info.audio_channels == 1 else f'{info.audio_channels} ch'}"
    return "\n".join(
        [
            "📹 <b>Video received</b>",
            f"Duration: {_hhmmss(info.duration)}",
            f"Resolution: {info.resolution_label} @ {info.fps:.1f} fps",
            f"Audio: {audio}",
            "",
            ASK_OVERLAY,
        ]
    )


def status_report(job_id: str, *, running: bool, elapsed: float = 0.0, stage: int = 0) -> str:
    if running:
        return (
            f"⏳ Job <code>{job_id}</code> is running.\n"
            f"{status_line(stage)}\nElapsed: {_hhmmss(elapsed)}"
        )
    return f"⏳ Job <code>{job_id}</code> is queued and will start as soon as a slot is free."


# --------------------------------------------------------------------------- #
# Queue / state texts
# --------------------------------------------------------------------------- #
def queued_message(position: int, ahead: int | None = None) -> str:
    if ahead is None or ahead <= 0:
        return "⏳ Your job is queued and will start in a moment."
    noun = "job" if ahead == 1 else "jobs"
    return f"⏳ You are #{position} in the queue ({ahead} {noun} ahead of you)."


def busy_message(active: int, limit: int) -> str:
    return (
        f"The bot is currently processing {active} video(s). "
        "Your job will start as soon as a slot is free."
    )


CANCELLED = "Processing was cancelled. Send a video whenever you are ready."
NOTHING_TO_CANCEL = "There is nothing to cancel right now. Send a video whenever you are ready."
CANCELLING = "⏹️ Cancelling — this can take a few seconds..."
NOTHING_TO_SKIP = "There is nothing to skip right now. Send a video whenever you are ready."

REUSED_VIDEO = "I already have this video — continuing with it."

OVERLAY_SAVED = "🖼️ Overlay saved. Choose a position:"
OVERLAY_SIZE_PROMPT = "🖼️ Choose the overlay size:"
OVERLAY_SKIPPED = "No overlay — continuing without one."
OVERLAY_NEEDS_VIDEO = (
    "Send me the video first — after that I can place an overlay image on top of it."
)
OVERLAY_INVALID = (
    "That file is not a usable image. Please send a PNG, JPG or WebP image."
)
STALE_BUTTON = "This choice is no longer active."
NO_ACTIVE_JOB = "You have no active job right now. Send a video whenever you are ready."
OVERLAY_POSITION_SAVED = "🖼️ Position saved."
USER_BUSY = (
    "You already have a video being processed. "
    "Use /cancel to stop it, or wait until the current one is finished."
)

SIZE_LABELS = {
    "original": "Original size",
    "25%": "25%",
    "50%": "50%",
    "75%": "75%",
    "100%": "100% (full width)",
    "full screen": "⛶ Full screen — covers the video, black turns transparent",
}

POSITION_LABELS = {
    "top-left": "↖️ Top left",
    "top-right": "↗️ Top right",
    "center": "⏺️ Center",
    "bottom-left": "↙️ Bottom left",
    "bottom-right": "↘️ Bottom right",
}

HELP_SHORT = (
    "Send me a video and I will build a 20-minute slideshow from it with "
    "voice-only audio. Use /help for details, /cancel to stop, /skip to skip "
    "the overlay step, /status to see the current job."
)

ERROR_FOOTER = "\n\nIf this keeps happening, try another video or try again later."


def friendly_error(message: str, *, footer: bool = True) -> str:
    """Wrap a user-safe message, making sure the tone stays friendly.

    Pass ``footer=False`` for rejections where retrying cannot help (e.g. a
    hard size limit) so the generic "...try again later" hint is suppressed.
    """
    text = (message or "").strip()
    if not text:
        text = "Something went wrong while processing your video. Please try again."
    if not text.endswith(("!", ".", "?")):
        text += "."
    if (
        not footer
        or "try again" in text.lower()
        or "try another" in text.lower()
        or "later" in text.lower()
    ):
        return f"⚠️ {text}"
    return f"⚠️ {text}{ERROR_FOOTER}"
