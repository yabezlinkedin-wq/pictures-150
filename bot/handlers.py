"""
Telegram command / message / callback handlers.

Per-user state machine::

    idle ──video──> awaiting overlay ──image──> choose position ──> choose size ─┐
                        │                                                       │
                        └── /skip  or  "Skip" button  or  no image ────────────┴──> processing

The whole conversation for one job happens in a single status message: it
starts as a download progress bar and ends as the "Processing complete"
summary - with the final video attached below it.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from bot import messages
from bot.file_io import download_file
from bot.jobs import (
    EMPTY_MARKUP,
    JobHandle,
    ProgressEditor,
    job_manager,
    new_job_id,
)
from config import settings
from processing import overlay as overlay_module
from processing import video_info
from processing.errors import NoAudioStreamError, ProcessingError, TelegramIOError
from processing.pipeline import JobSpec

log = logging.getLogger(__name__)


class State:
    IDLE = "idle"
    AWAIT_OVERLAY = "await_overlay"
    AWAIT_POSITION = "await_position"
    AWAIT_SIZE = "await_size"
    PROCESSING = "processing"


@dataclass
class Session:
    """Everything the bot remembers about one user between messages."""

    state: str = State.IDLE
    handle: JobHandle | None = None
    video_file_unique_id: str | None = None
    overlay_path: Path | None = None
    overlay_position: str | None = None
    overlay_size_label: str | None = None

    @property
    def job_dir(self) -> Path | None:
        return self.handle.job_dir if self.handle is not None else None

    @property
    def status(self) -> Message | None:
        return self.handle.status if self.handle is not None else None


_sessions: dict[int, Session] = {}

_VIDEO_SUFFIXES = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg",
    ".ts", ".flv", ".wmv", ".3gp", ".ogv",
}
_MIME_SUFFIXES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
    "video/x-msvideo": ".avi",
    "video/mpeg": ".mpg",
    "video/3gpp": ".3gp",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


def _session_for(user_id: int) -> Session:
    session = _sessions.get(user_id)
    if session is None:
        session = Session()
        _sessions[user_id] = session
    return session


async def _remove_dir(path: Path | None) -> None:
    if path is None:
        return
    await asyncio.to_thread(shutil.rmtree, path, True)


async def _discard_session(session: Session) -> None:
    """Forget a pending (not yet submitted) job and delete its files."""
    if session.state == State.PROCESSING:
        return
    if not settings.KEEP_TEMP_ON_ERROR:
        await _remove_dir(session.job_dir)
    session.handle = None
    session.video_file_unique_id = None
    session.overlay_path = None
    session.overlay_position = None
    session.overlay_size_label = None
    session.state = State.IDLE


def _suffix_for(file_name: str | None, mime_type: str | None, fallback: str) -> str:
    suffix = Path(file_name or "").suffix.lower()
    if suffix in _VIDEO_SUFFIXES or suffix in settings.OVERLAY_SUPPORTED_EXTENSIONS:
        return suffix
    return _MIME_SUFFIXES.get((mime_type or "").lower(), fallback)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    session = _session_for(user.id)
    if session.state == State.PROCESSING:
        await message.reply_text(messages.USER_BUSY, parse_mode="HTML")
        return
    await _discard_session(session)
    await message.reply_text(messages.WELCOME, parse_mode="HTML")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    limit_mb = settings.effective_download_limit() / (1024 * 1024)
    await update.effective_message.reply_text(
        messages.HELP.format(max_mb=limit_mb), parse_mode="HTML"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    session = _session_for(user.id)
    handle = session.handle
    if handle is None:
        live = job_manager.find_for_user(user.id)
        if live is None:
            await update.effective_message.reply_text(messages.NO_ACTIVE_JOB)
            return
        handle = live
    await update.effective_message.reply_text(
        messages.status_report(
            handle.job_id,
            running=handle.running,
            elapsed=handle.elapsed,
            stage=handle.stage,
        ),
        parse_mode="HTML",
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    session = _session_for(user.id)
    if session.state == State.PROCESSING:
        if await job_manager.cancel_user(user.id):
            await update.effective_message.reply_text(messages.CANCELLING)
        else:
            await update.effective_message.reply_text(messages.NOTHING_TO_CANCEL)
        return
    if session.state in (State.AWAIT_OVERLAY, State.AWAIT_POSITION, State.AWAIT_SIZE):
        await _discard_session(session)
        await update.effective_message.reply_text(messages.CANCELLED)
        return
    await update.effective_message.reply_text(messages.NOTHING_TO_CANCEL)


async def cmd_skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    session = _sessions.get(user.id)
    if session is None or session.state not in (
        State.AWAIT_OVERLAY,
        State.AWAIT_POSITION,
        State.AWAIT_SIZE,
    ):
        await update.effective_message.reply_text(messages.NOTHING_TO_SKIP)
        return
    session.overlay_path = None
    session.overlay_position = None
    session.overlay_size_label = None
    handle = session.handle
    if handle is not None:
        handle.spec.overlay_path = None
        handle.spec.overlay_position = None
        handle.spec.overlay_size_label = None
    await _begin_processing(session, note=messages.OVERLAY_SKIPPED)


# --------------------------------------------------------------------------- #
# incoming media
# --------------------------------------------------------------------------- #
async def on_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A video (or a video document / animation) arrived."""
    message = update.effective_message
    user = update.effective_user
    session = _session_for(user.id)

    media = message.video or message.animation
    document = None
    if media is None and message.document is not None:
        mime = (message.document.mime_type or "").lower()
        if mime.startswith("video/") or _suffix_for(
            message.document.file_name, mime, ""
        ) in _VIDEO_SUFFIXES:
            document = message.document
    if media is None and document is None:
        await message.reply_text(messages.HELP_SHORT)
        return

    if session.state == State.PROCESSING or job_manager.active_count(user.id) >= settings.MAX_JOBS_PER_USER:
        await message.reply_text(messages.USER_BUSY, parse_mode="HTML")
        return

    file_id = (media or document).file_id
    file_unique_id = getattr(media or document, "file_unique_id", None)
    file_size = getattr(media or document, "file_size", None)
    limit = settings.effective_download_limit()
    if file_size and file_size > limit:
        text = (
            f"This video is {file_size / 1024 / 1024:.0f} MB, which is above the "
            f"current limit of {limit / 1024 / 1024:.0f} MB."
        )
        if not settings.TELEGRAM_LOCAL_MODE:
            # Telegram hard-caps bot downloads at 20 MB on the official API;
            # MAX_VIDEO_SIZE only takes full effect with the local Bot API
            # server, and retrying cannot help - explain instead of hinting.
            text += (
                " This limit comes from Telegram's official Bot API and cannot be "
                "raised in the bot itself - it only lifts when the bot runs with "
                "the bundled local Bot API server (docker compose)."
            )
        await message.reply_text(
            messages.friendly_error(text, footer=False),
            parse_mode="HTML",
        )
        return

    # Same video sent again while we are waiting for the overlay: keep the
    # already-downloaded file instead of downloading it a second time.
    if (
        session.state in (State.AWAIT_OVERLAY, State.AWAIT_POSITION, State.AWAIT_SIZE)
        and file_unique_id is not None
        and file_unique_id == session.video_file_unique_id
        and session.job_dir is not None
    ):
        await message.reply_text(messages.REUSED_VIDEO)
        await _ask_overlay(context, session)
        return

    # A different video replaces the previous pending one.
    await _discard_session(session)

    suffix = _suffix_for(getattr(media or document, "file_name", None), getattr(media or document, "mime_type", None), ".mp4")
    job_id = new_job_id(user.id)
    job_dir = settings.TEMP_DIRECTORY / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"{settings.SOURCE_FILENAME}{suffix}"

    spec = JobSpec(
        job_id=job_id,
        user_id=user.id,
        chat_id=message.chat_id,
        source_path=video_path,
        work_dir=job_dir,
    )
    handle = JobHandle(
        job_id=job_id,
        user_id=user.id,
        chat_id=message.chat_id,
        spec=spec,
    )
    session.handle = handle
    session.state = State.AWAIT_OVERLAY
    session.video_file_unique_id = file_unique_id

    status = await message.reply_text(
        messages.transfer_text("⬇️ Downloading video...", 0.0),
        parse_mode="HTML",
    )
    handle.status = status
    handle.status_alive = True
    editor = ProgressEditor(asyncio.get_running_loop(), context.bot, handle)

    first = True

    def on_progress(fraction: float) -> None:
        nonlocal first
        detail = ""
        if file_size:
            detail = f"{fraction * file_size / 1024 / 1024:.1f} MB / {file_size / 1024 / 1024:.1f} MB"
        editor.set_text(
            messages.transfer_text("⬇️ Downloading video...", fraction, detail),
            force=first or fraction >= 1.0,
        )
        first = False

    try:
        await download_file(
            context.bot,
            file_id,
            video_path,
            expected_size=file_size,
            progress_cb=on_progress,
        )
    except TelegramIOError as exc:
        log.warning("download failed for user %s: %s", user.id, exc)
        await editor.flush(messages.friendly_error(exc.user_message))
        await _discard_session(session)
        return
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        log.exception("unexpected download failure for user %s", user.id)
        await editor.flush(messages.friendly_error(""))
        await _discard_session(session)
        return

    # -- validate before asking anything else ----------------------------- #
    try:
        info = await asyncio.to_thread(video_info.probe_video, video_path)
        if not info.has_audio:
            raise NoAudioStreamError(f"{video_path.name} has no audio stream")
    except ProcessingError as exc:
        log.warning("video rejected for user %s: %s", user.id, exc)
        await editor.flush(messages.friendly_error(exc.user_message))
        await _discard_session(session)
        return

    handle.spec.overlay_path = None
    session.state = State.AWAIT_OVERLAY
    await _ask_overlay(context, session, info=info)


async def _ask_overlay(
    context: ContextTypes.DEFAULT_TYPE,
    session: Session,
    *,
    info: video_info.VideoInfo | None = None,
) -> None:
    if info is None:
        try:
            info = await asyncio.to_thread(video_info.probe_video, session.handle.spec.source_path)
        except ProcessingError:  # best effort - the pipeline probes again anyway
            info = None
    text = messages.video_received_text(info) if info is not None else messages.ASK_OVERLAY
    session.handle.reply_markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Skip overlay", callback_data="ov:skip")]]
    )
    editor = ProgressEditor(asyncio.get_running_loop(), context.bot, session.handle)
    await editor.flush(text, timeout=6.0)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A photo - only useful as an overlay image."""
    await _handle_overlay_candidate(update, context, suffix=".jpg", file_id=update.effective_message.photo[-1].file_id, file_size=update.effective_message.photo[-1].file_size)


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.effective_message.document
    mime = (document.mime_type or "").lower()
    if mime.startswith("video/") or _suffix_for(document.file_name, mime, "") in _VIDEO_SUFFIXES:
        await on_video(update, context)
        return
    suffix = _suffix_for(document.file_name, mime, "")
    if mime.startswith("image/") or suffix in settings.OVERLAY_SUPPORTED_EXTENSIONS:
        await _handle_overlay_candidate(
            update, context, suffix=suffix, file_id=document.file_id, file_size=document.file_size
        )
        return
    await update.effective_message.reply_text(messages.ASK_VIDEO)


async def _handle_overlay_candidate(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    suffix: str,
    file_id: str,
    file_size: int | None,
) -> None:
    message = update.effective_message
    user = update.effective_user
    session = _sessions.get(user.id)

    if session is None or session.state not in (
        State.AWAIT_OVERLAY,
        State.AWAIT_POSITION,
        State.AWAIT_SIZE,
    ):
        await message.reply_text(messages.OVERLAY_NEEDS_VIDEO, parse_mode="HTML")
        return
    if suffix not in settings.OVERLAY_SUPPORTED_EXTENSIONS:
        await message.reply_text(messages.OVERLAY_INVALID, parse_mode="HTML")
        return
    if file_size and file_size > settings.effective_download_limit():
        await message.reply_text(
            messages.friendly_error("That image is too large to be used as an overlay."),
            parse_mode="HTML",
        )
        return

    handle = session.handle
    job_dir = handle.job_dir
    overlay_path = job_dir / f"overlay{suffix}"
    editor = ProgressEditor(asyncio.get_running_loop(), context.bot, handle)
    editor.set_text(messages.transfer_text("⬇️ Downloading the overlay image...", 0.0), force=True)
    try:
        await download_file(
            context.bot,
            file_id,
            overlay_path,
            expected_size=file_size,
            progress_cb=lambda fraction: editor.set_text(
                messages.transfer_text("⬇️ Downloading the overlay image...", fraction),
                force=fraction >= 1.0,
            ),
        )
        await asyncio.to_thread(overlay_module.validate_overlay, overlay_path)
    except TelegramIOError as exc:
        await editor.flush(messages.friendly_error(exc.user_message))
        return
    except ProcessingError as exc:  # OverlayError
        log.info("overlay rejected for user %s: %s", user.id, exc)
        await editor.flush(messages.friendly_error(exc.user_message))
        return
    except Exception:  # noqa: BLE001
        log.exception("overlay download failed for user %s", user.id)
        await editor.flush(messages.friendly_error(""))
        return

    if session.overlay_path and session.overlay_path != overlay_path and session.overlay_path.exists():
        try:
            session.overlay_path.unlink()
        except OSError:
            pass
    session.overlay_path = overlay_path
    session.overlay_position = None
    session.overlay_size_label = None
    handle.spec.overlay_path = overlay_path
    session.state = State.AWAIT_POSITION
    handle.reply_markup = _position_keyboard()
    await editor.flush(messages.OVERLAY_SAVED, timeout=6.0)


def _position_keyboard() -> InlineKeyboardMarkup:
    positions = list(settings.OVERLAY_POSITIONS)
    rows: list[list[InlineKeyboardButton]] = []
    for index in range(0, len(positions), 2):
        rows.append(
            [
                InlineKeyboardButton(
                    messages.POSITION_LABELS[position], callback_data=f"ov:pos:{position}"
                )
                for position in positions[index : index + 2]
            ]
        )
    return InlineKeyboardMarkup(rows)


def _size_keyboard() -> InlineKeyboardMarkup:
    labels = list(settings.OVERLAY_SIZE_PRESETS)
    rows: list[list[InlineKeyboardButton]] = []
    for index in range(0, len(labels), 3):
        rows.append(
            [
                InlineKeyboardButton(
                    messages.SIZE_LABELS.get(label, label), callback_data=f"ov:size:{label}"
                )
                for label in labels[index : index + 3]
            ]
        )
    return InlineKeyboardMarkup(rows)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query: CallbackQuery = update.callback_query
    data = query.data or ""
    user = query.from_user
    session = _sessions.get(user.id)

    if not data.startswith("ov:") or session is None or session.state not in (
        State.AWAIT_OVERLAY,
        State.AWAIT_POSITION,
        State.AWAIT_SIZE,
    ):
        await query.answer(messages.STALE_BUTTON, show_alert=False)
        return

    await query.answer()
    handle = session.handle

    if data == "ov:skip":
        session.overlay_path = None
        session.overlay_position = None
        session.overlay_size_label = None
        handle.spec.overlay_path = None
        await _begin_processing(session, note=messages.OVERLAY_SKIPPED)
        return

    if data.startswith("ov:pos:"):
        position = data[len("ov:pos:") :]
        if position not in settings.OVERLAY_POSITIONS:
            await query.answer(messages.STALE_BUTTON, show_alert=False)
            return
        session.overlay_position = position
        handle.spec.overlay_position = position
        session.state = State.AWAIT_SIZE
        handle.reply_markup = _size_keyboard()
        editor = ProgressEditor(asyncio.get_running_loop(), context.bot, handle)
        await editor.flush(
            f"{messages.OVERLAY_POSITION_SAVED}\n{messages.OVERLAY_SIZE_PROMPT}", timeout=6.0
        )
        return

    if data.startswith("ov:size:"):
        label = data[len("ov:size:") :]
        if label not in settings.OVERLAY_SIZE_PRESETS:
            await query.answer(messages.STALE_BUTTON, show_alert=False)
            return
        session.overlay_size_label = label
        handle.spec.overlay_size_label = label
        await _begin_processing(session)
        return

    await query.answer(messages.STALE_BUTTON, show_alert=False)


# --------------------------------------------------------------------------- #
# job submission
# --------------------------------------------------------------------------- #
async def _begin_processing(session: Session, *, note: str | None = None) -> None:
    handle = session.handle
    if handle is None:
        session.state = State.IDLE
        return
    session.state = State.PROCESSING
    handle.reply_markup = EMPTY_MARKUP  # drop the inline keyboards
    if note:
        editor = ProgressEditor(asyncio.get_running_loop(), None, handle)
        editor.set_text(note, force=True)
    try:
        ahead = await job_manager.submit(handle)
    except Exception:  # noqa: BLE001
        log.exception("could not submit job %s", handle.job_id)
        session.state = State.IDLE
        editor = ProgressEditor(asyncio.get_running_loop(), None, handle)
        await editor.flush(messages.friendly_error(""))
        return
    log.info(
        "user %s started job %s (ahead=%d, overlay=%s/%s)",
        handle.user_id,
        handle.job_id,
        ahead,
        handle.spec.overlay_position or "-",
        handle.spec.overlay_size_label or "-",
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    session = _sessions.get(user.id)
    if session is not None and session.state == State.PROCESSING:
        await message.reply_text(messages.USER_BUSY, parse_mode="HTML")
        return
    if session is not None and session.state in (State.AWAIT_OVERLAY, State.AWAIT_POSITION, State.AWAIT_SIZE):
        await message.reply_text(messages.ASK_OVERLAY, parse_mode="HTML")
        return
    await message.reply_text(messages.HELP_SHORT)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last-resort handler: log the traceback, tell the user something friendly."""
    log.error("unhandled exception while processing an update", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message is not None:
        try:
            await update.effective_message.reply_text(
                messages.friendly_error("Something went wrong on my side."), parse_mode="HTML"
            )
        except TelegramError:
            pass
