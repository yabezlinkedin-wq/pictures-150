"""
Job queue, progress reporting and delivery.

One process hosts one :class:`JobManager`.  Handlers download the incoming
video (that part is pure network I/O and runs concurrently for every user),
then submit a :class:`JobHandle`; a small pool of workers runs the heavy
pipeline one job at a time (``MAX_CONCURRENT_JOBS``) inside a thread pool.

Progress flows the other way: the pipeline thread calls a callback which
schedules status-message edits on the event loop, throttled to
``PROGRESS_UPDATE_INTERVAL`` seconds so Telegram never rate-limits us.  A
message that was deleted by the user is detected and replaced once.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from telegram import Bot, InlineKeyboardMarkup, Message
from telegram.constants import ChatAction
from telegram.error import TelegramError

from bot import file_io, messages
from config import settings
from processing.errors import JobCancelledError, ProcessingError, TelegramIOError
from processing.pipeline import JobSpec, PipelineResult, run_pipeline

log = logging.getLogger(__name__)

_ACTION_INTERVAL = 15.0
_STATUS_RECREATE_LIMIT = 3

# An empty inline keyboard removes the buttons of the status message.
EMPTY_MARKUP = InlineKeyboardMarkup([])


def new_job_id(user_id: int) -> str:
    """``job_<user_id>_<timestamp>_<random>`` - unique, filesystem friendly."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"job_{user_id}_{stamp}_{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------- #
# handle
# --------------------------------------------------------------------------- #
@dataclass
class JobHandle:
    """One user request, from "queued" to "video delivered"."""

    job_id: str
    user_id: int
    chat_id: int
    spec: JobSpec
    status: Message | None = None
    reply_markup: InlineKeyboardMarkup | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    stage: int = 0
    status_alive: bool = True
    recreate_attempts: int = 0

    def __post_init__(self) -> None:
        self.status_alive = self.status is not None

    @property
    def job_dir(self) -> Path:
        return Path(self.spec.work_dir)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - (self.started or self.created)

    @property
    def running(self) -> bool:
        return self.started is not None


# --------------------------------------------------------------------------- #
# progress reporting
# --------------------------------------------------------------------------- #
class ProgressEditor:
    """
    Throttled, thread-safe editor of one job's status message.

    ``set_text`` may be called from any thread (the pipeline thread, HTTPX's
    reading loop, the event loop itself); the actual ``edit_text`` call always
    happens on the event loop and only one edit is in flight at a time.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        bot: Bot | None,
        handle: JobHandle,
        *,
        interval: float | None = None,
    ) -> None:
        self._loop = loop
        self._bot = bot
        self._handle = handle
        self._interval = settings.PROGRESS_UPDATE_INTERVAL if interval is None else interval
        self._lock = threading.Lock()
        self._last_sent = 0.0
        self._last_text = ""
        self._next_text: str | None = None
        self._task: asyncio.Task | None = None

    # -- producer side (any thread) --------------------------------------- #
    def set_text(self, text: str, *, force: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force:
                if now - self._last_sent < self._interval:
                    return
                if text == self._last_text:
                    return
            self._last_sent = now
            self._last_text = text
        self._schedule(text)

    def _schedule(self, text: str) -> None:
        try:
            self._loop.call_soon_threadsafe(self._apply, text)
        except RuntimeError:  # loop already closed (shutdown)
            pass

    # -- consumer side (event loop) --------------------------------------- #
    def _apply(self, text: str) -> None:
        if self._task is not None and not self._task.done():
            self._next_text = text  # coalesce: the newest text wins
            return
        self._task = self._loop.create_task(self._write(text))

    async def flush(self, text: str, *, timeout: float = 8.0) -> None:
        """Send ``text`` immediately and wait until it has been written."""
        self.set_text(text, force=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pending = self._next_text is not None or (self._task is not None and not self._task.done())
            if not pending:
                return
            await asyncio.sleep(0.05)

    async def _write(self, text: str) -> None:
        handle = self._handle
        markup = handle.reply_markup
        try:
            if handle.status is not None and handle.status_alive:
                try:
                    kwargs = {"parse_mode": "HTML"}
                    if markup is not None:
                        kwargs["reply_markup"] = markup
                    await handle.status.edit_text(text, **kwargs)
                    return
                except TelegramError as exc:
                    if "not modified" in str(exc).lower():
                        return
                    log.debug("status edit failed for job %s: %s", handle.job_id, exc)
                    handle.status_alive = False
            if self._bot is not None and handle.recreate_attempts < _STATUS_RECREATE_LIMIT:
                handle.recreate_attempts += 1
                try:
                    kwargs = {"parse_mode": "HTML"}
                    if markup is not None:
                        kwargs["reply_markup"] = markup
                    handle.status = await self._bot.send_message(handle.chat_id, text, **kwargs)
                    handle.status_alive = True
                except TelegramError as exc:
                    log.debug("could not recreate status message for %s: %s", handle.job_id, exc)
        except asyncio.CancelledError:
            raise
        finally:
            nxt, self._next_text = self._next_text, None
            if nxt is not None:
                self._apply(nxt)


class PipelineProgress:
    """``progress_cb`` for :func:`processing.pipeline.run_pipeline`."""

    def __init__(self, editor: ProgressEditor, handle: JobHandle) -> None:
        self._editor = editor
        self._handle = handle
        self._last_stage = -1

    def __call__(self, overall: float, stage: int, stage_fraction: float, detail: str) -> None:
        if self._handle.cancel_event.is_set():
            return  # the pipeline is about to abort; no more edits
        percent = min(100.0, max(0.0, overall * 100.0))
        stage_changed = stage != self._last_stage
        self._last_stage = stage
        self._handle.stage = stage
        detail_line = messages.stage_detail(stage, stage_fraction) or detail
        self._editor.set_text(
            messages.progress_text(percent, stage, detail_line),
            force=stage_changed or percent >= 100.0,
        )


class UploadProgress:
    """``progress_cb`` for :func:`bot.file_io.send_video_file`."""

    def __init__(self, editor: ProgressEditor, total_bytes: int) -> None:
        self._editor = editor
        self._total = max(1, total_bytes)
        self._first = True

    def __call__(self, fraction: float) -> None:
        total_mb = self._total / 1024 / 1024
        detail = f"{fraction * total_mb:.1f} MB / {total_mb:.1f} MB"
        self._editor.set_text(
            messages.transfer_text("⬆️ Uploading the final video...", fraction, detail),
            force=self._first or fraction >= 1.0,
        )
        self._first = False


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #
class JobManager:
    """asyncio queue + worker pool around :func:`run_pipeline`."""

    def __init__(self, *, max_concurrent: int | None = None) -> None:
        self._max = max(1, max_concurrent or settings.MAX_CONCURRENT_JOBS)
        self._queue: asyncio.Queue[JobHandle] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._jobs: dict[str, JobHandle] = {}
        self._job_dirs: dict[str, Path] = {}
        self._by_user: dict[int, set[str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bot: Bot | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._running = 0
        self._stopping = False

    # -- lifecycle --------------------------------------------------------- #
    async def start(self, bot: Bot) -> None:
        self._loop = asyncio.get_running_loop()
        self._bot = bot
        self._executor = ThreadPoolExecutor(max_workers=self._max, thread_name_prefix="pipeline")
        for index in range(self._max):
            self._workers.append(
                asyncio.create_task(self._worker(index), name=f"job-worker-{index}")
            )
        log.info(
            "job manager ready: %d worker(s), max %d job(s) per user",
            self._max,
            settings.MAX_JOBS_PER_USER,
        )

    async def stop(self) -> None:
        self._stopping = True
        for handle in list(self._jobs.values()):
            handle.cancel_event.set()
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        for job_id, path in list(self._job_dirs.items()):
            log.info("shutdown: dropping temp directory of %s (%s)", job_id, path)
            self._drop_job_dir(job_id, keep=settings.KEEP_TEMP_ON_ERROR)
        self._jobs.clear()
        self._by_user.clear()
        log.info("job manager stopped")

    # -- bookkeeping ------------------------------------------------------- #
    def active_count(self, user_id: int) -> int:
        return len(self._by_user.get(user_id, ()))

    def find_for_user(self, user_id: int) -> JobHandle | None:
        ids = self._by_user.get(user_id)
        if not ids:
            return None
        handles = [self._jobs[job_id] for job_id in ids if job_id in self._jobs]
        if not handles:
            return None
        return max(handles, key=lambda handle: handle.created)

    def stats(self) -> tuple[int, int]:
        """``(running, queued)`` - used by /status style diagnostics."""
        return self._running, self._queue.qsize()

    # -- submission -------------------------------------------------------- #
    async def submit(self, handle: JobHandle) -> int:
        """Queue a job; returns how many jobs are ahead of it."""
        if self._loop is None:
            raise RuntimeError("JobManager.start() was never called")
        self._jobs[handle.job_id] = handle
        self._job_dirs[handle.job_id] = handle.job_dir
        self._by_user.setdefault(handle.user_id, set()).add(handle.job_id)
        ahead = self._queue.qsize() + self._running
        if ahead > 0 and handle.status is not None:
            editor = ProgressEditor(self._loop, self._bot, handle)
            await editor.flush(messages.queued_message(ahead + 1, ahead), timeout=4.0)
        await self._queue.put(handle)
        log.info("job %s queued (ahead=%d, user=%s)", handle.job_id, ahead, handle.user_id)
        return ahead

    async def cancel_user(self, user_id: int) -> bool:
        """Ask every job of ``user_id`` to stop (kills running subprocesses)."""
        ids = list(self._by_user.get(user_id, ()))
        if not ids:
            return False
        for job_id in ids:
            handle = self._jobs.get(job_id)
            if handle is None:
                continue
            log.info("cancelling job %s on user request", job_id)
            handle.cancel_event.set()
        return True

    def _finish(self, handle: JobHandle) -> None:
        self._jobs.pop(handle.job_id, None)
        ids = self._by_user.get(handle.user_id)
        if ids is not None:
            ids.discard(handle.job_id)
            if not ids:
                self._by_user.pop(handle.user_id, None)

    # -- worker ------------------------------------------------------------ #
    async def _worker(self, index: int) -> None:
        while True:
            handle = await self._queue.get()
            try:
                if self._stopping:
                    continue
                await self._process(handle)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a worker must never die
                log.exception("worker %d: unexpected failure for job %s", index, handle.job_id)
            finally:
                self._queue.task_done()

    async def _process(self, handle: JobHandle) -> None:
        assert self._loop is not None and self._executor is not None
        self._running += 1
        handle.started = time.monotonic()
        editor = ProgressEditor(self._loop, self._bot, handle)
        progress = PipelineProgress(editor, handle)
        action_task = asyncio.create_task(self._chat_action_loop(handle))

        result: PipelineResult | None = None
        failure_text: str | None = None
        keep_temp = False
        try:
            if handle.cancel_event.is_set():
                raise JobCancelledError("cancelled before it started")
            await editor.flush(messages.status_line(0), timeout=4.0)
            result = await self._loop.run_in_executor(
                self._executor,
                functools.partial(
                    run_pipeline,
                    handle.spec,
                    progress_cb=progress,
                    cancel_event=handle.cancel_event,
                ),
            )
        except JobCancelledError:
            log.info("job %s cancelled", handle.job_id)
            failure_text = messages.CANCELLED
        except ProcessingError as exc:
            log.warning("job %s failed: %s", handle.job_id, exc)
            failure_text = messages.friendly_error(exc.user_message)
            keep_temp = settings.KEEP_TEMP_ON_ERROR
        except Exception:  # noqa: BLE001 - never leak a traceback to the user
            log.exception("job %s crashed", handle.job_id)
            failure_text = messages.friendly_error("")
            keep_temp = settings.KEEP_TEMP_ON_ERROR
        finally:
            self._running -= 1
            self._finish(handle)
            action_task.cancel()
            await asyncio.gather(action_task, return_exceptions=True)

        if result is None:
            await editor.flush(failure_text or messages.friendly_error(""))
            await asyncio.to_thread(self._cleanup_job_dir, handle, keep=keep_temp)
            return

        # ---- deliver ---------------------------------------------------- #
        log.info(
            "job %s processed in %.1fs (%s, %.1f MB) - uploading",
            handle.job_id,
            time.monotonic() - handle.started,
            messages.format_clock(result.duration),
            result.size_bytes / 1024 / 1024,
        )
        try:
            await file_io.send_video_file(
                self._bot,  # type: ignore[arg-type]
                handle.chat_id,
                result.final_path,
                caption=messages.READY_CAPTION,
                duration=result.duration,
                width=result.video_info.display_width,
                height=result.video_info.display_height,
                progress_cb=UploadProgress(editor, result.size_bytes),
            )
        except TelegramIOError as exc:
            log.error("job %s upload failed: %s", handle.job_id, exc)
            await editor.flush(messages.friendly_error(exc.user_message))
            await asyncio.to_thread(self._cleanup_job_dir, handle, keep=settings.KEEP_TEMP_ON_ERROR)
            return
        except Exception:  # noqa: BLE001
            log.exception("job %s upload crashed", handle.job_id)
            await editor.flush(messages.friendly_error(""))
            await asyncio.to_thread(self._cleanup_job_dir, handle, keep=settings.KEEP_TEMP_ON_ERROR)
            return

        await editor.flush(
            messages.final_summary(
                duration=result.duration,
                size_bytes=result.size_bytes,
                frames_line=messages.frames_line(result.frames_report),
                timing_line=messages.timing_line(result.stage_seconds),
            )
        )
        await asyncio.to_thread(self._cleanup_job_dir, handle)
        log.info(
            "job %s delivered (total %.1fs)",
            handle.job_id,
            time.monotonic() - handle.started,
        )

    async def _chat_action_loop(self, handle: JobHandle) -> None:
        """Keep the "sending video" indicator alive while the job runs."""
        try:
            while True:
                try:
                    await self._bot.send_chat_action(  # type: ignore[union-attr]
                        chat_id=handle.chat_id, action=ChatAction.UPLOAD_VIDEO
                    )
                except TelegramError as exc:
                    log.debug("chat action failed: %s", exc)
                await asyncio.sleep(_ACTION_INTERVAL)
        except asyncio.CancelledError:
            pass

    # -- cleanup ----------------------------------------------------------- #
    def _cleanup_job_dir(self, handle: JobHandle, *, keep: bool = False) -> None:
        """Remove the job's working directory (idempotent, best effort)."""
        if handle.cancel_event.is_set() and settings.KEEP_TEMP_ON_ERROR:
            keep = True
        self._drop_job_dir(handle.job_id, keep=keep)

    def _drop_job_dir(self, job_id: str, *, keep: bool = False) -> None:
        path = self._job_dirs.pop(job_id, None)
        if path is None:
            return  # already handled
        if keep or not settings.DELETE_TEMP_AFTER_SEND:
            log.info("keeping temp directory %s", path)
            return
        try:
            shutil.rmtree(path, ignore_errors=True)
            log.debug("removed temp directory %s", path)
        except OSError as exc:  # noqa: BLE001 - cleanup must never fail the job
            log.debug("could not remove %s: %s", path, exc)


# Module-level singleton shared by the handlers and the application.
job_manager = JobManager()
