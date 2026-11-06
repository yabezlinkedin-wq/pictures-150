"""
Job registry and background workers for the web interface.

The processing itself lives in :mod:`processing.image_export`
(:func:`run_pictures_pipeline`): it turns the uploaded video into 150
side-by-side pictures and packs them into one ZIP.  The function is
synchronous and Telegram-free, so this module only adds what a browser needs:

* an in-memory registry carrying progress information (the UI polls it),
* a small worker pool (``MAX_CONCURRENT_JOBS``), one pipeline per worker,
* cancellation through the same ``threading.Event`` the bot uses,
* automatic removal of finished jobs after ``WEB_JOB_TTL_SECONDS``.

Everything in here stays independent of the Telegram layer (``bot/``): the two
front ends share the pipeline, not each other.
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
from typing import Any

from config import settings
from processing.errors import JobCancelledError, ProcessingError
from processing.image_export import PicturesJobSpec, PicturesResult, run_pictures_pipeline

log = logging.getLogger(__name__)

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"
_FINAL_STATES = (STATE_DONE, STATE_ERROR, STATE_CANCELLED)

STAGE_TITLES = {
    0: "Preparing...",
    1: "Selecting photos...",
    2: "Building 150 pictures...",
    3: "Zipping the pictures...",
}

_REAPER_INTERVAL = 300.0  # seconds between cleanup passes


def new_job_id() -> str:
    """``web_<timestamp>_<random>`` - unique and filesystem friendly."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"web_{stamp}_{uuid.uuid4().hex[:6]}"


def job_dir_for(job_id: str) -> Path:
    """Working directory of a job (below the configured temp directory)."""
    return settings.TEMP_DIRECTORY / "web" / job_id


def stage_detail(stage: int, fraction: float) -> str:
    """Numeric sub-line for a stage, e.g. ``extracted 42 of 300 photos``."""
    fraction = min(1.0, max(0.0, fraction))
    if stage == 1:
        done = int(round(fraction * settings.PHOTO_COUNT))
        return f"extracted {done} of {settings.PHOTO_COUNT} photos"
    if stage == 2:
        done = int(round(fraction * settings.FRAME_COUNT))
        return f"built {done} of {settings.FRAME_COUNT} pictures"
    if stage == 3:
        done = int(round(fraction * settings.FRAME_COUNT))
        return f"packed {done} of {settings.FRAME_COUNT} pictures"
    return ""


# --------------------------------------------------------------------------- #
# one job
# --------------------------------------------------------------------------- #
@dataclass
class WebJob:
    """One browser upload, from "queued" to "downloaded"."""

    job_id: str
    source_name: str          # original filename (shown in the UI)
    source_path: Path         # where the upload was stored
    work_dir: Path
    state: str = STATE_QUEUED
    overall: float = 0.0
    stage: int = 0
    detail: str = "Waiting in the queue"
    error: str | None = None
    files_removed: bool = False
    result: PicturesResult | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None

    @property
    def spec(self) -> PicturesJobSpec:
        """Adapt to the pipeline's job description."""
        return PicturesJobSpec(
            job_id=self.job_id,
            source_path=self.source_path,
            work_dir=self.work_dir,
        )

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else time.monotonic()
        return end - self.started

    def to_dict(self, *, queue_position: int = 0) -> dict[str, Any]:
        """JSON snapshot for the UI."""
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "state": self.state,
            "source_name": self.source_name,
            "percent": round(self.overall * 100.0, 1),
            "stage": self.stage,
            "stage_title": STAGE_TITLES.get(self.stage, STAGE_TITLES[0]),
            "detail": self.detail,
            "queue_position": queue_position,
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
        }
        if (
            self.state == STATE_DONE
            and self.result is not None
            and not self.files_removed
        ):
            payload["download_url"] = f"/api/jobs/{self.job_id}/download"
            payload["download_name"] = f"{settings.WEB_DOWNLOAD_NAME}.zip"
            payload["size_mb"] = round(self.result.size_bytes / 1024 / 1024, 1)
            payload["pictures"] = self.result.picture_count
        return payload


# --------------------------------------------------------------------------- #
# registry + workers
# --------------------------------------------------------------------------- #
class WebJobStore:
    """In-memory job registry and worker pool around :func:`run_pipeline`."""

    def __init__(self, *, max_concurrent: int | None = None, ttl: int | None = None) -> None:
        self._max = max(1, max_concurrent or settings.MAX_CONCURRENT_JOBS)
        self._ttl = settings.WEB_JOB_TTL_SECONDS if ttl is None else ttl
        self._jobs: dict[str, WebJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._reaper_task: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = 0
        self._stopping = False
        # Guards the fields written by the pipeline thread (see _on_progress).
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        """Start the workers; must be called from the event loop."""
        self._loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(
            max_workers=self._max, thread_name_prefix="web-pipeline"
        )
        for index in range(self._max):
            self._workers.append(
                self._loop.create_task(self._worker(index), name=f"web-worker-{index}")
            )
        self._reaper_task = self._loop.create_task(self._reaper(), name="web-reaper")
        log.info(
            "web job store ready: %d worker(s), results kept for %ds",
            self._max,
            self._ttl,
        )

    async def stop(self) -> None:
        self._stopping = True
        for job in list(self._jobs.values()):
            job.cancel_event.set()
        tasks = list(self._workers)
        if self._reaper_task is not None:
            tasks.append(self._reaper_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._reaper_task = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        for job in list(self._jobs.values()):
            self._remove_files(job)
        self._jobs.clear()
        log.info("web job store stopped")

    # -- bookkeeping ------------------------------------------------------- #
    def get(self, job_id: str) -> WebJob | None:
        return self._jobs.get(job_id)

    def stats(self) -> tuple[int, int]:
        """``(running, queued)``."""
        return self._running, self._queue.qsize()

    def queue_position(self, job: WebJob) -> int:
        """1-based position while queued, 0 once running/finished."""
        if job.state != STATE_QUEUED:
            return 0
        ahead = 0
        for other in self._jobs.values():
            if other is job or other.state in _FINAL_STATES:
                continue
            if other.started is not None or other.created < job.created:
                ahead += 1
        return ahead + 1

    # -- submission / cancellation ---------------------------------------- #
    def submit(self, job: WebJob) -> int:
        """Register a job whose files are already on disk; returns its position.

        Must be called from the event loop (``asyncio.Queue`` is not
        thread-safe).
        """
        if self._loop is None:
            raise RuntimeError("WebJobStore.start() was never called")
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job.job_id)
        position = self.queue_position(job)
        log.info("web job %s queued (position=%d)", job.job_id, position)
        return position

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop (kills the running ffmpeg/demucs subprocesses)."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.state not in _FINAL_STATES:
            log.info("cancelling web job %s on user request", job_id)
        job.cancel_event.set()
        return True

    def after_download(self, job_id: str) -> None:
        """Called once the browser has finished downloading the result."""
        job = self._jobs.get(job_id)
        if job is None or job.files_removed:
            return
        log.info("web job %s downloaded - releasing its files", job_id)
        self._remove_files(job)

    # -- worker ------------------------------------------------------------ #
    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            try:
                if job is None or self._stopping:
                    continue
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a worker must never die
                log.exception("web worker %d: unexpected failure for job %s", index, job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job: WebJob) -> None:
        assert self._executor is not None and self._loop is not None
        if job.cancel_event.is_set():
            self._finish(job, STATE_CANCELLED, detail="Cancelled before it started")
            self._remove_files(job, keep=settings.KEEP_TEMP_ON_ERROR)
            return

        self._running += 1
        job.started = time.monotonic()
        job.state = STATE_RUNNING
        job.detail = "Starting..."
        log.info("web job %s started (%s)", job.job_id, job.source_name)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                functools.partial(
                    run_pictures_pipeline,
                    job.spec,
                    progress_cb=functools.partial(self._on_progress, job),
                    cancel_event=job.cancel_event,
                ),
            )
        except JobCancelledError:
            log.info("web job %s cancelled", job.job_id)
            self._finish(job, STATE_CANCELLED, detail="Cancelled")
            self._remove_files(job, keep=settings.KEEP_TEMP_ON_ERROR)
        except ProcessingError as exc:
            log.warning("web job %s failed: %s", job.job_id, exc)
            self._finish(job, STATE_ERROR, error=exc.user_message)
            self._remove_files(job, keep=settings.KEEP_TEMP_ON_ERROR)
        except Exception:  # noqa: BLE001 - never leak a traceback to the browser
            log.exception("web job %s crashed", job.job_id)
            self._finish(
                job,
                STATE_ERROR,
                error="Something went wrong while processing the video. Please try again.",
            )
            self._remove_files(job, keep=settings.KEEP_TEMP_ON_ERROR)
        else:
            # The result must stay on disk until it has been downloaded (or
            # the TTL reaper collects it) - so no cleanup here.
            job.result = result
            self._finish(
                job,
                STATE_DONE,
                overall=1.0,
                stage=3,
                detail=f"{result.picture_count} pictures ready",
            )
            log.info(
                "web job %s finished in %.1fs (%d pictures, %.1f MB)",
                job.job_id,
                result.elapsed_seconds,
                result.picture_count,
                result.size_bytes / 1024 / 1024,
            )
        finally:
            self._running -= 1

    def _finish(
        self,
        job: WebJob,
        state: str,
        *,
        overall: float | None = None,
        stage: int | None = None,
        detail: str | None = None,
        error: str | None = None,
    ) -> None:
        job.state = state
        job.finished = time.monotonic()
        if overall is not None:
            job.overall = overall
        if stage is not None:
            job.stage = stage
        if detail is not None:
            job.detail = detail
        job.error = error

    # -- progress (called from the pipeline thread) ------------------------- #
    def _on_progress(
        self,
        job: WebJob,
        overall: float,
        stage: int,
        stage_fraction: float,
        detail: str,
    ) -> None:
        if job.cancel_event.is_set():
            return  # the pipeline is about to abort; no more updates
        with self._lock:
            job.overall = overall
            job.stage = stage
            job.detail = stage_detail(stage, stage_fraction) or detail or STAGE_TITLES.get(stage, "")

    # -- cleanup ------------------------------------------------------------ #
    def _remove_files(self, job: WebJob, *, keep: bool = False) -> None:
        """Delete the job's working directory (idempotent, best effort)."""
        if job.files_removed:
            return
        if keep or not settings.DELETE_TEMP_AFTER_SEND:
            log.info("keeping the files of web job %s in %s", job.job_id, job.work_dir)
            return
        try:
            shutil.rmtree(job.work_dir, ignore_errors=True)
            log.debug("removed %s", job.work_dir)
        except OSError as exc:  # noqa: BLE001 - cleanup must never fail the job
            log.debug("could not remove %s: %s", job.work_dir, exc)
        job.files_removed = True

    async def _reaper(self) -> None:
        """Remove finished jobs that were never downloaded."""
        try:
            while True:
                await asyncio.sleep(_REAPER_INTERVAL)
                now = time.monotonic()
                for job in list(self._jobs.values()):
                    if (
                        job.state in _FINAL_STATES
                        and job.finished is not None
                        and now - job.finished > self._ttl
                    ):
                        log.info("reaping web job %s", job.job_id)
                        self._remove_files(job)
                        self._jobs.pop(job.job_id, None)
        except asyncio.CancelledError:
            pass


# Module-level singleton shared by the FastAPI app.
job_store = WebJobStore()
