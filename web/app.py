"""
Browser interface for the 150-picture tool.

    python -m web.app            # then open http://localhost:8080

Upload a video, get a ZIP with 150 side-by-side pictures taken from it:

* the video is uploaded straight from the browser, so the Telegram Bot API
  size limits do not apply - only ``MAX_VIDEO_SIZE`` (config/settings.py),
* progress is served as JSON and polled by the page,
* ``GET /api/jobs/<id>/download`` returns the finished picture ZIP.

There is no overlay step, no audio separation and no video render.

Set ``WEB_PASSWORD`` in ``.env`` before the server is reachable from the
internet: the whole interface (download included) then sits behind HTTP
Basic auth.
"""

from __future__ import annotations

import base64
import hmac
import logging
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.background import BackgroundTask

from config import settings
from config.logging_setup import setup_logging
from web.jobs import (
    STATE_DONE,
    WebJob,
    job_dir_for,
    job_store,
    new_job_id,
)

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"

_CHUNK_BYTES = 1024 * 1024
# Multipart framing and the form fields need room on top of the video itself
# when checking Content-Length early.
_MULTIPART_HEADROOM = 32 * 1024 * 1024

_SAFE_SUFFIX = re.compile(r"\.[a-z0-9]{1,8}$")

# Vercel can serve the page but not run the pipeline (4.5 MB request cap, no
# FFmpeg, 300 s function limit).  Say so instead of accepting a job that is
# guaranteed to fail.
_VERCEL_NOTICE = (
    "This deployment runs on Vercel, which cannot process videos: uploads are "
    "limited to 4.5 MB, the serverless runtime has no FFmpeg and a function "
    "stops after 300 seconds. Deploy the included Dockerfile.web on Render "
    "(render.yaml) or any Docker host to use the tool."
)


# --------------------------------------------------------------------------- #
# app lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_directories()
    job_store.start()
    password_note = "password required" if settings.WEB_PASSWORD else "NO password (local use only)"
    log.info(
        "web interface ready on %s:%d - max upload %.0f MB, %s",
        settings.WEB_HOST,
        settings.WEB_PORT,
        settings.MAX_VIDEO_SIZE / 1024 / 1024,
        password_note,
    )
    try:
        yield
    finally:
        await job_store.stop()


app = FastAPI(
    title="150-picture tool", version="1.0", lifespan=lifespan, docs_url=None, redoc_url=None
)


# --------------------------------------------------------------------------- #
# auth + early upload-size gate
# --------------------------------------------------------------------------- #
def _basic_auth_ok(header: str | None) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    return hmac.compare_digest(username, settings.WEB_USERNAME) and hmac.compare_digest(
        password, settings.WEB_PASSWORD
    )


@app.middleware("http")
async def gate(request: Request, call_next):  # type: ignore[no-untyped-def]
    if settings.WEB_PASSWORD and request.url.path != "/api/health":
        if not _basic_auth_ok(request.headers.get("authorization")):
            return PlainTextResponse(
                "Authentication required.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="video-tool"'},
            )

    # Refuse oversized uploads before the body is written to disk anywhere.
    if request.method == "POST" and request.url.path == "/api/jobs":
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            limit = settings.MAX_VIDEO_SIZE + _MULTIPART_HEADROOM
            if int(content_length) > limit:
                return _too_large_response(settings.MAX_VIDEO_SIZE)

    return await call_next(request)


def _too_large_message(limit: int) -> str:
    return f"The file is larger than the limit of {limit / 1024 / 1024:.0f} MB."


def _too_large_response(limit: int) -> PlainTextResponse:
    return PlainTextResponse(_too_large_message(limit), status_code=413)


# --------------------------------------------------------------------------- #
# pages / config
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_INDEX_HTML)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> PlainTextResponse:
    return PlainTextResponse("", status_code=204)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def public_config() -> dict[str, object]:
    """Everything the page needs to render its form."""
    return {
        "max_upload_mb": round(settings.MAX_VIDEO_SIZE / 1024 / 1024),
        "password_required": bool(settings.WEB_PASSWORD),
        "picture_count": settings.FRAME_COUNT,
        "photo_count": settings.PHOTO_COUNT,
        "stage_count": 3,
        "serverless": settings.ON_VERCEL,
    }


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
@app.post("/api/jobs", status_code=202)
async def create_job(
    video: Annotated[UploadFile, File()],
) -> JSONResponse:
    if settings.ON_VERCEL:
        raise HTTPException(status_code=503, detail=_VERCEL_NOTICE)
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file was uploaded.")

    job_id = new_job_id()
    work_dir = job_dir_for(job_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    display_name = _display_name(video.filename)
    try:
        source_path = work_dir / f"{settings.SOURCE_FILENAME}{_safe_suffix(display_name)}"
        video_bytes = await _stream_to_disk(video, source_path, settings.MAX_VIDEO_SIZE)
    except BaseException:
        # Any failure (rejection, client disconnect, ...) must not leave a
        # half-written work directory behind.
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    job = WebJob(
        job_id=job_id,
        source_name=display_name,
        source_path=source_path,
        work_dir=work_dir,
    )
    position = job_store.submit(job)
    log.info(
        "web job %s accepted (%s, %.1f MB, position=%d)",
        job_id,
        display_name,
        video_bytes / 1024 / 1024,
        position,
    )
    return JSONResponse(status_code=202, content={"job": job.to_dict(queue_position=position)})


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, object]:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return {"job": job.to_dict(queue_position=job_store.queue_position(job))}


@app.post("/api/jobs/{job_id}/cancel", status_code=202)
async def cancel_job(job_id: str) -> dict[str, object]:
    if not job_store.cancel(job_id):
        raise HTTPException(status_code=404, detail="Unknown job.")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download")
async def download(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if job is None or job.state != STATE_DONE or job.result is None:
        raise HTTPException(status_code=404, detail="The result is not ready yet.")
    path = job.result.zip_path
    if not path.is_file():
        raise HTTPException(
            status_code=410,
            detail="The result has already been downloaded and removed from the server.",
        )
    filename = f"{settings.WEB_DOWNLOAD_NAME}-{job.job_id}.zip"
    # Free the disk space once the transfer has finished.
    return FileResponse(
        path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(job_store.after_download, job_id),
    )


# --------------------------------------------------------------------------- #
# upload helpers
# --------------------------------------------------------------------------- #
def _display_name(filename: str) -> str:
    """A short, harmless version of the client's filename (for the UI only)."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name[:120] or "video"


def _safe_suffix(filename: str) -> str:
    match = _SAFE_SUFFIX.search(Path(filename).suffix.lower())
    return match.group(0) if match else ""


async def _stream_to_disk(upload: UploadFile, destination: Path, limit: int) -> int:
    """Copy an upload to disk in chunks, refusing anything over ``limit``."""
    written = 0
    with destination.open("wb") as sink:
        while chunk := await upload.read(_CHUNK_BYTES):
            written += len(chunk)
            if written > limit:
                raise HTTPException(status_code=413, detail=_too_large_message(limit))
            sink.write(chunk)
    if written == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return written


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    import uvicorn

    setup_logging()
    uvicorn.run(
        app,
        host=settings.WEB_HOST,
        port=settings.WEB_PORT,
        log_config=None,  # keep the logging configuration from setup_logging()
    )


if __name__ == "__main__":
    main()
