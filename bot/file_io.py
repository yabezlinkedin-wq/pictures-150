"""
Telegram file transfer helpers with progress reporting.

The official Bot API caps downloads at 20 MB and uploads at 50 MB; a local Bot
API server (see ``docker-compose.yml``) lifts both limits to 2000 MB.  These
helpers work with either mode:

* downloads stream over HTTP (or are copied directly when the local Bot API
  server already wrote the file to this machine),
* uploads stream from disk through a counting file wrapper, so the 20-minute
  output video never has to be held in RAM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

import httpx
from telegram import Bot, InputFile, Message
from telegram.error import TelegramError

from config import settings
from processing.errors import TelegramIOError

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

_CHUNK_SIZE = 256 * 1024
_DOWNLOAD_ATTEMPTS = 3
_HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=180.0, write=180.0, pool=60.0)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
async def download_file(
    bot: Bot,
    file_id: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Download a Telegram file to ``destination`` (streamed, with progress)."""
    try:
        tg_file = await bot.get_file(file_id)
    except TelegramError as exc:
        raise TelegramIOError(f"getFile failed: {exc}") from exc

    file_path = tg_file.file_path or ""
    if not file_path:
        raise TelegramIOError("Telegram did not return a file path.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s (declared size %s)", file_path, destination.name, expected_size)

    # Local Bot API server: file_path is an absolute path on its own disk; when
    # the bot shares that filesystem the file can simply be copied.
    if os.path.isabs(file_path) and Path(file_path).is_file():
        await asyncio.to_thread(
            _copy_with_progress, Path(file_path), destination, expected_size, progress_cb
        )
        return destination

    url = f"{bot.base_file_url}/{file_path}"
    last_error: Exception | None = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            await _stream_download(url, destination, expected_size, progress_cb)
            return destination
        except (httpx.HTTPError, TelegramIOError) as exc:
            last_error = exc
            log.warning("download attempt %d/%d failed: %s", attempt, _DOWNLOAD_ATTEMPTS, exc)
            if destination.exists():
                destination.unlink()
            if attempt < _DOWNLOAD_ATTEMPTS:
                await asyncio.sleep(2.0 * attempt)

    raise TelegramIOError(f"Could not download the file from Telegram: {last_error}")


async def _stream_download(
    url: str,
    destination: Path,
    expected_size: int | None,
    progress_cb: ProgressCallback | None,
) -> None:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise TelegramIOError(
                    f"Telegram file server answered HTTP {response.status_code}"
                )
            total = expected_size or int(response.headers.get("content-length") or 0)
            done = 0
            with open(destination, "wb") as handle:
                async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                    handle.write(chunk)
                    done += len(chunk)
                    if progress_cb and total:
                        progress_cb(min(0.999, done / total))
    if progress_cb:
        progress_cb(1.0)
    log.info("download finished: %s (%s bytes)", destination.name, destination.stat().st_size)


def _copy_with_progress(
    source: Path,
    destination: Path,
    expected_size: int | None,
    progress_cb: ProgressCallback | None,
) -> None:
    total = expected_size or source.stat().st_size
    done = 0
    with open(source, "rb") as src, open(destination, "wb") as dst:
        while True:
            chunk = src.read(_CHUNK_SIZE)
            if not chunk:
                break
            dst.write(chunk)
            done += len(chunk)
            if progress_cb and total:
                progress_cb(min(0.999, done / total))
    if progress_cb:
        progress_cb(1.0)


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #
class _ProgressReader:
    """
    File-like wrapper that reports the number of bytes read.

    HTTPX reads the multipart body from this object in 64 KB chunks, so the
    progress callback runs on the networking thread - callers must treat it as
    thread-safe (the job progress reporter is).
    """

    def __init__(self, path: Path, progress_cb: ProgressCallback | None) -> None:
        self._handle = open(path, "rb")  # noqa: SIM115 - closed in close()
        self._size = max(1, path.stat().st_size)
        self._read = 0
        self._callback = progress_cb
        self.name = str(path)

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        if chunk:
            self._read += len(chunk)
            if self._callback is not None:
                try:
                    self._callback(min(1.0, self._read / self._size))
                except Exception:  # noqa: BLE001 - progress must never break the upload
                    log.debug("upload progress callback failed", exc_info=True)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()

    def fileno(self) -> int:
        return self._handle.fileno()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # noqa: BLE001
            pass


async def send_video_file(
    bot: Bot,
    chat_id: int,
    path: Path,
    *,
    caption: str | None = None,
    duration: float | None = None,
    width: int | None = None,
    height: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Message:
    """Send the finished video (streamed from disk, with upload progress)."""
    reader = _ProgressReader(path, progress_cb)
    input_file = InputFile(reader, filename=path.name, read_file_handle=False)
    try:
        message = await bot.send_video(
            chat_id=chat_id,
            video=input_file,
            filename=path.name,
            caption=caption,
            duration=int(round(duration)) if duration else None,
            width=width,
            height=height,
            supports_streaming=True,
            read_timeout=settings.TELEGRAM_READ_TIMEOUT,
            write_timeout=settings.TELEGRAM_WRITE_TIMEOUT,
        )
        log.info("video sent to %s (%s)", chat_id, path.name)
        return message
    except TelegramError as exc:
        text = str(exc)
        if "too large" in text.lower() or "entity too large" in text.lower():
            raise TelegramIOError(
                f"Telegram rejected the video as too large: {text}"
            ) from exc
        raise TelegramIOError(f"sendVideo failed: {text}") from exc
    finally:
        reader.close()


def copy_to_output(path: Path, job_id: str) -> Path:
    """Persist a copy of the finished video (used when SAVE_OUTPUT_COPY is on)."""
    settings.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = settings.OUTPUT_DIRECTORY / f"{job_id}_{path.name}"
    shutil.copy2(path, destination)
    return destination
