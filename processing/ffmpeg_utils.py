"""
Thin, well-behaved wrappers around the FFmpeg / FFprobe command line tools.

* long running commands report progress through ``progress_cb(fraction)``
* every command can time out (the process is killed and a friendly error
  is raised)
* stderr is captured (tail only) so technical details end up in the log,
  never in the Telegram chat
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Sequence

from processing.errors import (
    FFmpegError,
    JobCancelledError,
    JobTimeoutError,
    ToolMissingError,
    VideoInfoError,
)

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

FFMPEG_BINARY = "ffmpeg"
FFPROBE_BINARY = "ffprobe"


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #
def ensure_tools_available() -> None:
    """Fail fast with a clear message when FFmpeg is not installed."""
    missing = [tool for tool in (FFMPEG_BINARY, FFPROBE_BINARY) if shutil.which(tool) is None]
    if missing:
        raise ToolMissingError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install FFmpeg (https://ffmpeg.org) or use the provided Docker image."
        )


# --------------------------------------------------------------------------- #
# generic process runner
# --------------------------------------------------------------------------- #
def _emit_lines(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split ``buffer`` on \\r or \\n (tqdm uses carriage returns)."""
    lines: list[bytes] = []
    while True:
        idx_r = buffer.find(b"\r")
        idx_n = buffer.find(b"\n")
        candidates = [i for i in (idx_r, idx_n) if i != -1]
        if not candidates:
            break
        idx = min(candidates)
        lines.append(buffer[:idx])
        buffer = buffer[idx + 1 :]
    return lines, buffer


def run_process(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: Path | str | None = None,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    stderr_tail_lines: int = 60,
    extra_env: dict[str, str] | None = None,
    cancel_event: "threading.Event | None" = None,
) -> tuple[int, list[str], list[str]]:
    """
    Run ``cmd`` while streaming both pipes through reader threads.

    Returns ``(returncode, stdout_lines, stderr_tail)``.
    Raises :class:`JobTimeoutError` when the command exceeds ``timeout`` and
    :class:`JobCancelledError` when ``cancel_event`` is set while running.
    """
    import os

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    log.debug("run: %s", format_command(cmd))
    proc = subprocess.Popen(  # noqa: S603 - arguments are fully controlled
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stdout_lines: list[str] = []
    stderr_tail: list[str] = []
    stderr_tail_deque: list[str] = []  # kept small, only for error reporting

    def reader(stream, is_stdout: bool) -> None:  # type: ignore[no-untyped-def]
        buffer = b""
        try:
            while True:
                # read1() returns as soon as *some* bytes are available;
                # read() would block until the full 4096 bytes have piled up,
                # which held progress updates back for minutes (tqdm writes
                # small chunks separated by carriage returns).
                chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
                if not chunk:
                    break
                buffer += chunk
                lines, buffer = _emit_lines(buffer)
                for raw in lines:
                    text = raw.decode("utf-8", "replace").strip()
                    if not text:
                        continue
                    if is_stdout:
                        stdout_lines.append(text)
                        if on_stdout_line:
                            on_stdout_line(text)
                    else:
                        if on_stderr_line:
                            on_stderr_line(text)
                        stderr_tail_deque.append(text)
                        if len(stderr_tail_deque) > stderr_tail_lines:
                            del stderr_tail_deque[0]
        except Exception:  # pragma: no cover - defensive
            log.debug("reader thread error", exc_info=True)
        finally:
            remainder = buffer.decode("utf-8", "replace").strip()
            if remainder:
                if is_stdout:
                    stdout_lines.append(remainder)
                else:
                    stderr_tail_deque.append(remainder)

    t_out = threading.Thread(target=reader, args=(proc.stdout, True), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, False), daemon=True)
    t_out.start()
    t_err.start()

    cancelled = threading.Event()
    if cancel_event is not None:

        def watch() -> None:
            while not cancelled.is_set() and proc.poll() is None:
                if cancel_event.wait(0.25):
                    cancelled.set()
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001 - process may have just exited
                        pass
                    return

        threading.Thread(target=watch, daemon=True).start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        raise JobTimeoutError(
            f"Command timed out after {timeout:.0f}s: {format_command(cmd)}"
        ) from None
    finally:
        t_out.join(timeout=10)
        t_err.join(timeout=10)

    if cancelled.is_set():
        raise JobCancelledError(f"Command cancelled: {format_command(cmd)}")

    stderr_tail.extend(stderr_tail_deque)
    return returncode, stdout_lines, stderr_tail


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


# --------------------------------------------------------------------------- #
# ffprobe
# --------------------------------------------------------------------------- #
def ffprobe_json(path: Path | str, *, timeout: float = 120.0) -> dict:
    cmd = [
        FFPROBE_BINARY,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    returncode, stdout_lines, stderr_tail = run_process(cmd, timeout=timeout)
    if returncode != 0:
        detail = " | ".join(stderr_tail[-5:]) or f"exit code {returncode}"
        log.error("ffprobe failed for %s: %s", path, detail)
        raise VideoInfoError(f"ffprobe failed for {path}: {detail}")
    try:
        return json.loads("\n".join(stdout_lines))
    except json.JSONDecodeError as exc:  # pragma: no cover - very unlikely
        raise VideoInfoError(f"Could not parse ffprobe output for {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #
class _FfmpegProgressParser:
    """Convert ``-progress pipe:1`` output into a 0..1 fraction."""

    def __init__(self, total_duration: float | None, callback: ProgressCallback | None) -> None:
        self.total_duration = total_duration if total_duration and total_duration > 0 else None
        self.callback = callback
        self.last_fraction = 0.0

    def status(self, line: str) -> None:
        if not self.callback or self.total_duration is None:
            return
        if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
            raw = line.split("=", 1)[1].strip()
            if raw in {"N/A", ""}:
                return
            try:
                # ffmpeg reports microseconds for both keys (historic quirk)
                seconds = int(raw) / 1_000_000.0
            except ValueError:
                return
            fraction = max(0.0, min(1.0, seconds / self.total_duration))
            if fraction >= self.last_fraction:
                self.last_fraction = fraction
                self.callback(fraction)
        elif line.startswith("progress=end"):
            self.last_fraction = 1.0
            self.callback(1.0)


def run_ffmpeg(
    args: Sequence[str],
    *,
    total_duration: float | None = None,
    progress_cb: ProgressCallback | None = None,
    timeout: float | None = None,
    cwd: Path | str | None = None,
    error_cls: type[FFmpegError] = FFmpegError,
    error_message: str = "FFmpeg command failed",
    extra_env: dict[str, str] | None = None,
    cancel_event: "threading.Event | None" = None,
) -> None:
    """
    Run an FFmpeg command.

    ``args`` must not contain the leading ``ffmpeg`` binary - it is added here
    together with the logging / progress flags.
    """
    cmd: list[str] = [FFMPEG_BINARY]
    cmd += ["-hide_banner", "-nostats", "-loglevel", "warning"]
    if progress_cb is not None:
        cmd += ["-progress", "pipe:1"]
    cmd += [str(a) for a in args]

    parser = _FfmpegProgressParser(total_duration, progress_cb)
    returncode, _stdout, stderr_tail = run_process(
        cmd,
        timeout=timeout,
        cwd=cwd,
        on_stdout_line=parser.status,
        extra_env=extra_env,
        cancel_event=cancel_event,
    )
    if returncode != 0:
        detail = "\n".join(stderr_tail[-20:]) or f"exit code {returncode}"
        log.error("%s\ncommand: %s\n%s", error_message, format_command(cmd), detail)
        raise error_cls(f"{error_message}: {detail}")
    if progress_cb is not None:
        progress_cb(1.0)


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0 or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"  # pragma: no cover
