"""
Voice / music separation.

Uses Demucs (htdemucs) in ``--two-stems=vocals`` mode: a real source
separation model - *not* an EQ filter.  The vocal stem is kept, the
instrumental stem is discarded.

The model runs on CUDA when a GPU is available and falls back to CPU
otherwise (``DEMUCS_DEVICE=auto``).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path

from config import settings
from processing.errors import SeparationError
from processing.ffmpeg_utils import ProgressCallback, ffprobe_json, run_process

log = logging.getLogger(__name__)

_PERCENT_RE = re.compile(r"(\d{1,3})%\|")


def _torch_import():  # type: ignore[no-untyped-def]
    try:
        import torch  # noqa: PLC0415 - optional heavy dependency
    except Exception as exc:  # noqa: BLE001
        raise SeparationError(f"PyTorch is not available: {exc}") from exc
    return torch


def gpu_available() -> bool:
    try:
        torch = _torch_import()
    except SeparationError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - broken driver etc.
        return False


def resolve_device() -> str:
    """``auto`` -> cuda when possible, otherwise cpu."""
    requested = settings.DEMUCS_DEVICE
    if requested in {"cuda", "cpu"}:
        return requested
    device = "cuda" if gpu_available() else "cpu"
    log.info("vocal separation device: %s (auto-detected)", device)
    return device


def resolve_jobs(device: str) -> int:
    """
    Number of segment workers for the separator (``DEMUCS_JOBS``).

    ``0`` means auto: on CPU half of the logical cores process segments in
    parallel - measured ~1.6x faster than a single worker on a quad-core
    laptop.  The cap is 8 workers: on big cloud vCPUs (16+) 8 workers roughly
    double the throughput of 4, which is what brings a 20-minute separation
    down to ~5 minutes there.  Each worker needs ~1.5 GB of RAM, so on hosts
    with less than ~12 GB free set ``DEMUCS_JOBS=4`` explicitly.  On a GPU a
    single worker already saturates the card, so auto resolves to 1 there.
    """
    if settings.DEMUCS_JOBS > 0:
        return settings.DEMUCS_JOBS
    if device != "cpu":
        return 1
    cores = os.cpu_count() or 2
    return max(1, min(8, cores // 2))


def download_model() -> None:
    """Pre-fetch the Demucs weights (used at Docker build time)."""
    from demucs.pretrained import get_model  # noqa: PLC0415

    log.info("downloading model %s into %s", settings.MODEL_NAME, settings.MODELS_DIRECTORY)
    get_model(settings.MODEL_NAME)


def separate_vocals(
    audio_path: Path,
    work_dir: Path,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_event=None,  # threading.Event | None
) -> Path:
    """
    Run source separation and return the path of the vocal stem.

    ``audio_path`` stays untouched; ``vocals.wav`` is written next to it.
    """
    if settings.MODEL_BACKEND != "demucs":
        raise SeparationError(f"Unsupported MODEL_BACKEND: {settings.MODEL_BACKEND}")

    device = resolve_device()
    jobs = resolve_jobs(device)
    stems_dir = work_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    separator_script = Path(__file__).resolve().parent.parent / "scripts" / "demucs_runner.py"
    if separator_script.is_file():
        base_cmd = [sys.executable, str(separator_script)]
    else:  # pragma: no cover - used when only the package is installed
        base_cmd = [sys.executable, "-m", "demucs.separate"]

    cmd = [
        *base_cmd,
        "--two-stems=vocals",
        "-n",
        settings.MODEL_NAME,
        "-d",
        device,
        "-j",
        str(jobs),
        "--overlap",
        f"{settings.DEMUCS_OVERLAP:g}",
        "-o",
        str(stems_dir),
        "--filename",
        "{stem}.{ext}",
    ]
    if settings.DEMUCS_EXTRA_ARGS:
        cmd += settings.DEMUCS_EXTRA_ARGS.split()
    cmd.append(str(audio_path))

    extra_env = {
        "TORCH_HOME": str(settings.MODELS_DIRECTORY),
        "HF_HOME": str(settings.MODELS_DIRECTORY / "hub"),
    }
    if jobs > 1 and device == "cpu":
        # Keep worker threads x torch threads ~= logical cores so the workers
        # do not fight each other for CPU time.
        per_worker = max(1, (os.cpu_count() or 2) // jobs)
        extra_env["OMP_NUM_THREADS"] = str(per_worker)
        extra_env["MKL_NUM_THREADS"] = str(per_worker)

    log.info(
        "separating vocals: model=%s device=%s jobs=%d overlap=%g file=%s",
        settings.MODEL_NAME,
        device,
        jobs,
        settings.DEMUCS_OVERLAP,
        audio_path.name,
    )

    last_percent = -1

    def on_stderr(line: str) -> None:
        nonlocal last_percent
        match = _PERCENT_RE.search(line)
        if match and progress_cb is not None:
            percent = min(100, int(match.group(1)))
            if percent > last_percent:
                last_percent = percent
                progress_cb(percent / 100.0)

    returncode, _stdout, stderr_tail = run_process(
        cmd,
        timeout=settings.JOB_TIMEOUT_SECONDS,
        on_stderr_line=on_stderr,
        extra_env=extra_env,
        cancel_event=cancel_event,
    )
    if returncode != 0:
        detail = "\n".join(stderr_tail[-15:]) or f"exit code {returncode}"
        log.error("demucs failed:\n%s", detail)
        raise SeparationError(f"Vocal separation failed: {detail}")

    vocals = _find_vocals(stems_dir)
    if vocals is None:
        raise SeparationError("Vocal separation finished but produced no vocal stem.")

    # stable name + drop the instrumental/drum/bass stems to free disk space
    target = work_dir / settings.VOCALS_FILENAME
    if target.exists():
        target.unlink()
    shutil.move(str(vocals), str(target))
    for leftover in stems_dir.rglob("*"):
        if leftover.is_file():
            leftover.unlink()
    shutil.rmtree(stems_dir, ignore_errors=True)

    if progress_cb:
        progress_cb(1.0)

    _verify_duration(target, audio_path)
    log.info("vocals ready: %s (%.1f MB)", target.name, target.stat().st_size / 1024 / 1024)
    return target


def _find_vocals(stems_dir: Path) -> Path | None:
    for pattern in ("**/vocals.wav", "**/vocals.flac", "**/vocals.mp3", "**/*vocals*"):
        for candidate in sorted(stems_dir.glob(pattern)):
            if candidate.is_file() and "no_vocals" not in candidate.name:
                return candidate
    return None


def _verify_duration(vocals: Path, original: Path) -> None:
    try:
        vocals_info = ffprobe_json(vocals)
        vocals_duration = float((vocals_info.get("format") or {}).get("duration") or 0.0)
        original_duration = float((ffprobe_json(original).get("format") or {}).get("duration") or 0.0)
    except Exception as exc:  # noqa: BLE001 - verification is best effort
        log.warning("could not verify vocals duration: %s", exc)
        return
    if vocals_duration <= 0:
        raise SeparationError("The separated vocal track is empty.")
    if original_duration and abs(vocals_duration - original_duration) > 1.0:
        log.warning(
            "vocals duration %.2fs differs from source audio %.2fs",
            vocals_duration,
            original_duration,
        )
