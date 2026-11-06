#!/usr/bin/env python3
"""
End-to-end pipeline check (the acceptance criteria of the specification).

The script builds a small synthetic source video - 300 unique slides (the
three reference photos from ``vibe_images/``, each wearing a unique
golden-ratio hue border and a slide number), speech from FFmpeg's ``flite``
filter and a music bed made of two sine tones - then runs the *real*
pipeline on it and verifies:

  1. exactly 300 photos are selected and named photo_001 .. photo_300,
  2. they are composed into 150 pairs (photo 1+2, ..., photo 299+300),
     every pair a 16:9 canvas with photo 2k-1 left and photo 2k right,
  3. every photo comes from its own equal source segment,
  4. both pair halves really show the right reference photo,
  5. the photos are completely static (no zoom / pan / animation) and every
     pair is displayed for exactly 8 s (150 x 8 s = 1200 s = 20 minutes),
  6. the primary face of every photo sits at the centre of its half (the
     biggest face wins - and it is never cut off),
  7. the slideshow (and the final video) last exactly 20:00,
  8. the vocals are separated from the music,
  9. the overlay appears at the right place, keeping its alpha channel,
 10. the final MP4 has a video *and* an audio stream,
 11. the final file is acceptable for Telegram (size, faststart),
 12. temporary files are removed when the job finishes.

The whole job directory is snapshotted the moment stage 5 completes, so the
checks can run on the real intermediate files while the pipeline cleans up
behind itself - which lets criterion 12 be verified as well.

Checks 4 and 6 need the three reference photos in ``vibe_images/``; they are
skipped (the source falls back to plain colour slides) when the files are
missing.

Usage:
    python scripts/e2e_check.py                        # 80 s source video
    python scripts/e2e_check.py --duration 45 --keep   # keep all files
    python scripts/e2e_check.py --source-size 1280x720 # exercise the 720p canvas
"""

from __future__ import annotations

import argparse
import colorsys
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from config import settings
from config.logging_setup import setup_logging
from processing import face_center
from processing import overlay as overlay_module
from processing import video_info
from processing.ffmpeg_utils import ffprobe_json, human_size
from processing.pipeline import JobSpec, PipelineResult, run_pipeline

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

CANVAS = (640, 360)              # synthetic source resolution (--source-size overrides)
OVERLAY_SIZE = (240, 90)         # original overlay size (kept by "original")
OVERLAY_FILL = (220, 30, 30)     # opaque half of the overlay
SIG_SIZE = (64, 36)              # signature used to match photos / pairs
TONE_HZ = (6300.0, 7500.0)       # music bed - far away from the speech range
TONE_ATTENUATION_DB = 6.0        # minimum drop of the tones in vocals.wav
_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the morning sun "
    "rises slowly over the quiet valley and the river keeps flowing past "
    "the old stone bridge every single day"
)

# reference material: the synthetic slides are rendered from these photos
VIBE_DIRNAME = "vibe_images"
VIBE_PATTERNS = ("portrait-two-faces*.png", "portrait-single-right*.png", "landscape-no-face*.png")
FACE_EXPECTED = (True, True, False)  # does this reference photo contain a face?

# samples used by the checks (1-based indices)
PHOTO_SAMPLES = (3, 43, 83, 123, 163, 203, 243, 283)  # one photo per source segment
SLOT_SAMPLES = (1, 37, 74, 111, 148)                  # pairs covering all three photos, both sides
STATIC_SLOTS = (1, 40, 90, 149)
BOUNDARY_SLOTS = (30, 80, 130)

# tolerances
STATIC_TOLERANCE_MAE = 3.0      # a still may not change inside its 8 s slot
BOUNDARY_MIN_MAE = 6.0          # adjacent pairs must differ by at least this
FACE_TOLERANCE_FRACTION = 0.04  # face centre may sit this far from the half centre
FACE_TOLERANCE_MIN_PX = 20
FACE_MARGIN_PX = 2              # at least this much air between face and half edge

VIBE_BASES: list[Image.Image] | None = None  # the three photos, fitted to CANVAS


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
class Reporter:
    """Collects PASS / FAIL / SKIP lines and prints them immediately."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        status = "PASS" if ok else "FAIL"
        self.results.append((status, name, detail))
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
        return ok

    def skip(self, name: str, detail: str = "") -> None:
        self.results.append(("SKIP", name, detail))
        print(f"  [SKIP] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for status, _, _ in self.results if status == "FAIL")


def run(args: list[str], *, timeout: float = 300.0) -> bytes:
    completed = subprocess.run(args, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise RuntimeError(f"{Path(args[0]).name} failed:\n  " + "\n  ".join(tail))
    return completed.stdout


# --------------------------------------------------------------------------- #
# synthetic input
# --------------------------------------------------------------------------- #
def _find_font() -> Path | None:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_vibe_bases() -> list[Image.Image] | None:
    """The three reference photos (fitted to the canvas), or ``None`` when absent."""
    directory = PROJECT_ROOT / VIBE_DIRNAME
    bases: list[Image.Image] = []
    for pattern in VIBE_PATTERNS:
        matches = sorted(directory.glob(pattern))
        if not matches:
            return None
        with Image.open(matches[0]) as image:
            bases.append(ImageOps.fit(image.convert("RGB"), CANVAS, method=Image.LANCZOS))
    return bases


def _border_colour(index: int) -> tuple[int, int, int]:
    # uniform spacing: any two slides differ by at least 1/300 of the hue
    # circle (a golden-ratio rotation had near-collisions at index +/- 144 -
    # especially with 144 % 3 == 0 sharing the base photo - which made the
    # photo-matching checks ambiguous).
    hue = (index / settings.PHOTO_COUNT) % 1.0
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, 0.85, 0.9))


def _vibe_slide(index: int, font_path: Path | None) -> Image.Image:
    """Reference photo + a unique hue border + the slide number."""
    assert VIBE_BASES is not None
    image = VIBE_BASES[index % len(VIBE_BASES)].copy()
    draw = ImageDraw.Draw(image)
    thickness = max(6, CANVAS[1] // 22)
    draw.rectangle([0, 0, CANVAS[0] - 1, CANVAS[1] - 1], outline=_border_colour(index), width=thickness)
    if font_path is not None:
        font = ImageFont.truetype(str(font_path), max(12, thickness - 6))
        draw.text(
            (CANVAS[0] // 2, thickness // 2),
            f"{index + 1:03d}",
            font=font,
            fill=(255, 255, 255),
            stroke_width=max(1, thickness // 16),
            stroke_fill=(0, 0, 0),
            anchor="mm",
        )
    return image


def _slide_image(index: int, font_path: Path | None) -> Image.Image:
    """Fallback slide: a unique background hue, a huge number and a growing strip."""
    hue = (index / settings.PHOTO_COUNT) % 1.0
    value = 0.35 + 0.25 * (index / settings.PHOTO_COUNT)
    background = tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, 0.65, value))
    label = str(index + 1)

    if font_path is not None:
        image = Image.new("RGB", CANVAS, background)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(font_path), int(CANVAS[1] * 2 / 3))
        draw.text(
            (CANVAS[0] // 2, CANVAS[1] // 2),
            label,
            font=font,
            fill=(255, 255, 255),
            anchor="mm",
        )
    else:  # no TrueType font around: draw small and blow it up (blocky but unique)
        small = Image.new("RGB", (64, 36), background)
        draw = ImageDraw.Draw(small)
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            ((64 - (bbox[2] - bbox[0])) / 2 - bbox[0], (36 - (bbox[3] - bbox[1])) / 2 - bbox[1]),
            label,
            font=font,
            fill=(255, 255, 255),
        )
        image = small.resize(CANVAS, Image.NEAREST)

    # a dark strip that grows with the segment index (one more unique feature)
    draw = ImageDraw.Draw(image)
    width = int(CANVAS[0] * (index + 1) / settings.PHOTO_COUNT)
    draw.rectangle([0, CANVAS[1] - 24, width - 1, CANVAS[1] - 1], fill=(16, 16, 16))
    return image


def _slide(index: int, font_path: Path | None) -> Image.Image:
    if VIBE_BASES:
        return _vibe_slide(index, font_path)
    return _slide_image(index, font_path)


def _speech_text(duration: float) -> str:
    repeats = min(12, max(2, math.ceil(duration / 9.0) + 1))
    return " ".join([_SENTENCE] * repeats)


def build_source_video(path: Path, duration: float, work_dir: Path) -> None:
    """300 unique slides + speech + music bed, encoded as H.264/AAC."""
    slides_dir = work_dir / "source_slides"
    if slides_dir.exists():
        shutil.rmtree(slides_dir)
    slides_dir.mkdir(parents=True)

    font_path = _find_font()
    for index in range(settings.PHOTO_COUNT):
        _slide(index, font_path).save(slides_dir / f"slide_{index:03d}.png")
    mode = "3 reference photos with unique borders" if VIBE_BASES else "synthetic colour slides"
    print(
        f"  source: {settings.PHOTO_COUNT} slides ({mode}), one every "
        f"{duration / settings.PHOTO_COUNT:.3f}s"
        + ("" if font_path else " (no TrueType font found - using blocky fallback)")
    )

    tones = "+".join(
        f"{amplitude}*sin(2*PI*{frequency}*t)"
        for amplitude, frequency in ((0.20, TONE_HZ[0]), (0.16, TONE_HZ[1]))
    )
    args = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-start_number", "0",
        "-framerate", f"{settings.PHOTO_COUNT / duration:.6f}",
        "-i", str(slides_dir / "slide_%03d.png"),
        "-f", "lavfi", "-i", f"flite=text='{_speech_text(duration)}':voice=slt",
        "-f", "lavfi", "-i", f"aevalsrc={tones}:s=44100:d={duration:.3f}",
        "-filter_complex",
        "[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[speech];"
        "[2:a]aformat=sample_fmts=fltp:channel_layouts=stereo[music];"
        "[speech][music]amix=inputs=2:duration=longest:normalize=0,"
        "alimiter=limit=0.95[aout]",
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-t", f"{duration:.3f}", "-movflags", "+faststart",
        str(path),
    ]
    run(args)


def build_overlay(path: Path) -> None:
    """Opaque on the left, fully transparent on the right (tests alpha)."""
    image = Image.new("RGBA", OVERLAY_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, OVERLAY_SIZE[0] // 2 - 1, OVERLAY_SIZE[1] - 1], fill=OVERLAY_FILL + (255,))
    image.save(path, "PNG")


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def grab_signature(video: Path, timestamp: float) -> np.ndarray:
    """A 64x36 RGB signature of the frame at ``timestamp`` (for matching)."""
    raw = run([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1",
        "-vf", f"scale={SIG_SIZE[0]}:{SIG_SIZE[1]}:flags=bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    expected = SIG_SIZE[0] * SIG_SIZE[1] * 3
    if len(raw) < expected:
        raise RuntimeError(f"ffmpeg returned {len(raw)} bytes instead of {expected}")
    return np.frombuffer(raw[:expected], np.uint8).reshape(SIG_SIZE[1], SIG_SIZE[0], 3).astype(np.float32)


def grab_png(video: Path, timestamp: float, out_png: Path) -> None:
    """Save the full-resolution frame at ``timestamp`` as a PNG."""
    run([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1",
        "-update", "1", str(out_png),
    ])


def signature_of(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(SIG_SIZE, Image.BILINEAR), np.float32)


def load_signatures(directory: Path, prefix: str, count: int) -> dict[int, np.ndarray]:
    signatures: dict[int, np.ndarray] = {}
    for index in range(1, count + 1):
        with Image.open(directory / f"{prefix}_{index:03d}.jpg") as image:
            signatures[index] = signature_of(image)
    return signatures


def match_signature(signatures: dict[int, np.ndarray], signature: np.ndarray) -> tuple[int, float, float]:
    """Return ``(best_index, best_mae, runner_up_mae)``."""
    scores = sorted((sig_mae(signature, frame), index) for index, frame in signatures.items())
    (best_mae, best), (second_mae, _) = scores[0], scores[1]
    return best, best_mae, second_mae


def sig_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), np.float32)


def decode_mono(path: Path, *, start: float, duration: float, rate: int = 16000) -> np.ndarray:
    raw = run([
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-",
    ])
    samples = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if samples.size < rate:  # less than a second of audio
        raise RuntimeError(f"no usable audio decoded from {path.name}")
    return samples


def tone_level_db(samples: np.ndarray, rate: int, frequency: float) -> float:
    """Level of one tone relative to the speech band around it (in dB)."""
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    freqs = np.fft.rfftfreq(samples.size, 1.0 / rate)
    tone = float(np.sqrt(np.sum(spectrum[(freqs >= frequency - 40) & (freqs <= frequency + 40)] ** 2)))
    floor = float(np.median(spectrum[(freqs >= 250) & (freqs <= 7800)]))
    return 20.0 * math.log10(max(tone, 1e-9) / max(floor, 1e-9))


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_selection(reporter: Reporter, artifacts: Path, canvas: tuple[int, int], result: PipelineResult) -> None:
    frame_dir = artifacts / settings.FRAMES_DIRNAME
    photos = sorted(frame_dir.glob("photo_*.jpg"))
    reporter.record(
        "1. exactly 300 photos selected",
        len(photos) == settings.PHOTO_COUNT,
        f"{len(photos)} files in {settings.FRAMES_DIRNAME}/",
    )
    expected_photos = [f"photo_{index:03d}.jpg" for index in range(1, settings.PHOTO_COUNT + 1)]
    reporter.record("   photos named photo_001 .. photo_300", [p.name for p in photos] == expected_photos)

    pairs = sorted(frame_dir.glob("pair_*.jpg"))
    reporter.record(
        "2. 150 photo pairs composed (photo 1+2, ..., photo 299+300)",
        len(pairs) == settings.FRAME_COUNT,
        f"{len(pairs)} files in {settings.FRAMES_DIRNAME}/",
    )
    expected_pairs = [f"pair_{index:03d}.jpg" for index in range(1, settings.FRAME_COUNT + 1)]
    reporter.record("   pairs named pair_001 .. pair_150", [p.name for p in pairs] == expected_pairs)

    geometry = ""
    for pair in pairs:
        with Image.open(pair) as image:
            size = image.size
        if size != canvas:
            geometry = f"{pair.name} is {size[0]}x{size[1]}"
            break
    ratio = canvas[0] / canvas[1]
    reporter.record(
        "   every pair is a 16:9 canvas split into two halves",
        not geometry and abs(ratio - 16.0 / 9.0) < 1e-6,
        geometry or f"{canvas[0]}x{canvas[1]} = 2 x {canvas[0] // 2} x {canvas[1]}",
    )

    report = result.frames_report
    reporter.record(
        "   quality report clean (no copies / forced / corrupt)",
        report.total == settings.PHOTO_COUNT and report.frames_copied == 0 and report.segments_forced == 0,
        report.summary(),
    )

    flat = 0
    for path in photos + pairs:
        try:
            with Image.open(path) as image:
                image.load()
                gray = np.asarray(image.convert("L"), np.float32)
        except Exception:  # noqa: BLE001 - unreadable file
            flat += 1
            continue
        if gray.std() < settings.FLAT_FRAME_STD_THRESHOLD or gray.mean() < settings.BLACK_FRAME_MEAN_THRESHOLD:
            flat += 1
    reporter.record("   every photo and pair readable, none black or flat", flat == 0, f"{flat} unusable file(s)")


def check_photo_segments(reporter: Reporter, source: Path, source_duration: float, artifacts: Path) -> None:
    signatures = load_signatures(artifacts / settings.FRAMES_DIRNAME, "photo", settings.PHOTO_COUNT)
    segment_length = source_duration / settings.PHOTO_COUNT

    mismatches, margins = [], []
    for index in PHOTO_SAMPLES:
        probe = grab_signature(source, (index - 0.5) * segment_length)
        best, best_mae, second_mae = match_signature(signatures, probe)
        margins.append(second_mae - best_mae)
        if best != index:
            mismatches.append(f"segment {index} -> photo_{best:03d}")
    reporter.record(
        "3. one representative photo per equal segment",
        not mismatches,
        "; ".join(mismatches)
        or f"source midpoints map 1:1 to photo_001..photo_300 (min margin {min(margins):.2f})",
    )


def check_pair_structure(reporter: Reporter, artifacts: Path) -> None:
    if VIBE_BASES is None:
        reporter.skip("4. every pair half shows the right photo", "reference photos not available")
        return
    base_sigs = [signature_of(base) for base in VIBE_BASES]
    count = len(VIBE_BASES)

    mismatches, margins = [], []
    for index in SLOT_SAMPLES:
        path = artifacts / settings.FRAMES_DIRNAME / f"pair_{index:03d}.jpg"
        if not path.is_file():
            mismatches.append(f"pair_{index:03d}.jpg missing")
            continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            halves = (
                ("left", rgb.crop((0, 0, width // 2, height)), (2 * index - 2) % count),
                ("right", rgb.crop((width // 2, 0, width, height)), (2 * index - 1) % count),
            )
        for side, half, expected in halves:
            signature = signature_of(half)
            scores = sorted((sig_mae(signature, base), base_index) for base_index, base in enumerate(base_sigs))
            (best_mae, best), (second_mae, _) = scores[0], scores[1]
            margins.append(second_mae - best_mae)
            if best != expected:
                mismatches.append(f"pair {index} {side} shows photo {best} (expected {expected})")
    reporter.record(
        "4. every pair half shows the right photo (left 2k-1, right 2k)",
        not mismatches,
        "; ".join(mismatches)
        or f"left/right halves map 1:1 to the reference photos (min margin {min(margins):.2f})",
    )


def check_static(reporter: Reporter, final: Path, artifacts: Path) -> None:
    slot = settings.FRAME_DURATION_SECONDS

    # a) the picture inside one slot never moves (no zoom / pan / animation)
    worst = 0.0
    for index in STATIC_SLOTS:
        start = (index - 1) * slot
        early = artifacts / f"static_{index:03d}_early.png"
        late = artifacts / f"static_{index:03d}_late.png"
        grab_png(final, start + 1.0, early)
        grab_png(final, start + 6.0, late)
        worst = max(worst, sig_mae(load_rgb(early), load_rgb(late)))
    reporter.record(
        "5. photos are completely static (no zoom / pan / animation)",
        worst < STATIC_TOLERANCE_MAE,
        f"largest movement inside a slot: {worst:.2f} MAE (limit {STATIC_TOLERANCE_MAE:g})",
    )

    # b) the change between two pairs sits exactly at k x 8 s
    issues, min_switch, max_hold = [], None, 0.0
    for index in BOUNDARY_SLOTS:
        boundary = index * slot
        before = artifacts / f"boundary_{index:03d}_before.png"
        after = artifacts / f"boundary_{index:03d}_after.png"
        hold_before = artifacts / f"boundary_{index:03d}_hold_before.png"
        hold_after = artifacts / f"boundary_{index:03d}_hold_after.png"
        grab_png(final, boundary - 0.25, before)
        grab_png(final, boundary + 0.25, after)
        grab_png(final, boundary - 2.0, hold_before)
        grab_png(final, boundary + 2.0, hold_after)
        drift_before = sig_mae(load_rgb(before), load_rgb(hold_before))
        drift_after = sig_mae(load_rgb(after), load_rgb(hold_after))
        switch = sig_mae(load_rgb(before), load_rgb(after))
        max_hold = max(max_hold, drift_before, drift_after)
        min_switch = switch if min_switch is None else min(min_switch, switch)
        if drift_before >= STATIC_TOLERANCE_MAE:
            issues.append(f"slot {index} is not static ({drift_before:.1f})")
        if drift_after >= STATIC_TOLERANCE_MAE:
            issues.append(f"slot {index + 1} is not static ({drift_after:.1f})")
        if switch < BOUNDARY_MIN_MAE:
            issues.append(f"no switch at {boundary:.0f}s (frame change only {switch:.1f})")
    reporter.record(
        "   pair switches exactly at k x 8 s, perfectly static inside a slot",
        not issues,
        "; ".join(issues) or f"smallest switch {min_switch:.1f}, largest hold drift {max_hold:.2f}",
    )

    # c) slot k shows pair k (the whole 150 pairs x 8 s = 20 min sequence)
    pair_sigs = load_signatures(artifacts / settings.FRAMES_DIRNAME, "pair", settings.FRAME_COUNT)
    exact, min_margin = 0, None
    misses: list[str] = []
    for index in SLOT_SAMPLES:
        probe = artifacts / f"probe_slot_{index:03d}.png"
        grab_png(final, (index - 0.5) * slot, probe)
        with Image.open(probe) as image:
            signature = signature_of(image)
        best, best_mae, second_mae = match_signature(pair_sigs, signature)
        margin = second_mae - best_mae
        min_margin = margin if min_margin is None else min(min_margin, margin)
        if best == index:
            exact += 1
        elif (best - index) % 3 != 0:  # same-rotation pairs only differ in the border
            misses.append(f"slot {index} -> pair {best} (t={(index - 0.5) * slot:.0f}s)")
    reporter.record(
        "   slot k shows pair k (150 pairs x 8 s = 20 min)",
        not misses,
        "; ".join(misses) or f"{exact}/{len(SLOT_SAMPLES)} exact matches, smallest margin {min_margin:.2f}",
    )


def check_faces(reporter: Reporter, artifacts: Path, canvas: tuple[int, int]) -> None:
    if VIBE_BASES is None:
        reporter.skip("6. primary face centred in every half (never cut off)", "reference photos not available")
        return
    half_w, half_h = canvas[0] // 2, canvas[1]
    tolerance = max(FACE_TOLERANCE_MIN_PX, round(FACE_TOLERANCE_FRACTION * half_h))

    failures: list[str] = []
    checked, worst = 0, 0.0
    for index in SLOT_SAMPLES:
        probe = artifacts / f"probe_slot_{index:03d}.png"
        if not probe.is_file():
            failures.append(f"slot {index}: probe frame missing")
            continue
        with Image.open(probe) as image:
            rgb = image.convert("RGB")
            for side, expected in (("left", (2 * index - 2) % 3), ("right", (2 * index - 1) % 3)):
                box = (0, 0, half_w, half_h) if side == "left" else (half_w, 0, canvas[0], half_h)
                half = rgb.crop(box)
                if FACE_EXPECTED[expected]:
                    face = face_center.detect_primary_face(half)
                    if face is None:
                        failures.append(f"slot {index} {side}: no face detected")
                        continue
                    checked += 1
                    cx, cy = face.center
                    offset = max(abs(cx - half_w / 2.0), abs(cy - half_h / 2.0))
                    worst = max(worst, offset)
                    if offset > tolerance:
                        failures.append(
                            f"slot {index} {side}: face {offset:.0f}px off centre (limit {tolerance}px)"
                        )
                    if not (
                        FACE_MARGIN_PX <= face.x
                        and FACE_MARGIN_PX <= face.y
                        and face.x + face.w <= half_w - FACE_MARGIN_PX
                        and face.y + face.h <= half_h - FACE_MARGIN_PX
                    ):
                        failures.append(f"slot {index} {side}: face touches the half edge")
                else:
                    gray = np.asarray(half.convert("L"), np.float32)
                    if (
                        gray.std() < settings.FLAT_FRAME_STD_THRESHOLD
                        or gray.mean() < settings.BLACK_FRAME_MEAN_THRESHOLD
                    ):
                        failures.append(f"slot {index} {side}: fallback crop is blank")
    reporter.record(
        "6. primary face centred in every half (never cut off)",
        not failures,
        "; ".join(failures)
        or f"{checked} face halves within {tolerance}px of the half centre (worst {worst:.0f}px)",
    )


def check_durations(reporter: Reporter, final: Path, artifacts: Path, result: PipelineResult) -> None:
    expected = float(settings.SLIDESHOW_DURATION_SECONDS)
    reporter.record(
        f"7. slideshow lasts {expected / 60:.0f} minutes",
        abs(result.duration - expected) < 1e-6,
        f"pipeline reports {result.duration:.1f}s",
    )

    for name, path in (("slideshow", artifacts / settings.SLIDESHOW_FILENAME), ("final video", final)):
        duration = float((ffprobe_json(path).get("format") or {}).get("duration") or 0.0)
        reporter.record(
            f"   {name} duration",
            abs(duration - expected) <= 1.0,
            f"{duration:.2f}s (expected {expected:.0f}s +-1s)",
        )


def check_separation(reporter: Reporter, source: Path, artifacts: Path, duration: float) -> None:
    vocals = artifacts / settings.VOCALS_FILENAME
    if not vocals.is_file():
        reporter.record("8. vocals separated from the music", False, f"{vocals.name} missing")
        return

    window = max(1.5, min(4.0, duration - 3.0))
    start = min(1.5, duration * 0.1)
    mix = decode_mono(source, start=start, duration=window)
    voice = decode_mono(vocals, start=start, duration=window)

    drops, levels = [], []
    for frequency in TONE_HZ:
        mix_db = tone_level_db(mix, 16000, frequency)
        voice_db = tone_level_db(voice, 16000, frequency)
        levels.append(f"{frequency:.0f}Hz {mix_db:.1f}->{voice_db:.1f} dB")
        drops.append(mix_db - voice_db)
    strong_enough = all(level > 12.0 for level in (tone_level_db(mix, 16000, f) for f in TONE_HZ))
    reporter.record(
        "8. vocals separated from the music",
        sum(drops) / len(drops) >= TONE_ATTENUATION_DB and strong_enough,
        "; ".join(levels) + f"; mean drop {sum(drops) / len(drops):.1f} dB",
    )

    correlation = float(np.corrcoef(mix, voice)[0, 1])
    rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(voice ** 2))), 1e-9))
    reporter.record(
        "   vocals contain the voice (not silence, not the music)",
        correlation > 0.15 and -45.0 < rms_db < -6.0,
        f"correlation with the mix {correlation:.2f}, vocal level {rms_db:.1f} dBFS",
    )


def check_overlay(
    reporter: Reporter,
    final: Path,
    artifacts: Path,
    canvas: tuple[int, int],
    position: str,
) -> None:
    prepared = artifacts / "overlay_prepared.png"
    if not prepared.is_file():
        reporter.record("9. overlay appears correctly", False, "overlay_prepared.png missing")
        return
    with Image.open(prepared) as image:
        size = image.size
        has_alpha = "A" in image.getbands()
    reporter.record(
        "9. overlay prepared without stretching (alpha kept)",
        size == OVERLAY_SIZE and has_alpha,
        f"{size[0]}x{size[1]} (input {OVERLAY_SIZE[0]}x{OVERLAY_SIZE[1]}), alpha={has_alpha}",
    )

    x, y = overlay_module.position_xy(position, canvas[0], canvas[1], size[0], size[1], settings.OVERLAY_MARGIN)
    expected = (x, y, x + size[0] // 2 - 1, y + size[1] - 1)

    probe = artifacts / "probe_overlay.png"
    grab_png(final, 100.0, probe)  # t=100s -> slot 13 -> pair_013
    try:
        picture = load_rgb(probe)
    except Exception:  # noqa: BLE001 - ffmpeg produced nothing
        reporter.record("   opaque half sits at " + position + " + margin", False, f"no frame at t=100s in {final.name}")
        return

    # differential check: compare the encoded frame with the raw pair picture.
    # The overlay area is the only place where they differ strongly, so photo
    # content (red clothes, warm hues, ...) cannot pollute the bounding box
    # the way a colour mask would.
    underlying_path = artifacts / settings.FRAMES_DIRNAME / "pair_013.jpg"
    underlying = load_rgb(underlying_path) if underlying_path.is_file() else None
    if underlying is not None:
        delta = np.abs(picture - underlying).mean(axis=2)
        strong = delta > 40.0
        if strong.any():
            ys, xs = np.nonzero(strong)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            coverage = float(strong[expected[1] : expected[3] + 1, expected[0] : expected[2] + 1].mean())
        else:
            bbox, coverage = None, 0.0
        reporter.record(
            f"   opaque half sits at {position} + margin",
            bbox is not None
            and all(abs(a - b) <= 6 for a, b in zip(bbox, expected))
            and coverage > 0.8,
            f"changed area {bbox} covering {coverage * 100:.0f}% of the expected rect, expected {expected}",
        )
    else:
        reporter.skip(f"   opaque half sits at {position} + margin", "pair_013.jpg missing")

    crop = picture[y + 10 : y + 80, x + 10 : x + 110].reshape(-1, 3)
    if crop.size:
        mean = crop.mean(axis=0)
        reporter.record(
            "   overlay colour intact",
            mean[0] > 150 and mean[1] < 90 and mean[2] < 90,
            f"mean RGB {[round(float(c), 1) for c in mean]}",
        )

    # the transparent half must show the pair picture underneath (pair 13 at t=100s)
    if underlying is not None:
        hole = picture[y + 10 : y + 80, x + size[0] // 2 + 10 : x + size[0] - 10]
        reference = underlying[y + 10 : y + 80, x + size[0] // 2 + 10 : x + size[0] - 10]
        mae = sig_mae(hole, reference)
        reporter.record(
            "   transparent area keeps the pair picture (alpha preserved)",
            mae < 30.0,
            f"mean abs difference {mae:.1f} vs the raw pair",
        )
    else:
        reporter.skip("   transparent area keeps the pair picture (alpha preserved)", "pair_013.jpg missing")

    visible = True
    for timestamp in (4.0, 1195.0):  # first and last slot
        shot = artifacts / f"probe_t{int(timestamp)}.png"
        grab_png(final, timestamp, shot)
        pixel = load_rgb(shot)[y + 20, x + 20]
        if not (pixel[0] > 170 and pixel[0] - pixel[1] > 90 and pixel[0] - pixel[2] > 90):
            visible = False
    reporter.record("   overlay visible from the first to the last slot", visible)


def check_streams(reporter: Reporter, final: Path, canvas: tuple[int, int]) -> None:
    data = ffprobe_json(final)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    reporter.record(
        "10. final MP4 contains video + audio",
        video is not None and audio is not None,
        f"video={video.get('codec_name') if video else '-'}, audio={audio.get('codec_name') if audio else '-'}",
    )
    if video is not None:
        reporter.record(
            "   H.264 / yuv420p at the canvas size",
            video.get("codec_name") == "h264"
            and video.get("pix_fmt") == "yuv420p"
            and (int(video.get("width", 0)), int(video.get("height", 0))) == canvas,
            f"{video.get('codec_name')} {video.get('pix_fmt')} {video.get('width')}x{video.get('height')}",
        )
        fps = float(video.get("avg_frame_rate", "0/1").split("/")[0]) / max(
            1.0, float(video.get("avg_frame_rate", "0/1").split("/")[1])
        )
        reporter.record(
            "   frame rate matches SLIDESHOW_FPS",
            abs(fps - settings.SLIDESHOW_FPS) < 0.1,
            f"{fps:.2f} fps (expected {settings.SLIDESHOW_FPS})",
        )
    if audio is not None:
        reporter.record(
            "   AAC audio at the configured rate",
            audio.get("codec_name") == "aac"
            and int(audio.get("sample_rate", 0)) == settings.AUDIO_SAMPLE_RATE
            and int(audio.get("channels", 0)) == settings.AUDIO_CHANNELS,
            f"{audio.get('codec_name')} {audio.get('sample_rate')}Hz {audio.get('channels')}ch",
        )


def check_telegram(reporter: Reporter, final: Path) -> None:
    size = final.stat().st_size
    limit = settings.effective_upload_limit()
    reporter.record(
        "11. output acceptable for Telegram",
        size <= limit,
        f"{human_size(size)} (limit {human_size(limit)})",
    )
    head = final.open("rb").read(2 * 1024 * 1024)
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    reporter.record(
        "   faststart (moov before mdat)",
        moov != -1 and (mdat == -1 or moov < mdat),
        f"moov@{moov}, mdat@{mdat}",
    )


def check_cleanup(reporter: Reporter, job_dir: Path, spec: JobSpec, keep: bool) -> None:
    if not settings.DELETE_TEMP_AFTER_SEND:
        reporter.skip("12. temporary files removed", "DELETE_TEMP_AFTER_SEND is off")
        return

    leftovers = [
        path.name
        for path in (
            job_dir / settings.FRAMES_DIRNAME,
            job_dir / settings.SLIDESHOW_FILENAME,
            job_dir / settings.VOCALS_FILENAME,
            job_dir / settings.AUDIO_FILENAME,
            job_dir / "overlay_prepared.png",
        )
        if path.exists()
    ]
    reporter.record(
        "12. intermediates removed after the run",
        not leftovers and (job_dir / settings.FINAL_FILENAME).is_file(),
        f"still there: {', '.join(leftovers)}" if leftovers else "only final_video.mp4 remains",
    )

    if keep:
        reporter.skip("   whole job directory removed (as the bot does after sending)", "--keep")
        return

    # this is exactly what the bot's JobManager does after the video was sent
    from bot.jobs import JobHandle, JobManager

    manager = JobManager()
    handle = JobHandle(job_id=spec.job_id, user_id=0, chat_id=0, spec=spec)
    manager._job_dirs[spec.job_id] = job_dir  # noqa: SLF001 - test of the real cleanup path
    manager._cleanup_job_dir(handle)  # noqa: SLF001
    reporter.record(
        "   whole job directory removed after delivery",
        not job_dir.exists(),
        str(job_dir),
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    global CANVAS, VIBE_BASES  # noqa: PLW0603 - CLI override / lazy reference photos
    parser = argparse.ArgumentParser(description="End-to-end pipeline check (spec section 23).")
    parser.add_argument(
        "--duration",
        type=float,
        default=80.0,
        help="source video length in seconds (default 80; keep >= 70 so every slide spans >= 7 encoded frames)",
    )
    parser.add_argument("--keep", action="store_true", help="keep every file for inspection")
    parser.add_argument(
        "--source-size",
        default="640x360",
        metavar="WxH",
        help="resolution of the synthetic source video (default %(default)s); "
        "pass 1280x720 to exercise the 720p output canvas",
    )
    parser.add_argument(
        "--position",
        default="bottom-right",
        choices=settings.OVERLAY_POSITIONS,
        help="overlay position to test (default %(default)s; keeps the overlay away from the faces)",
    )
    args = parser.parse_args(argv)
    if args.duration < settings.MIN_SOURCE_DURATION_SECONDS:
        parser.error(f"--duration must be at least {settings.MIN_SOURCE_DURATION_SECONDS:.0f} seconds")
    try:
        width, height = (int(part) for part in args.source_size.lower().split("x", 1))
    except ValueError:
        parser.error(f"--source-size must look like WxH (e.g. 640x360), got {args.source_size!r}")
    if width < 64 or height < 64 or width % 2 or height % 2:
        parser.error("--source-size needs even dimensions of at least 64x64")
    CANVAS = (width, height)
    VIBE_BASES = load_vibe_bases()

    setup_logging()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = settings.TEMP_DIRECTORY / f"e2e_{stamp}"
    job_dir = root / "job"
    artifacts = root / "artifacts"
    job_dir.mkdir(parents=True)
    artifacts.mkdir(parents=True)

    source = job_dir / "source.mp4"
    overlay_input = job_dir / "overlay_input.png"
    spec = JobSpec(
        job_id=f"e2e_{stamp}",
        user_id=0,
        chat_id=0,
        source_path=source,
        work_dir=job_dir,
        overlay_path=overlay_input,
        overlay_position=args.position,
        overlay_size_label="original",
    )

    print(f"e2e check - model {settings.MODEL_NAME}, device {settings.DEMUCS_DEVICE}")
    print(f"  work directory: {root}")
    frames_per_slide = 30.0 * args.duration / settings.PHOTO_COUNT
    if frames_per_slide < 7.0:
        print(
            f"  note: a {args.duration:.0f}s source gives only {frames_per_slide:.1f} frames per slide - "
            "durations below ~70s can mis-align the extractor's late offsets; 80s is the safe default"
        )
    print(f"  building a {args.duration:.0f}s synthetic source video ...")
    build_source_video(source, args.duration, root)
    build_overlay(overlay_input)

    info = video_info.probe_video(source)
    canvas = video_info.compute_output_dimensions(info)
    print(f"  source: {info.resolution_label} {info.duration:.1f}s -> canvas {canvas[0]}x{canvas[1]}")

    reporter = Reporter()
    snapshots = {"copies": 0}

    def progress_cb(overall: float, stage: int, fraction: float, detail: str) -> None:
        if stage != progress_cb.last_stage or overall * 100 >= progress_cb.last_percent + 5:
            progress_cb.last_stage = stage
            progress_cb.last_percent = int(overall * 100)
            print(f"    step {stage}/5  {overall * 100:5.1f}%  {detail}".rstrip())
        if stage >= 3:  # grab the original audio before the pipeline deletes it
            audio = job_dir / settings.AUDIO_FILENAME
            if audio.exists() and not (artifacts / audio.name).exists():
                shutil.copy2(audio, artifacts / audio.name)
        if stage == 5 and fraction >= 1.0:
            # snapshot everything (photos, pairs, slideshow, vocals, final video)
            # before the pipeline cleans up - the checks below run on this copy
            shutil.copytree(job_dir, artifacts, dirs_exist_ok=True)
            snapshots["copies"] += 1

    progress_cb.last_stage = -1  # type: ignore[attr-defined]
    progress_cb.last_percent = -5  # type: ignore[attr-defined]

    print("\nrunning the pipeline")
    try:
        result = run_pipeline(spec, progress_cb=progress_cb)
    except Exception as exc:  # noqa: BLE001 - report, keep the files, fail
        print(f"\npipeline failed: {type(exc).__name__}: {exc}")
        print(f"files kept in {root}")
        return 1

    print(f"\nchecks  (job {spec.job_id}, {snapshots['copies']} snapshot(s))")
    check_selection(reporter, artifacts, canvas, result)
    check_photo_segments(reporter, source, info.duration, artifacts)
    check_pair_structure(reporter, artifacts)
    check_static(reporter, result.final_path, artifacts)
    check_faces(reporter, artifacts, canvas)
    check_durations(reporter, result.final_path, artifacts, result)
    check_separation(reporter, source, artifacts, info.duration)
    check_overlay(reporter, result.final_path, artifacts, canvas, args.position)
    check_streams(reporter, result.final_path, canvas)
    check_telegram(reporter, result.final_path)
    check_cleanup(reporter, job_dir, spec, args.keep)

    print("\ntimings")
    for stage, seconds in sorted(result.stage_seconds.items()):
        print(f"  stage {stage}/5: {seconds:6.1f}s")
    print(f"  total:     {result.elapsed_seconds:6.1f}s")
    print(f"  photo report: {result.frames_report.summary()}")

    failed = reporter.failed
    print(f"\n{'ALL CHECKS PASSED' if failed == 0 else f'{failed} CHECK(S) FAILED'}")
    if failed == 0 and not args.keep:
        shutil.rmtree(root, ignore_errors=True)
        print("artifacts removed (re-run with --keep to inspect them)")
    else:
        print(f"files kept in {root}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
