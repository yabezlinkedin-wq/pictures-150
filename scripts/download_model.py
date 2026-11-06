"""
Pre-download the vocal-separation model weights.

Called at Docker build time and usable as a warm-up step in a manual install::

    python scripts/download_model.py                        # -> models/
    python scripts/download_model.py --target /opt/cache     # Docker build
    python scripts/download_model.py --model htdemucs_ft

The weights are cached in ``<target>/hub`` (the HuggingFace cache directory
used by Demucs 4.x), so every later run - and every job - reuses them and
never has to wait for the download again.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Demucs weights used for vocal separation."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="directory that should hold the weights "
        "(default: MODELS_DIRECTORY from the environment, normally ./models)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model name (default: MODEL_NAME from the environment, normally htdemucs)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ``config.settings`` reads the environment at import time, so all
    # overrides have to be applied before it is imported.
    if args.target is not None:
        target = args.target.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        os.environ["MODELS_DIRECTORY"] = str(target)
        os.environ["TORCH_HOME"] = str(target)
        os.environ["HF_HOME"] = str(target / "hub")
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    from config import settings
    from config.logging_setup import setup_logging

    setup_logging()

    from processing import vocal_separator
    from processing.errors import SeparationError

    try:
        vocal_separator.download_model()
    except (SeparationError, ImportError) as exc:
        print(f"error: could not download the model: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - network/disk problems, show them plainly
        print(f"error: model download failed: {exc}", file=sys.stderr)
        return 1

    print(f"model '{settings.MODEL_NAME}' is ready (cache: {settings.MODELS_DIRECTORY})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
