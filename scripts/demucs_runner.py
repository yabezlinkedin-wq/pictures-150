"""
Wrapper around ``demucs.separate`` used by :mod:`processing.vocal_separator`.

Why a wrapper?

* it guarantees the model cache lives inside the project (``models/``),
* it applies a small compatibility shim for recent PyTorch releases
  (PyTorch >= 2.6 changed the default of ``torch.load(weights_only=...)``,
  which breaks older Demucs checkpoints - they are trusted local weights),
* all CLI arguments are passed through unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# importing settings sets TORCH_HOME / HF_HOME so weights stay in models/
from config import settings  # noqa: F401,E402  (import for side effects)

import torch  # noqa: E402

_original_load = torch.load


def _patched_load(*args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load  # type: ignore[assignment]

from demucs.separate import main  # noqa: E402

if __name__ == "__main__":
    main()
