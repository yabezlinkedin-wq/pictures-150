"""Logging configuration shared by the bot and the stand-alone scripts."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from config import settings

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s | %(message)s"
_LOG_MAX_BYTES = 10 * 1024 * 1024
_LOG_BACKUPS = 3


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.LOG_FILE:
        try:
            settings.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
            handlers.append(
                logging.handlers.RotatingFileHandler(
                    settings.LOG_FILE,
                    maxBytes=_LOG_MAX_BYTES,
                    backupCount=_LOG_BACKUPS,
                    encoding="utf-8",
                )
            )
        except OSError as exc:  # logging must never crash startup
            print(f"warning: could not open the log file {settings.LOG_FILE}: {exc}", file=sys.stderr)

    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)

    # Third-party chatter we do not want at INFO level
    for noisy in ("httpx", "httpcore", "urllib3", "telegram.request", "telegram.ext.Updater"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
