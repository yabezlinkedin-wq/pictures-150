"""
Application wiring.

Creates the :class:`telegram.ext.Application`, registers every handler and
starts polling.  Two startup modes are supported and selected through the
environment (see ``.env.example``):

* official Bot API (``https://api.telegram.org``) - 20 MB downloads / 50 MB
  uploads,
* a local Bot API server (``TELEGRAM_API_BASE_URL=http://localhost:8081/bot``)
  - up to 2000 MB, which is what the 20-minute output normally needs.
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from bot import handlers
from bot.jobs import job_manager
from config import settings
from config.logging_setup import setup_logging
from processing.ffmpeg_utils import ensure_tools_available

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# application
# --------------------------------------------------------------------------- #
async def _post_init(application: Application) -> None:
    await job_manager.start(application.bot)
    try:
        me = await application.bot.get_me()
        log.info("connected to Telegram as @%s (id %s)", me.username, me.id)
    except Exception as exc:  # noqa: BLE001 - informational only
        log.warning("could not call getMe: %s", exc)
    log.info("configuration:\n%s", settings.summary())


async def _post_shutdown(application: Application) -> None:
    await job_manager.stop()


def _register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", handlers.cmd_start))
    application.add_handler(CommandHandler("help", handlers.cmd_help))
    application.add_handler(CommandHandler("status", handlers.cmd_status))
    application.add_handler(CommandHandler("cancel", handlers.cmd_cancel))
    application.add_handler(CommandHandler("skip", handlers.cmd_skip))

    application.add_handler(CallbackQueryHandler(handlers.on_callback, pattern=r"^ov:"))

    application.add_handler(
        MessageHandler(filters.VIDEO | filters.ANIMATION, handlers.on_video)
    )
    application.add_handler(MessageHandler(filters.Document.ALL, handlers.on_document))
    application.add_handler(MessageHandler(filters.PHOTO, handlers.on_photo))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.on_text)
    )

    application.add_error_handler(handlers.on_error)


def build_application() -> Application:
    builder = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN)

    if settings.TELEGRAM_API_BASE_URL:
        log.info("using Bot API server at %s (local mode)", settings.TELEGRAM_API_BASE_URL)
        builder = builder.base_url(settings.TELEGRAM_API_BASE_URL).local_mode(True)
        if settings.TELEGRAM_BASE_FILE_URL:
            builder = builder.base_file_url(settings.TELEGRAM_BASE_FILE_URL)

    builder = (
        builder.connect_timeout(settings.TELEGRAM_CONNECT_TIMEOUT)
        .read_timeout(settings.TELEGRAM_READ_TIMEOUT)
        .write_timeout(settings.TELEGRAM_WRITE_TIMEOUT)
        .pool_timeout(settings.TELEGRAM_POOL_TIMEOUT)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
    )

    application = builder.build()
    _register_handlers(application)
    return application


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    setup_logging()

    problems = settings.validate()
    fatals = [problem for problem in problems if "TELEGRAM_BOT_TOKEN" in problem]
    if fatals:
        for problem in problems:
            log.error("configuration problem: %s", problem)
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is missing - copy .env.example to .env and fill it in."
        )
    for problem in problems:
        log.warning("configuration problem: %s", problem)

    settings.ensure_directories()
    ensure_tools_available()  # ffmpeg/ffprobe must exist before we accept videos

    application = build_application()
    log.info("starting long polling (Ctrl+C / SIGTERM to stop)")
    # Default stop_signals (SIGINT/SIGTERM/SIGABRT) are kept so that ``docker stop``
    # triggers a graceful shutdown - that is what runs ``_post_shutdown`` and
    # cleans up any job directories that are still on disk.
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
