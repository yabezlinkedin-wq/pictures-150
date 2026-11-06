"""Browser interface for the video tool (FastAPI).

Same processing pipeline as the Telegram bot - ``processing.pipeline`` - but
the video is uploaded and downloaded directly through the browser, so the
Telegram Bot API size limits do not apply.

Entry point::

    python -m web.app
"""
