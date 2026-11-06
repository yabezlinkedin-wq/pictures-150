"""
Exception hierarchy for the processing pipeline.

Every exception carries ``user_message`` - a short, non-technical sentence that
is safe to show to the Telegram user.  Technical details stay in the logs.
"""

from __future__ import annotations


class ProcessingError(Exception):
    """Base class for all expected processing failures."""

    default_user_message = "Processing failed. Please try another video."

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or self.default_user_message


class ToolMissingError(ProcessingError):
    default_user_message = (
        "The video processing service is not configured correctly. Please try again later."
    )


class FFmpegError(ProcessingError):
    default_user_message = "Video processing failed. Please try another video."


class VideoInfoError(ProcessingError):
    default_user_message = (
        "Sorry, this file could not be read as a video. Please send a valid video file."
    )


class NoAudioStreamError(ProcessingError):
    default_user_message = "Sorry, this video does not contain an audio track."


class VideoTooShortError(ProcessingError):
    default_user_message = "Sorry, this video is too short to be processed."


class FrameExtractionError(ProcessingError):
    default_user_message = (
        "Sorry, frames could not be extracted from this video. Please try another video."
    )


class SlideshowError(ProcessingError):
    default_user_message = "Sorry, the slideshow could not be generated. Please try again."


class PictureExportError(ProcessingError):
    default_user_message = (
        "Sorry, the 150 pictures could not be prepared. Please try again."
    )


class AudioExtractionError(ProcessingError):
    default_user_message = "Sorry, the audio of this video could not be processed."


class SeparationError(ProcessingError):
    default_user_message = (
        "Sorry, the voice separation service is currently unavailable. Please try again later."
    )


class OverlayError(ProcessingError):
    default_user_message = (
        "Sorry, the overlay image could not be processed. "
        "Please send a PNG, JPG or WebP image."
    )


class RenderError(ProcessingError):
    default_user_message = "Sorry, the final video could not be rendered. Please try again."


class OutputTooLargeError(ProcessingError):
    default_user_message = (
        "The final video is too large to be sent through Telegram. Please try a shorter video."
    )


class InsufficientDiskSpaceError(ProcessingError):
    default_user_message = (
        "The server is low on disk space right now. Please try again later."
    )


class JobTimeoutError(ProcessingError):
    default_user_message = "Processing took too long and was cancelled. Please try a shorter video."


class JobCancelledError(ProcessingError):
    default_user_message = "Processing was cancelled. Send a video whenever you are ready."


class TelegramIOError(ProcessingError):
    default_user_message = (
        "Sorry, the file could not be transferred through Telegram. Please try again."
    )
