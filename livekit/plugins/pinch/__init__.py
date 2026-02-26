"""livekit-plugins-pinch — real-time speech-to-speech translation via Pinch."""

from .models import TranscriptEvent, TranslatorOptions
from .translator import (
    Translator,
    PinchAuthError,
    PinchError,
    PinchRateLimitError,
    PinchSessionError,
)
from .version import __version__

__all__ = [
    # Core classes
    "Translator",
    "TranslatorOptions",
    "TranscriptEvent",
    # Exceptions
    "PinchError",
    "PinchAuthError",
    "PinchRateLimitError",
    "PinchSessionError",
    # Version
    "__version__",
]
