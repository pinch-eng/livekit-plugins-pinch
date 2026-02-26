# livekit-plugins-pinch

An official-style **LiveKit Agents** plugin that wraps the
[Pinch](https://www.startpinch.com) real-time **speech-to-speech translation** API.

---

## How it works

Pinch provides a real-time speech-to-speech translation pipeline. When you
create a session it gives back a dedicated **LiveKit room** where its
translation agent lives. This plugin:
1. Calls the Pinch REST API to create a translation session.
2. Connects to the Pinch LiveKit room.
3. Forwards the user's audio from your existing LiveKit room into the Pinch
   room.
4. Subscribes to the translated audio track published by the Pinch agent and
   re-publishes it back into your room.
5. Surfaces transcript events (both source-language recognition and
   translated text) via a simple callback API.

All wiring is handled internally — your pipeline just calls `start()` and
`stop()`.

---

## Installation

```bash
pip install livekit-plugins-pinch
```

### Dependencies

| Package | Purpose |
|---|---|
| `livekit` | RTC SDK — room/track management |
| `livekit-agents` | Base plugin interfaces |
| `aiohttp` | Async HTTP for session creation |

Python **≥ 3.9** is required.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `PINCH_API_KEY` | **Yes** | Your Pinch API key. Obtain one at [startpinch.com](https://www.startpinch.com). |
| `LIVEKIT_URL` | Yes (your app) | `wss://` URL for your LiveKit Cloud project. |
| `LIVEKIT_TOKEN` | Yes (your app) | A valid participant JWT for your LiveKit room. |

---

## Quick start

```python
import asyncio
from livekit import rtc
from livekit.plugins.pinch import Translator, TranslatorOptions, TranscriptEvent

async def main():
    # Connect to your LiveKit room
    room = rtc.Room()
    await room.connect(LIVEKIT_URL, LIVEKIT_TOKEN)

    # Configure and start the Pinch translator
    translator = Translator(
        options=TranslatorOptions(
            source_language="en-US",   # BCP-47 source language hint
            target_language="es-ES",   # BCP-47 target language
            voice_type="clone",        # "clone" | "female" | "male"
        )
        # api_key is read from PINCH_API_KEY automatically
    )

    @translator.on_transcript
    def handle(event: TranscriptEvent) -> None:
        label = "ORIGINAL" if event.is_original else "TRANSLATED"
        if event.is_final:
            print(f"[{label}] {event.text}")

    await translator.start(room)

    # … run your agent / application logic …

    await translator.stop()
    await room.disconnect()
```

---

## API reference

### `TranslatorOptions`

```python
@dataclass
class TranslatorOptions:
    source_language: str        # e.g. "en-US"
    target_language: str        # e.g. "es-ES"
    voice_type: str = "clone"   # "clone" | "female" | "male"
```

### `TranscriptEvent`

```python
@dataclass
class TranscriptEvent:
    type: str               # "original_transcript" | "translated_transcript"
    text: str               # recognised or translated text
    is_final: bool          # True once the segment is committed
    language_detected: str  # BCP-47 code of the detected language
    timestamp: float        # Unix timestamp (seconds) from Pinch
    confidence: float       # 0–1 confidence score (0 if not provided)

    # Convenience properties
    is_original: bool       # True when type == "original_transcript"
    is_translated: bool     # True when type == "translated_transcript"
```

### `Translator`

```python
class Translator:
    def __init__(
        self,
        *,
        options: TranslatorOptions,
        api_key: str | None = None,   # falls back to PINCH_API_KEY env var
    ) -> None: ...

    async def start(self, source_room: rtc.Room) -> None: ...
    async def stop(self) -> None: ...

    def on_transcript(
        self, callback: Callable[[TranscriptEvent], None]
    ) -> Callable[[TranscriptEvent], None]: ...

    def remove_transcript_listener(
        self, callback: Callable[[TranscriptEvent], None]
    ) -> None: ...
```

#### `start(source_room)`

Kicks off the full pipeline:

- Creates a Pinch session via the REST API.
- Connects to the Pinch LiveKit room (with up to 3 exponential-backoff retries).
- Publishes the user's mic audio (from `source_room`) to the Pinch room.
- Subscribes to the Pinch agent's translated audio and re-publishes it into
  `source_room` as the `"pinch-translated"` track.
- Registers the data-channel listener for transcript events.

#### `stop()`

Cancels all background tasks, unpublishes tracks, and cleanly disconnects from
the Pinch room. Safe to call multiple times.

#### `on_transcript(callback)`

Registers a transcript callback. Works both as a regular call **and** as a
decorator:

```python
# Decorator
@translator.on_transcript
def handle(ev: TranscriptEvent) -> None:
    print(ev.text)

# Direct call
translator.on_transcript(my_handler)
```

---

## Error handling

| Exception | Cause |
|---|---|
| `PinchAuthError` | Invalid or missing API key (HTTP 401) |
| `PinchRateLimitError` | API rate limit exceeded (HTTP 429) |
| `PinchSessionError` | Session creation failed, server error, or room connection failed after max retries |
| `PinchError` | Base class for all Pinch exceptions |

---

## Supported languages

Pinch supports a wide range of languages and dialects. For the full list of
BCP-47 language codes accepted by `source_language` and `target_language`, see
[Pinch supported languages](https://www.startpinch.com/docs/supported-languages).

---

## Running the example

```bash
export PINCH_API_KEY="pk_..."
export LIVEKIT_URL="wss://my-project.livekit.cloud"
export LIVEKIT_TOKEN="eyJhbGci..."

# Optional overrides
export SOURCE_LANGUAGE="en-US"
export TARGET_LANGUAGE="es-ES"
export VOICE_TYPE="clone"

python examples/basic_translation.py
```

---

## Running the tests

```bash
pip install pytest pytest-asyncio
pytest -v tests/
```

---

## Development install

```bash
git clone https://github.com/pinch-eng/livekit-plugins-pinch
cd livekit-plugins-pinch
pip install -e ".[dev]"
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
