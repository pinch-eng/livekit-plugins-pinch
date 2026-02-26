"""Pinch real-time speech-to-speech translation plugin for LiveKit Agents."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Callable, List, Optional

import aiohttp
from livekit import rtc

from .models import TranscriptEvent, TranslatorOptions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PINCH_SESSION_URL = "https://www.startpinch.com/api/session"

# Standard LiveKit/WebRTC audio parameters used throughout the bridge.
_SAMPLE_RATE = 48_000
_NUM_CHANNELS = 1

# Reconnection policy for the Pinch room connection.
_MAX_CONNECT_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds (doubled on each subsequent attempt)

# How long to wait for the Pinch translation agent to publish an audio track
# before giving up (seconds).
_AGENT_TRACK_TIMEOUT = 30.0
_AGENT_TRACK_POLL_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PinchError(Exception):
    """Base class for all Pinch plugin errors."""


class PinchAuthError(PinchError):
    """Raised when the API key is invalid or missing (HTTP 401)."""


class PinchRateLimitError(PinchError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""


class PinchSessionError(PinchError):
    """Raised when session creation or room connection fails."""


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class Translator:
    """Wraps the Pinch real-time speech-to-speech translation API.

    Typical usage inside a LiveKit Agents pipeline::

        from livekit.plugins.pinch import Translator, TranslatorOptions

        translator = Translator(
            options=TranslatorOptions(
                source_language="en-US",
                target_language="es-ES",
                voice_type="clone",
            )
        )

        @translator.on_transcript
        def handle(event):
            if event.is_translated and event.is_final:
                print(f"[{event.language_detected}] {event.text}")

        await translator.start(agent_room)
        # … run your agent …
        await translator.stop()

    Parameters
    ----------
    api_key:
        Pinch API key.  Falls back to the ``PINCH_API_KEY`` environment
        variable when not supplied.
    options:
        :class:`~livekit.plugins.pinch.TranslatorOptions` describing the
        source/target languages and voice type.
    """

    def __init__(
        self,
        *,
        options: TranslatorOptions,
        api_key: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("PINCH_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Pinch API key is required. Supply it via the `api_key` "
                "argument or set the PINCH_API_KEY environment variable."
            )
        self._api_key: str = resolved_key
        self._options: TranslatorOptions = options

        # Rooms
        self._source_room: Optional[rtc.Room] = None
        self._pinch_room: Optional[rtc.Room] = None

        # Audio sources used to publish into each room
        self._pinch_audio_source: Optional[rtc.AudioSource] = None   # user → Pinch
        self._source_audio_source: Optional[rtc.AudioSource] = None  # Pinch → user

        # Local tracks published into each room
        self._pinch_local_track: Optional[rtc.LocalAudioTrack] = None
        self._source_local_track: Optional[rtc.LocalAudioTrack] = None

        # Track publication objects (needed for un-publishing)
        self._pinch_publication: Optional[rtc.LocalTrackPublication] = None
        self._source_publication: Optional[rtc.LocalTrackPublication] = None

        # Background asyncio tasks
        self._tasks: List[asyncio.Task] = []

        # Transcript callbacks registered via on_transcript()
        self._transcript_callbacks: List[Callable[[TranscriptEvent], None]] = []

        # Internal state
        self._started = False
        self._stopped = False

    # ------------------------------------------------------------------
    # Public API — transcript callbacks
    # ------------------------------------------------------------------

    def on_transcript(
        self, callback: Callable[[TranscriptEvent], None]
    ) -> Callable[[TranscriptEvent], None]:
        """Register a callback for transcript events.

        Can be used as a decorator or called directly::

            @translator.on_transcript
            def handle(event: TranscriptEvent) -> None:
                print(event.text)

            # or:
            translator.on_transcript(my_handler)

        The callback is invoked for both ``"original_transcript"`` and
        ``"translated_transcript"`` message types.

        Parameters
        ----------
        callback:
            Callable that accepts a single :class:`TranscriptEvent`.

        Returns
        -------
        callback
            The same callable, so the method works as a decorator.
        """
        self._transcript_callbacks.append(callback)
        return callback

    def remove_transcript_listener(
        self, callback: Callable[[TranscriptEvent], None]
    ) -> None:
        """Remove a previously registered transcript callback."""
        try:
            self._transcript_callbacks.remove(callback)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Public API — lifecycle
    # ------------------------------------------------------------------

    async def start(self, source_room: rtc.Room) -> None:
        """Start the Pinch translation pipeline.

        1. Creates a Pinch session via the REST API.
        2. Connects to the Pinch LiveKit room.
        3. Publishes the user's audio from *source_room* into the Pinch room.
        4. Forwards translated audio from the Pinch room back into *source_room*.
        5. Listens on the Pinch data channel for transcript events.

        Parameters
        ----------
        source_room:
            The :class:`livekit.rtc.Room` that the user is connected to.
            The plugin subscribes to any audio tracks published there and
            bridges them into the Pinch room for translation.

        Raises
        ------
        PinchAuthError
            If the API key is rejected.
        PinchRateLimitError
            If the API rate limit is exceeded.
        PinchSessionError
            If session creation or room connection fails after retries.
        RuntimeError
            If :meth:`start` is called after :meth:`stop`.
        """
        if self._stopped:
            raise RuntimeError(
                "This Translator instance has already been stopped. "
                "Create a new Translator to start a fresh session."
            )
        if self._started:
            logger.warning("Translator.start() called more than once; ignoring.")
            return

        self._started = True
        self._source_room = source_room

        logger.info(
            "Pinch Translator starting  src=%s  tgt=%s  voice=%s",
            self._options.source_language,
            self._options.target_language,
            self._options.voice_type,
        )

        # 1. REST – create Pinch session ----------------------------------------
        session = await self._create_session()
        pinch_url: str = session["url"]
        pinch_token: str = session["token"]
        pinch_room_name: str = session["room_name"]
        logger.info("Pinch session created  room=%s", pinch_room_name)

        # 2. Connect to Pinch LiveKit room with retry ----------------------------
        self._pinch_room = await self._connect_with_retry(pinch_url, pinch_token)
        logger.info("Connected to Pinch room  room=%s", pinch_room_name)

        # 3. Publish user audio source into the Pinch room ----------------------
        self._pinch_audio_source = rtc.AudioSource(
            sample_rate=_SAMPLE_RATE, num_channels=_NUM_CHANNELS
        )
        self._pinch_local_track = rtc.LocalAudioTrack.create_audio_track(
            "pinch-input", self._pinch_audio_source
        )
        self._pinch_publication = (
            await self._pinch_room.local_participant.publish_track(
                self._pinch_local_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
        )
        logger.debug("Published user audio input into Pinch room.")

        # 4. Prepare audio source to publish translated audio back to source_room
        self._source_audio_source = rtc.AudioSource(
            sample_rate=_SAMPLE_RATE, num_channels=_NUM_CHANNELS
        )
        self._source_local_track = rtc.LocalAudioTrack.create_audio_track(
            "pinch-translated", self._source_audio_source
        )
        self._source_publication = (
            await self._source_room.local_participant.publish_track(
                self._source_local_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
        )
        logger.debug("Published translated audio output into source room.")

        # 5. Register data channel handler on the Pinch room --------------------
        self._pinch_room.on("data_received", self._on_pinch_data_received)

        # 6. Start background tasks ---------------------------------------------
        # Forward source_room mic → Pinch room
        self._tasks.append(
            asyncio.ensure_future(self._run_source_to_pinch_bridge())
        )
        # Forward Pinch translated audio → source_room
        self._tasks.append(
            asyncio.ensure_future(self._run_pinch_to_source_bridge())
        )

        logger.info("Pinch Translator started successfully.")

    async def stop(self) -> None:
        """Cleanly shut down the translation pipeline.

        Cancels all background tasks, unpublishes tracks, and disconnects
        from the Pinch room.  Safe to call multiple times.
        """
        if self._stopped:
            return
        self._stopped = True
        logger.info("Stopping Pinch Translator…")

        # Remove event handlers
        if self._pinch_room is not None:
            try:
                self._pinch_room.off("data_received", self._on_pinch_data_received)
            except Exception:
                pass

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Unpublish & disconnect Pinch room
        if self._pinch_room is not None:
            if self._pinch_publication is not None:
                try:
                    await self._pinch_room.local_participant.unpublish_track(
                        self._pinch_publication.sid
                    )
                except Exception:
                    pass
            try:
                await self._pinch_room.disconnect()
            except Exception:
                pass
            self._pinch_room = None

        # Unpublish translated track from source room
        if self._source_room is not None and self._source_publication is not None:
            try:
                await self._source_room.local_participant.unpublish_track(
                    self._source_publication.sid
                )
            except Exception:
                pass

        # Release audio sources
        self._pinch_audio_source = None
        self._source_audio_source = None

        logger.info("Pinch Translator stopped.")

    # ------------------------------------------------------------------
    # Internal – session creation
    # ------------------------------------------------------------------

    async def _create_session(self) -> dict:
        """POST /api/session and return the parsed JSON response."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "sourceLanguage": self._options.source_language,
            "targetLanguage": self._options.target_language,
            "voiceType": self._options.voice_type,
        }

        logger.debug("Creating Pinch session  url=%s", _PINCH_SESSION_URL)

        async with aiohttp.ClientSession() as http:
            async with http.post(
                _PINCH_SESSION_URL,
                headers=headers,
                json=body,
                allow_redirects=False,
            ) as resp:
                # Follow redirects manually so the Authorization header is
                # preserved — aiohttp (like most HTTP clients) strips it on
                # automatic cross-host redirects.
                if resp.status in (301, 302, 307, 308):
                    redirect_url = resp.headers.get("Location")
                    logger.debug("Following redirect → %s", redirect_url)
                    async with http.post(
                        redirect_url,
                        headers=headers,
                        json=body,
                    ) as resp:
                        pass  # fall through to the status checks below

                if resp.status == 401:
                    raise PinchAuthError(
                        "Pinch API key rejected (HTTP 401). "
                        "Check the PINCH_API_KEY environment variable."
                    )
                if resp.status == 429:
                    raise PinchRateLimitError(
                        "Pinch API rate limit exceeded (HTTP 429). "
                        "Please wait before retrying."
                    )
                if resp.status == 400:
                    detail = await resp.text()
                    raise PinchSessionError(
                        f"Bad request to Pinch API (HTTP 400): {detail}"
                    )
                if resp.status >= 500:
                    detail = await resp.text()
                    raise PinchSessionError(
                        f"Pinch API server error (HTTP {resp.status}): {detail}"
                    )
                if resp.status != 200:
                    detail = await resp.text()
                    raise PinchSessionError(
                        f"Unexpected response from Pinch API (HTTP {resp.status}): {detail}"
                    )

                data = await resp.json()
                if not all(k in data for k in ("url", "token", "room_name")):
                    raise PinchSessionError(
                        f"Pinch API returned an unexpected payload: {data!r}"
                    )
                return data

    # ------------------------------------------------------------------
    # Internal – room connection with retry
    # ------------------------------------------------------------------

    async def _connect_with_retry(self, url: str, token: str) -> rtc.Room:
        """Connect to a LiveKit room with exponential-backoff retry."""
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_CONNECT_RETRIES):
            room = rtc.Room()
            try:
                await room.connect(url, token)
                return room
            except Exception as exc:
                last_exc = exc
                if attempt == _MAX_CONNECT_RETRIES - 1:
                    break
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Pinch room connection attempt %d/%d failed: %s  — retrying in %.1fs",
                    attempt + 1,
                    _MAX_CONNECT_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise PinchSessionError(
            f"Could not connect to the Pinch room after "
            f"{_MAX_CONNECT_RETRIES} attempts."
        ) from last_exc

    # ------------------------------------------------------------------
    # Internal – audio bridge: source_room → Pinch room
    # ------------------------------------------------------------------

    async def _run_source_to_pinch_bridge(self) -> None:
        """Subscribe to every audio track in *source_room* and forward
        the raw PCM frames into *_pinch_audio_source*."""
        if self._source_room is None:
            return

        # Keep track of per-track streaming tasks so we can clean them up.
        streaming_tasks: List[asyncio.Task] = []

        async def _stream_track(track: rtc.RemoteAudioTrack) -> None:
            logger.debug("source→Pinch bridge: streaming track sid=%s", track.sid)
            try:
                audio_stream = rtc.AudioStream(
                    track,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=_NUM_CHANNELS,
                )
                async for event in audio_stream:
                    if self._stopped:
                        break
                    if self._pinch_audio_source is not None:
                        await self._pinch_audio_source.capture_frame(event.frame)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._stopped:
                    logger.warning(
                        "source→Pinch bridge error on track %s: %s", track.sid, exc
                    )

        def start_streaming(track: rtc.Track) -> None:
            if isinstance(track, rtc.RemoteAudioTrack):
                t = asyncio.ensure_future(_stream_track(track))
                streaming_tasks.append(t)

        # Handle tracks that are already subscribed at start time.
        for participant in self._source_room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    start_streaming(pub.track)

        # Handle tracks subscribed after start.
        @self._source_room.on("track_subscribed")
        def _on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            start_streaming(track)

        try:
            # Keep this task alive until stopped.
            while not self._stopped:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            self._source_room.off("track_subscribed", _on_track_subscribed)
            for t in streaming_tasks:
                t.cancel()
            if streaming_tasks:
                await asyncio.gather(*streaming_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal – audio bridge: Pinch room → source_room
    # ------------------------------------------------------------------

    async def _run_pinch_to_source_bridge(self) -> None:
        """Wait for the Pinch translation agent to publish an audio track,
        then forward its frames into *_source_audio_source*."""
        if self._pinch_room is None:
            return

        streaming_tasks: List[asyncio.Task] = []

        async def _stream_track(track: rtc.RemoteAudioTrack) -> None:
            logger.debug("Pinch→source bridge: streaming track sid=%s", track.sid)
            try:
                audio_stream = rtc.AudioStream(
                    track,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=_NUM_CHANNELS,
                )
                async for event in audio_stream:
                    if self._stopped:
                        break
                    if self._source_audio_source is not None:
                        await self._source_audio_source.capture_frame(event.frame)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if not self._stopped:
                    logger.warning(
                        "Pinch→source bridge error on track %s: %s", track.sid, exc
                    )

        def start_streaming(track: rtc.Track) -> None:
            if isinstance(track, rtc.RemoteAudioTrack):
                logger.info(
                    "Pinch translation agent published audio track sid=%s — "
                    "forwarding to source room.",
                    track.sid,
                )
                t = asyncio.ensure_future(_stream_track(track))
                streaming_tasks.append(t)

        # The Pinch translation agent might already be in the room (unlikely
        # but possible in fast follow-up calls).
        for participant in self._pinch_room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    start_streaming(pub.track)

        # Poll/wait for the agent track to appear.
        agent_track_seen = any(
            isinstance(pub.track, rtc.RemoteAudioTrack)
            for p in self._pinch_room.remote_participants.values()
            for pub in p.track_publications.values()
        )

        if not agent_track_seen:
            logger.info(
                "Waiting for Pinch translation agent to publish audio "
                "(timeout=%.0fs)…",
                _AGENT_TRACK_TIMEOUT,
            )

        @self._pinch_room.on("track_subscribed")
        def _on_pinch_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            start_streaming(track)

        try:
            while not self._stopped:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            self._pinch_room.off("track_subscribed", _on_pinch_track_subscribed)
            for t in streaming_tasks:
                t.cancel()
            if streaming_tasks:
                await asyncio.gather(*streaming_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal – data channel
    # ------------------------------------------------------------------

    def _on_pinch_data_received(self, data_packet: rtc.DataPacket) -> None:
        """Parse a Pinch data-channel message and emit a :class:`TranscriptEvent`."""
        try:
            raw = data_packet.data
            if isinstance(raw, (bytes, bytearray, memoryview)):
                raw = bytes(raw).decode("utf-8")
            msg: dict = json.loads(raw)
        except Exception as exc:
            logger.warning("Failed to decode Pinch data message: %s", exc)
            return

        msg_type = msg.get("type", "")
        if msg_type not in ("original_transcript", "translated_transcript"):
            # Unknown or internal Pinch message – ignore silently.
            return

        event = TranscriptEvent(
            type=msg_type,
            text=msg.get("text", ""),
            is_final=bool(msg.get("is_final", False)),
            language_detected=msg.get("language_detected", ""),
            timestamp=float(msg.get("timestamp", time.time())),
            confidence=float(msg.get("confidence", 0.0)),
        )

        logger.debug(
            "Transcript [%s] final=%s  %r",
            event.type,
            event.is_final,
            event.text[:80],
        )

        self._emit_transcript(event)

    def _emit_transcript(self, event: TranscriptEvent) -> None:
        """Invoke all registered transcript callbacks."""
        for cb in self._transcript_callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.exception(
                    "Unhandled exception in transcript callback %r: %s", cb, exc
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Translator src={self._options.source_language!r} "
            f"tgt={self._options.target_language!r} "
            f"voice={self._options.voice_type!r} "
            f"started={self._started} stopped={self._stopped}>"
        )
