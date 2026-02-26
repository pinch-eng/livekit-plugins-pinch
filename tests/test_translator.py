"""Tests for livekit-plugins-pinch.

Run with:  pytest -v tests/test_translator.py

The tests mock all external I/O (HTTP and LiveKit RTC) so they execute
fully offline with no credentials required.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from livekit.plugins.pinch import (
    Translator,
    TranslatorOptions,
    TranscriptEvent,
    PinchAuthError,
    PinchRateLimitError,
    PinchSessionError,
)
from livekit.plugins.pinch.models import TranslatorOptions, TranscriptEvent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_GOOD_SESSION = {
    "url": "wss://pinch-test.livekit.cloud",
    "token": "eyJhbGciOiJIUzI1NiJ9.test",
    "room_name": "api-abc123",
}


def _make_options(
    src: str = "en-US",
    tgt: str = "es-ES",
    voice: str = "clone",
) -> TranslatorOptions:
    return TranslatorOptions(
        source_language=src,
        target_language=tgt,
        voice_type=voice,
    )


def _make_mock_http_resp(status: int = 200, payload: Any = None, text: str = ""):
    """Build a minimal async context-manager mock for aiohttp responses."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload or _GOOD_SESSION)
    resp.text = AsyncMock(return_value=text)
    # Context-manager protocol
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_mock_http_session(resp_mock):
    """Build a minimal async context-manager mock for aiohttp.ClientSession."""
    post_cm = AsyncMock()
    post_cm.__aenter__ = AsyncMock(return_value=resp_mock)
    post_cm.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.post = MagicMock(return_value=post_cm)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _make_mock_rtc_room():
    """Return a MagicMock that looks enough like rtc.Room for our tests."""
    room = MagicMock()
    room.remote_participants = {}

    # local_participant
    lp = MagicMock()
    pub = MagicMock()
    pub.sid = "pub-sid-001"
    lp.publish_track = AsyncMock(return_value=pub)
    lp.unpublish_track = AsyncMock()
    room.local_participant = lp

    # Room lifecycle
    room.connect = AsyncMock()
    room.disconnect = AsyncMock()

    # Event emitter stubs
    room.on = MagicMock()
    room.off = MagicMock()

    return room


# ---------------------------------------------------------------------------
# TranslatorOptions tests
# ---------------------------------------------------------------------------


class TestTranslatorOptions:
    def test_defaults(self):
        opts = TranslatorOptions(source_language="en-US", target_language="es-ES")
        assert opts.voice_type == "clone"

    def test_custom_voice(self):
        opts = TranslatorOptions(
            source_language="en-US", target_language="fr-FR", voice_type="female"
        )
        assert opts.voice_type == "female"

    def test_invalid_voice_type_raises(self):
        with pytest.raises(ValueError, match="voice_type"):
            TranslatorOptions(
                source_language="en-US",
                target_language="es-ES",
                voice_type="robot",
            )


# ---------------------------------------------------------------------------
# TranscriptEvent tests
# ---------------------------------------------------------------------------


class TestTranscriptEvent:
    def test_is_original(self):
        ev = TranscriptEvent(
            type="original_transcript",
            text="Hello",
            is_final=True,
            language_detected="en-US",
            timestamp=0.0,
        )
        assert ev.is_original is True
        assert ev.is_translated is False

    def test_is_translated(self):
        ev = TranscriptEvent(
            type="translated_transcript",
            text="Hola",
            is_final=True,
            language_detected="en-US",
            timestamp=0.0,
        )
        assert ev.is_translated is True
        assert ev.is_original is False

    def test_confidence_defaults_to_zero(self):
        ev = TranscriptEvent(
            type="original_transcript",
            text="Hi",
            is_final=False,
            language_detected="en-US",
            timestamp=0.0,
        )
        assert ev.confidence == 0.0


# ---------------------------------------------------------------------------
# Translator constructor tests
# ---------------------------------------------------------------------------


class TestTranslatorConstructor:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("PINCH_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            Translator(options=_make_options())

    def test_accepts_env_var(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "test-key-from-env")
        t = Translator(options=_make_options())
        assert t._api_key == "test-key-from-env"

    def test_explicit_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "env-key")
        t = Translator(api_key="explicit-key", options=_make_options())
        assert t._api_key == "explicit-key"

    def test_repr(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "k")
        t = Translator(options=_make_options())
        r = repr(t)
        assert "en-US" in r
        assert "es-ES" in r


# ---------------------------------------------------------------------------
# Session creation tests (HTTP mocking)
# ---------------------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "test-key")
        t = Translator(options=_make_options())

        resp = _make_mock_http_resp(200, _GOOD_SESSION)
        http = _make_mock_http_session(resp)

        with patch("aiohttp.ClientSession", return_value=http):
            data = await t._create_session()

        assert data["room_name"] == "api-abc123"
        assert data["url"].startswith("wss://")

    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "bad-key")
        t = Translator(options=_make_options())

        resp = _make_mock_http_resp(401, text="Unauthorized")
        http = _make_mock_http_session(resp)

        with patch("aiohttp.ClientSession", return_value=http):
            with pytest.raises(PinchAuthError):
                await t._create_session()

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit_error(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())

        resp = _make_mock_http_resp(429, text="Too Many Requests")
        http = _make_mock_http_session(resp)

        with patch("aiohttp.ClientSession", return_value=http):
            with pytest.raises(PinchRateLimitError):
                await t._create_session()

    @pytest.mark.asyncio
    async def test_500_raises_session_error(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())

        resp = _make_mock_http_resp(500, text="Internal Server Error")
        http = _make_mock_http_session(resp)

        with patch("aiohttp.ClientSession", return_value=http):
            with pytest.raises(PinchSessionError):
                await t._create_session()

    @pytest.mark.asyncio
    async def test_malformed_response_raises_session_error(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())

        resp = _make_mock_http_resp(200, payload={"unexpected": "field"})
        http = _make_mock_http_session(resp)

        with patch("aiohttp.ClientSession", return_value=http):
            with pytest.raises(PinchSessionError, match="unexpected"):
                await t._create_session()


# ---------------------------------------------------------------------------
# Transcript callback tests
# ---------------------------------------------------------------------------


class TestTranscriptCallbacks:
    def _make_translator(self, monkeypatch) -> Translator:
        monkeypatch.setenv("PINCH_API_KEY", "key")
        return Translator(options=_make_options())

    def test_register_and_emit(self, monkeypatch):
        t = self._make_translator(monkeypatch)
        received: list[TranscriptEvent] = []
        t.on_transcript(received.append)

        ev = TranscriptEvent(
            type="translated_transcript",
            text="Hola",
            is_final=True,
            language_detected="en-US",
            timestamp=123.0,
        )
        t._emit_transcript(ev)
        assert len(received) == 1
        assert received[0].text == "Hola"

    def test_decorator_usage(self, monkeypatch):
        t = self._make_translator(monkeypatch)
        fired: list[TranscriptEvent] = []

        @t.on_transcript
        def handler(ev: TranscriptEvent) -> None:
            fired.append(ev)

        ev = TranscriptEvent(
            type="original_transcript",
            text="Hello",
            is_final=False,
            language_detected="en-US",
            timestamp=0.0,
        )
        t._emit_transcript(ev)
        assert len(fired) == 1

    def test_remove_listener(self, monkeypatch):
        t = self._make_translator(monkeypatch)
        received: list[TranscriptEvent] = []
        t.on_transcript(received.append)
        t.remove_transcript_listener(received.append)

        ev = TranscriptEvent(
            type="original_transcript",
            text="Hello",
            is_final=True,
            language_detected="en-US",
            timestamp=0.0,
        )
        t._emit_transcript(ev)
        assert received == []

    def test_faulty_callback_does_not_propagate(self, monkeypatch):
        """A crashing callback must not prevent other callbacks from running."""
        t = self._make_translator(monkeypatch)

        def bad_cb(ev):
            raise RuntimeError("oops")

        good_results: list[str] = []

        def good_cb(ev):
            good_results.append(ev.text)

        t.on_transcript(bad_cb)
        t.on_transcript(good_cb)

        ev = TranscriptEvent(
            type="original_transcript",
            text="hi",
            is_final=True,
            language_detected="en-US",
            timestamp=0.0,
        )
        # Should not raise
        t._emit_transcript(ev)
        assert good_results == ["hi"]


# ---------------------------------------------------------------------------
# _on_pinch_data_received tests
# ---------------------------------------------------------------------------


class TestDataReceived:
    def _make_translator(self, monkeypatch) -> Translator:
        monkeypatch.setenv("PINCH_API_KEY", "key")
        return Translator(options=_make_options())

    def _make_data_packet(self, payload: dict) -> MagicMock:
        pkt = MagicMock()
        pkt.data = json.dumps(payload).encode("utf-8")
        return pkt

    def test_original_transcript(self, monkeypatch):
        t = self._make_translator(monkeypatch)
        received: list[TranscriptEvent] = []
        t.on_transcript(received.append)

        pkt = self._make_data_packet(
            {
                "type": "original_transcript",
                "text": "Hello world",
                "is_final": True,
                "language_detected": "en-US",
                "timestamp": 1770933618.0,
                "confidence": 0.97,
            }
        )
        t._on_pinch_data_received(pkt)

        assert len(received) == 1
        ev = received[0]
        assert ev.type == "original_transcript"
        assert ev.text == "Hello world"
        assert ev.is_final is True
        assert ev.language_detected == "en-US"
        assert ev.confidence == pytest.approx(0.97)

    def test_translated_transcript(self, monkeypatch):
        t = self._make_translator(monkeypatch)
        received: list[TranscriptEvent] = []
        t.on_transcript(received.append)

        pkt = self._make_data_packet(
            {
                "type": "translated_transcript",
                "text": "Hola mundo",
                "is_final": True,
                "language_detected": "en-US",
                "timestamp": 1770933619.0,
            }
        )
        t._on_pinch_data_received(pkt)

        assert len(received) == 1
        assert received[0].is_translated is True
        assert received[0].text == "Hola mundo"

    def test_unknown_type_ignored(self, monkeypatch):
        t = self._make_translator(monkeypatch)
        received: list[TranscriptEvent] = []
        t.on_transcript(received.append)

        pkt = self._make_data_packet({"type": "ping"})
        t._on_pinch_data_received(pkt)
        assert received == []

    def test_malformed_json_ignored(self, monkeypatch):
        t = self._make_translator(monkeypatch)
        pkt = MagicMock()
        pkt.data = b"this is not json {"
        # Should not raise
        t._on_pinch_data_received(pkt)

    def test_string_payload(self, monkeypatch):
        """data field can sometimes be a plain string."""
        t = self._make_translator(monkeypatch)
        received: list[TranscriptEvent] = []
        t.on_transcript(received.append)

        pkt = MagicMock()
        pkt.data = json.dumps(
            {
                "type": "original_transcript",
                "text": "Test",
                "is_final": False,
                "language_detected": "en-US",
                "timestamp": 0.0,
            }
        )  # plain str, not bytes
        t._on_pinch_data_received(pkt)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# stop() tests
# ---------------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_idempotent(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())
        # stop() before start() should not raise
        await t.stop()
        await t.stop()  # second call should also be a no-op

    @pytest.mark.asyncio
    async def test_start_after_stop_raises(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())
        await t.stop()

        source_room = _make_mock_rtc_room()
        with pytest.raises(RuntimeError, match="stopped"):
            await t.start(source_room)


# ---------------------------------------------------------------------------
# Connect with retry tests
# ---------------------------------------------------------------------------


class TestConnectWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())

        mock_room = _make_mock_rtc_room()
        with patch("livekit.plugins.pinch.translator.rtc.Room", return_value=mock_room):
            result = await t._connect_with_retry("wss://test", "token")

        assert result is mock_room
        mock_room.connect.assert_awaited_once_with("wss://test", "token")

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())

        call_count = 0

        class FailTwiceRoom:
            async def connect(self, url, token):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("simulated failure")

        with patch("livekit.plugins.pinch.translator.rtc.Room", FailTwiceRoom):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await t._connect_with_retry("wss://test", "token")

        assert call_count == 3
        assert isinstance(result, FailTwiceRoom)

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setenv("PINCH_API_KEY", "key")
        t = Translator(options=_make_options())

        class AlwaysFailRoom:
            async def connect(self, url, token):
                raise ConnectionError("always fails")

        with patch("livekit.plugins.pinch.translator.rtc.Room", AlwaysFailRoom):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(PinchSessionError, match="3 attempts"):
                    await t._connect_with_retry("wss://test", "token")
