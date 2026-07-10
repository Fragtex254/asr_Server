from __future__ import annotations

import io
import wave
import asyncio
from pathlib import Path

import pytest

from asr_server.audio.preprocess import normalize_audio_path_to_wav_async, normalize_audio_to_wav
from asr_server.errors import AsrError


def make_wav(duration_seconds: float, sample_rate: int = 8_000, channels: int = 2) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * channels * frame_count)
    return output.getvalue()


def test_normalize_audio_to_wav_outputs_asr_friendly_wav() -> None:
    normalized = normalize_audio_to_wav(make_wav(0.1))

    with wave.open(io.BytesIO(normalized.audio), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16_000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 1600
    assert normalized.decode_ms >= 0


async def test_async_ffmpeg_timeout_terminates_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class HangingProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(60)
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = HangingProcess()

    async def create_process(*args: object, **kwargs: object) -> HangingProcess:
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    with pytest.raises(AsrError) as exc_info:
        await normalize_audio_path_to_wav_async(
            tmp_path / "upload",
            tmp_path / "normalized.wav",
            timeout_seconds=0.01,
        )

    assert exc_info.value.code == "audio_preprocess_timeout"
    assert process.terminated is True
