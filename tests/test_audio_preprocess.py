from __future__ import annotations

import io
import wave

from asr_server.audio.preprocess import normalize_audio_to_wav


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
