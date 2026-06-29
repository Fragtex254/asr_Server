from __future__ import annotations

import pytest

from asr_server.config import load_settings
from asr_server.main import create_app


def test_load_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_HOST", "127.0.0.1")
    monkeypatch.setenv("ASR_PORT", "19000")
    monkeypatch.setenv("ASR_PUBLIC_BASE_URL", "http://127.0.0.1:19000")
    monkeypatch.setenv("ASR_DEFAULT_MODEL", "qwen3-asr-0.6b")
    monkeypatch.setenv("ASR_ADAPTER", "qwen")
    monkeypatch.setenv("ASR_QWEN_BATCH_SIZE", "2")
    monkeypatch.setenv("ASR_MAX_QUEUED_JOBS", "3")
    monkeypatch.setenv("ASR_MAX_UPLOAD_MB", "4")

    settings = load_settings()

    assert settings.host == "127.0.0.1"
    assert settings.port == 19000
    assert settings.public_base_url == "http://127.0.0.1:19000"
    assert settings.default_model == "qwen3-asr-0.6b"
    assert settings.adapter == "qwen"
    assert settings.qwen_batch_size == 2
    assert settings.max_queued_jobs == 3
    assert settings.max_upload_mb == 4


def test_invalid_adapter_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_ADAPTER", "cpu")

    with pytest.raises(ValueError, match="ASR_ADAPTER"):
        load_settings()


def test_invalid_qwen_batch_size_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_QWEN_BATCH_SIZE", "0")

    with pytest.raises(ValueError, match="ASR_QWEN_BATCH_SIZE"):
        load_settings()


def test_invalid_max_queued_jobs_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_MAX_QUEUED_JOBS", "0")

    with pytest.raises(ValueError, match="ASR_MAX_QUEUED_JOBS"):
        load_settings()


def test_invalid_max_upload_mb_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_MAX_UPLOAD_MB", "0")

    with pytest.raises(ValueError, match="ASR_MAX_UPLOAD_MB"):
        load_settings()


def test_default_model_setting_controls_model_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_DEFAULT_MODEL", "qwen3-asr-0.6b")

    app = create_app()
    models = app.state.manager.list_models()

    defaults = [model["id"] for model in models if model["default"] is True]
    assert defaults == ["qwen3-asr-0.6b"]
