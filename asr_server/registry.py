from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Backend = Literal["auto", "transformers", "vllm"]
ModelStatus = Literal["unloaded", "loading", "loaded", "unloading_scheduled", "unloading", "error"]
MOSS_MODEL_ID = "moss-transcribe-diarize-0.9b"


QWEN_LANGUAGES = [
    "auto",
    "zh",
    "en",
    "yue",
    "ar",
    "de",
    "fr",
    "es",
    "pt",
    "id",
    "it",
    "ko",
    "ru",
    "th",
    "vi",
    "ja",
    "tr",
    "hi",
    "ms",
    "nl",
    "sv",
    "da",
    "fi",
    "pl",
    "cs",
    "fil",
    "fa",
    "el",
    "hu",
    "mk",
    "ro",
]

QWEN_CHINESE_DIALECTS = [
    "Anhui",
    "Dongbei",
    "Fujian",
    "Gansu",
    "Guizhou",
    "Hebei",
    "Henan",
    "Hubei",
    "Hunan",
    "Jiangxi",
    "Ningxia",
    "Shandong",
    "Shaanxi",
    "Shanxi",
    "Sichuan",
    "Tianjin",
    "Yunnan",
    "Zhejiang",
    "Cantonese-Hong-Kong-accent",
    "Cantonese-Guangdong-accent",
    "Wu",
    "Minnan",
]


@dataclass(frozen=True)
class ModelCapabilities:
    transcription: bool
    streaming: bool
    timestamps: list[str]
    forced_alignment: bool
    languages: list[str]
    chinese_dialects: list[str]
    backends: list[str]
    diarization: bool = False
    segment_timestamps: bool = False

    def to_api(self) -> dict[str, object]:
        return {
            "transcription": self.transcription,
            "streaming": self.streaming,
            "timestamps": self.timestamps,
            "forced_alignment": self.forced_alignment,
            "languages": self.languages,
            "chinese_dialects": self.chinese_dialects,
            "backends": self.backends,
            "diarization": self.diarization,
            "segment_timestamps": self.segment_timestamps,
        }


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    provider: str
    default: bool
    capabilities: ModelCapabilities


def default_models(
    default_model_id: str = "qwen3-asr-1.7b",
    *,
    enable_moss: bool = False,
) -> dict[str, ModelDefinition]:
    if default_model_id not in {"qwen3-asr-1.7b", "qwen3-asr-0.6b"}:
        raise ValueError("ASR_DEFAULT_MODEL must be qwen3-asr-1.7b or qwen3-asr-0.6b")
    qwen_capabilities = ModelCapabilities(
        transcription=True,
        streaming=False,
        timestamps=[],
        forced_alignment=False,
        languages=QWEN_LANGUAGES,
        chinese_dialects=QWEN_CHINESE_DIALECTS,
        backends=["transformers"],
    )
    models: dict[str, ModelDefinition] = {
        "qwen3-asr-1.7b": ModelDefinition(
            id="qwen3-asr-1.7b",
            provider="QwenLM",
            default=default_model_id == "qwen3-asr-1.7b",
            capabilities=qwen_capabilities,
        ),
        "qwen3-asr-0.6b": ModelDefinition(
            id="qwen3-asr-0.6b",
            provider="QwenLM",
            default=default_model_id == "qwen3-asr-0.6b",
            capabilities=qwen_capabilities,
        ),
    }
    if enable_moss:
        models[MOSS_MODEL_ID] = ModelDefinition(
            id=MOSS_MODEL_ID,
            provider="OpenMOSS-Team",
            default=False,
            capabilities=ModelCapabilities(
                transcription=True,
                streaming=False,
                timestamps=[],
                forced_alignment=False,
                languages=["auto", "zh", "en"],
                chinese_dialects=[],
                backends=["transformers"],
                diarization=True,
                segment_timestamps=True,
            ),
        )
    return models
