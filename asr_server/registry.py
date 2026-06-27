from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Backend = Literal["auto", "transformers", "vllm"]
ModelStatus = Literal["unloaded", "loading", "loaded", "unloading_scheduled", "unloading", "error"]


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

    def to_api(self) -> dict[str, object]:
        return {
            "transcription": self.transcription,
            "streaming": self.streaming,
            "timestamps": self.timestamps,
            "forced_alignment": self.forced_alignment,
            "languages": self.languages,
            "chinese_dialects": self.chinese_dialects,
            "backends": self.backends,
        }


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    provider: str
    default: bool
    capabilities: ModelCapabilities


def default_models() -> dict[str, ModelDefinition]:
    qwen_capabilities = ModelCapabilities(
        transcription=True,
        streaming=False,
        timestamps=[],
        forced_alignment=False,
        languages=QWEN_LANGUAGES,
        chinese_dialects=QWEN_CHINESE_DIALECTS,
        backends=["transformers", "vllm"],
    )
    return {
        "qwen3-asr-1.7b": ModelDefinition(
            id="qwen3-asr-1.7b",
            provider="QwenLM",
            default=True,
            capabilities=qwen_capabilities,
        ),
        "qwen3-asr-0.6b": ModelDefinition(
            id="qwen3-asr-0.6b",
            provider="QwenLM",
            default=False,
            capabilities=qwen_capabilities,
        ),
    }

