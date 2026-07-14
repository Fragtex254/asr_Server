from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


ExecutionMode = Literal["chunked", "native_long_form"]

QWEN_MAX_NEW_TOKENS = 4_096
MOSS_MAX_NEW_TOKENS = 65_536
DEFAULT_CHUNK_HARD_SECONDS = 300.0
MAX_AUDIO_SECONDS_PER_FILE = 21_600.0


@dataclass(frozen=True)
class ExecutionPlan:
    execution_mode: ExecutionMode
    requested_split_strategy: str
    split_strategy: str
    max_chunk_seconds: float | None
    hard_chunk_seconds: float
    allow_automatic_chunk_fallback: bool
    fallback_reason: str | None = None

    def to_api(self, *, speaker_scope: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.execution_mode,
            "requested_split_strategy": self.requested_split_strategy,
            "resolved_split_strategy": self.split_strategy,
            "speaker_scope": speaker_scope,
            "automatic_chunk_fallback": self.allow_automatic_chunk_fallback,
        }
        if self.fallback_reason is not None:
            payload["fallback_reason"] = self.fallback_reason
        return payload

    @property
    def warnings(self) -> list[str]:
        if self.fallback_reason is None:
            return []
        return [f"moss_native_long_form_fallback:{self.fallback_reason}"]


@dataclass(frozen=True)
class ModelExecutionPolicy:
    auto_execution_mode: ExecutionMode
    default_max_new_tokens: int
    max_new_tokens: int
    tokens_per_audio_second: float | None = None
    native_hard_seconds: float = DEFAULT_CHUNK_HARD_SECONDS
    chunk_hard_seconds: float = DEFAULT_CHUNK_HARD_SECONDS
    validated_native_max_seconds: float | None = None
    fallback_chunk_seconds: float | None = None

    def plan(
        self,
        *,
        requested_split_strategy: str,
        audio_duration_seconds: float | None = None,
        max_chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> ExecutionPlan:
        native_preferred = requested_split_strategy == "none" or (
            requested_split_strategy == "auto" and self.auto_execution_mode == "native_long_form"
        )
        if native_preferred and (max_chunk_seconds is not None or overlap_seconds is not None):
            from asr_server.errors import AsrError

            raise AsrError(
                422,
                "capability_not_supported",
                "chunk sizing options do not apply to native long-form execution",
                {
                    "requested_split_strategy": requested_split_strategy,
                    "recommended_split_strategy": "fixed",
                },
            )
        fallback = (
            requested_split_strategy == "auto"
            and self.auto_execution_mode == "native_long_form"
            and self.validated_native_max_seconds is not None
            and audio_duration_seconds is not None
            and audio_duration_seconds > self.validated_native_max_seconds
        )
        native = native_preferred and not fallback
        fallback_reason = "duration_exceeds_validated_native_limit" if fallback else None
        return ExecutionPlan(
            execution_mode="native_long_form" if native else "chunked",
            requested_split_strategy=requested_split_strategy,
            split_strategy="none" if native else "fixed" if fallback else requested_split_strategy,
            max_chunk_seconds=self.fallback_chunk_seconds if fallback else max_chunk_seconds,
            hard_chunk_seconds=self.native_hard_seconds if native else self.chunk_hard_seconds,
            allow_automatic_chunk_fallback=fallback,
            fallback_reason=fallback_reason,
        )

    def validate_max_new_tokens(self, requested: int | None) -> int | None:
        if requested is not None and requested > self.max_new_tokens:
            from asr_server.errors import AsrError

            raise AsrError(
                400,
                "bad_request",
                "max_new_tokens exceeds the model limit",
                {"max_new_tokens": requested, "max_new_tokens_limit": self.max_new_tokens},
            )
        return requested

    def resolve_max_new_tokens(self, requested: int | None, *, invocation_duration_seconds: float) -> int:
        if requested is not None:
            return requested
        if self.tokens_per_audio_second is None:
            return self.default_max_new_tokens
        scaled = math.ceil(max(invocation_duration_seconds, 0.0) * self.tokens_per_audio_second)
        return min(max(self.default_max_new_tokens, scaled), self.max_new_tokens)


QWEN_EXECUTION_POLICY = ModelExecutionPolicy(
    auto_execution_mode="chunked",
    default_max_new_tokens=512,
    max_new_tokens=QWEN_MAX_NEW_TOKENS,
)

MOSS_EXECUTION_POLICY = ModelExecutionPolicy(
    auto_execution_mode="native_long_form",
    default_max_new_tokens=2_048,
    max_new_tokens=MOSS_MAX_NEW_TOKENS,
    tokens_per_audio_second=12.0,
    native_hard_seconds=MAX_AUDIO_SECONDS_PER_FILE,
    # The RTX 5070 Ti validation completed at 1800.02 seconds. Keep only a
    # one-second container/decoder tolerance instead of claiming an untested
    # 31-minute native window.
    chunk_hard_seconds=1_801.0,
    validated_native_max_seconds=1_801.0,
    fallback_chunk_seconds=1_800.0,
)
