from __future__ import annotations

from asr_server.execution import MOSS_EXECUTION_POLICY, QWEN_EXECUTION_POLICY


def test_moss_auto_uses_native_long_form_with_duration_scaled_token_budget() -> None:
    plan = MOSS_EXECUTION_POLICY.plan(requested_split_strategy="auto", audio_duration_seconds=1_800.0)

    assert plan.execution_mode == "native_long_form"
    assert plan.split_strategy == "none"
    assert plan.hard_chunk_seconds == 21_600.0
    assert plan.allow_automatic_chunk_fallback is False
    assert MOSS_EXECUTION_POLICY.resolve_max_new_tokens(None, invocation_duration_seconds=1_800.0) == 21_600


def test_moss_auto_falls_back_explicitly_above_validated_native_window() -> None:
    plan = MOSS_EXECUTION_POLICY.plan(requested_split_strategy="auto", audio_duration_seconds=3_600.0)

    assert plan.execution_mode == "chunked"
    assert plan.split_strategy == "fixed"
    assert plan.max_chunk_seconds == 1_800.0
    assert plan.hard_chunk_seconds == 1_801.0
    assert plan.allow_automatic_chunk_fallback is True
    assert plan.fallback_reason == "duration_exceeds_validated_native_limit"


def test_moss_native_validation_boundary_has_only_decoder_tolerance() -> None:
    at_limit = MOSS_EXECUTION_POLICY.plan(
        requested_split_strategy="auto",
        audio_duration_seconds=1_801.0,
    )
    above_limit = MOSS_EXECUTION_POLICY.plan(
        requested_split_strategy="auto",
        audio_duration_seconds=1_801.001,
    )

    assert at_limit.execution_mode == "native_long_form"
    assert above_limit.execution_mode == "chunked"


def test_moss_explicit_none_can_challenge_unvalidated_longer_audio() -> None:
    plan = MOSS_EXECUTION_POLICY.plan(requested_split_strategy="none", audio_duration_seconds=5_406.0)

    assert plan.execution_mode == "native_long_form"
    assert plan.split_strategy == "none"
    assert MOSS_EXECUTION_POLICY.resolve_max_new_tokens(None, invocation_duration_seconds=5_406.0) == 64_872
    assert MOSS_EXECUTION_POLICY.resolve_max_new_tokens(None, invocation_duration_seconds=6_000.0) == 65_536


def test_qwen_auto_keeps_chunked_policy_and_existing_token_defaults() -> None:
    plan = QWEN_EXECUTION_POLICY.plan(requested_split_strategy="auto")

    assert plan.execution_mode == "chunked"
    assert plan.split_strategy == "auto"
    assert plan.hard_chunk_seconds == 300.0
    assert QWEN_EXECUTION_POLICY.resolve_max_new_tokens(None, invocation_duration_seconds=5_406.0) == 512


def test_moss_chunked_fallback_scales_tokens_to_each_model_invocation() -> None:
    plan = MOSS_EXECUTION_POLICY.plan(requested_split_strategy="fixed")

    assert plan.execution_mode == "chunked"
    assert plan.split_strategy == "fixed"
    assert MOSS_EXECUTION_POLICY.resolve_max_new_tokens(None, invocation_duration_seconds=120.0) == 2_048
