from __future__ import annotations

from pathlib import Path

import pytest

from asr_server.audio.workspace import validate_workspace_limits
from asr_server.config import Settings
from asr_server.errors import AsrError


def test_workspace_file_count_limit_is_enforced(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file-{index}").write_bytes(b"x")

    with pytest.raises(AsrError) as exc_info:
        validate_workspace_limits(tmp_path, Settings(max_workspace_files=2))

    assert exc_info.value.code == "storage_unavailable"
    assert exc_info.value.details["file_count"] == 3


def test_workspace_byte_limit_is_enforced(tmp_path: Path) -> None:
    (tmp_path / "large").write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(AsrError) as exc_info:
        validate_workspace_limits(tmp_path, Settings(max_workspace_mb=1))

    assert exc_info.value.details["max_workspace_bytes"] == 1024 * 1024
