from __future__ import annotations

import json
from pathlib import Path

from asr_server.audio.transcript import build_transcript_document, write_transcript_artifacts


def test_dedupes_prefix_only_when_segments_overlap() -> None:
    document = build_transcript_document(
        [
            {"start": 0.0, "end": 10.0, "text": "第一段结束重复文本", "language": "zh"},
            {"start": 8.0, "end": 18.0, "text": "重复文本第二段继续", "language": "zh"},
            {"start": 20.0, "end": 30.0, "text": "重复文本不应删除", "language": "zh"},
        ],
        metadata={"audio": "sample.wav"},
    )

    assert document.segments[1].text == "第二段继续"
    assert document.segments[1].deduped_prefix_chars == len("重复文本")
    assert document.segments[2].text == "重复文本不应删除"
    assert document.segments[2].deduped_prefix_chars == 0
    assert document.text == "第一段结束重复文本\n\n第二段继续\n\n重复文本不应删除"


def test_dedupes_prefix_found_near_previous_tail_with_punctuation_changes() -> None:
    document = build_transcript_document(
        [
            {"start": 0.0, "end": 10.0, "text": "你是我的东西。六，只。", "language": "zh"},
            {"start": 8.0, "end": 18.0, "text": "我的东西。六，只需要接受主人的命令。", "language": "zh"},
            {"start": 17.0, "end": 20.0, "text": "主人的命令，继续。", "language": "zh"},
        ],
        metadata={"audio": "sample.wav"},
    )

    assert document.segments[1].text == "需要接受主人的命令。"
    assert document.segments[2].text == "继续。"
    assert document.segments[1].deduped_prefix_chars == len("我的东西。六，只")
    assert document.segments[2].deduped_prefix_chars == len("主人的命令，")


def test_writes_json_txt_and_srt_artifacts(tmp_path: Path) -> None:
    document = build_transcript_document(
        [
            {"start": 1.0, "end": 2.5, "text": "你好", "language": "zh"},
            {"start": 2.0, "end": 4.0, "text": "你好世界", "language": "zh"},
        ],
        metadata={"audio": "sample.wav"},
    )

    paths = write_transcript_artifacts(document, tmp_path / "transcript")

    body = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert body["metadata"]["audio"] == "sample.wav"
    assert body["segments"][1]["deduped_prefix_chars"] == len("你好")
    assert paths["txt"].read_text(encoding="utf-8") == "你好\n\n世界\n"
    assert "00:00:01,000 --> 00:00:02,500" in paths["srt"].read_text(encoding="utf-8")
    assert "00:00:02,000 --> 00:00:04,000" in paths["srt"].read_text(encoding="utf-8")
