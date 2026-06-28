from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_AUDIO = "test-fixtures/audio/test_short.wav"


def require_cuda_torch() -> Any:
    torch = importlib.import_module("torch")
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.version.cuda is None:
        raise RuntimeError("当前环境是 CPU 版 torch，停止；请安装 PyTorch CUDA wheel 后重试")
    if not torch.cuda.is_available():
        raise RuntimeError("torch 看不到 CUDA，停止；请检查 WSL NVIDIA 驱动和 CUDA wheel")
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    return torch


def normalize_language(language: str) -> str | None:
    return None if language == "auto" else language


def result_text(result: Any) -> str:
    text = getattr(result, "text", "")
    if not isinstance(text, str):
        return str(text)
    return text


def result_language(result: Any) -> str:
    language = getattr(result, "language", "")
    if not isinstance(language, str):
        return str(language)
    return language


def run_transformers(args: argparse.Namespace) -> None:
    torch = require_cuda_torch()
    qwen_asr = importlib.import_module("qwen_asr")
    model_cls = qwen_asr.Qwen3ASRModel
    model = model_cls.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    results = model.transcribe(audio=str(args.audio), language=normalize_language(args.language))
    first = results[0]
    print("backend: transformers")
    print("language:", result_language(first))
    print("text:", result_text(first)[:500])
    if not result_text(first).strip():
        raise RuntimeError("transformers 后端转录结果为空")


def run_vllm(args: argparse.Namespace) -> None:
    require_cuda_torch()
    qwen_asr = importlib.import_module("qwen_asr")
    model_cls = qwen_asr.Qwen3ASRModel
    model = model_cls.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    results = model.transcribe(audio=str(args.audio), language=normalize_language(args.language))
    first = results[0]
    print("backend: vllm")
    print("language:", result_language(first))
    print("text:", result_text(first)[:500])
    if not result_text(first).strip():
        raise RuntimeError("vllm 后端转录结果为空")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WSL 侧 Qwen3-ASR 最小后端验收脚本；先跑通它，再开发服务端 adapter。",
    )
    parser.add_argument("--backend", required=True, choices=["transformers", "vllm"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio", type=Path, default=Path(DEFAULT_AUDIO))
    parser.add_argument("--language", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)
    if args.backend == "transformers":
        run_transformers(args)
        return
    run_vllm(args)


if __name__ == "__main__":
    main()

