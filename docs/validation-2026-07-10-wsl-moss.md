# WSL MOSS-Transcribe-Diarize Validation Record

日期：2026-07-10

状态：独立 MOSS backend smoke 与服务端 HTTP 路径均已在 WSL RTX 5070 Ti 环境通过。

## 环境

- 项目目录：`/home/fragt/services/asr-server`
- 模型 ID：`OpenMOSS-Team/MOSS-Transcribe-Diarize`
- 服务模型 ID：`moss-transcribe-diarize-0.9b`
- 后端：`transformers`
- 音频文件：`test-fixtures/audio/test_short.wav`

## 预检查

```bash
cd /home/fragt/services/asr-server
uv python install 3.12
uv sync
sudo pacman -Syu --needed git ffmpeg libsndfile
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.version.cuda is not None, "installed torch is CPU-only"
assert torch.cuda.is_available(), "torch cannot access CUDA"
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

## MOSS 依赖

```bash
uv pip install --torch-backend cu128 -r requirements/wsl-gpu-cu128.txt
```

安装后必须重新验收 torch：

```bash
uv run python - <<'PY'
import torch
import transformers

assert torch.version.cuda is not None, "torch became CPU-only after dependency install"
assert torch.cuda.is_available(), "CUDA unavailable after dependency install"
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
PY
```

## Smoke 命令

```bash
uv run python scripts/moss_backend_smoke.py \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio test-fixtures/audio/test_short.wav \
  --max-new-tokens 2048
```

## 待记录结果

- torch 版本：`2.11.0+cu128`
- CUDA 版本：`12.8`
- GPU 名称：`NVIDIA GeForce RTX 5070 Ti`
- GPU capability：`(12, 0)`
- transformers 版本：`5.14.0.dev0`
- MOSS package 版本：`0.1.0`
- MOSS package commit：`b5ad0f8386b155ddb89f9332ba3ca71891900357`
- Hugging Face model snapshot：`d7231bbae2587a4af278735eb765b318c4f64edd`
- max_new_tokens：`2048`
- 输出文本前 200 字：`[0.78][S01]你好你好你好，这个是测试音频，主要用于测试千问[7.34][7.94][S01]ASR模型一点七B的实际转入能力。[12.66]`
- parsed segment 数量：`2`
- 第一条 segment 示例：`{"start": 0.78, "end": 7.34, "speaker": "S01", "text": "你好你好你好，这个是测试音频，主要用于测试千问"}`

## 独立 smoke 结果

```text
model: OpenMOSS-Team/MOSS-Transcribe-Diarize
backend: transformers
loader: hf-remote-code
max_new_tokens: 2048
text first 200: [0.78][S01]你好你好你好，这个是测试音频，主要用于测试千问[7.34][7.94][S01]ASR模型一点七B的实际转入能力。[12.66]
parsed segment count: 2
first segment: {'start': 0.78, 'end': 7.34, 'speaker': 'S01', 'text': '你好你好你好，这个是测试音频，主要用于测试千问'}
```

对抗性记录：

- 本次下载了 Hugging Face remote code，服务端只固定加载 `OpenMOSS-Team/MOSS-Transcribe-Diarize`，不允许客户端传任意 repo id。
- 后续生产稳定后应 pin 到 `d7231bbae2587a4af278735eb765b318c4f64edd` 或明确记录更新后的 model revision。
- MOSS 依赖安装后已复验 `torch.version.cuda is not None` 和 `torch.cuda.is_available() is True`。

## 服务验收

MOSS smoke 通过后，再启动服务：

```bash
ASR_ADAPTER=qwen ASR_ENABLE_MOSS=1 \
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

Mac mini 验收：

```bash
curl --noproxy '*' http://192.168.31.137:18080/v1/models
curl --noproxy '*' -X POST http://192.168.31.137:18080/v1/audio/transcriptions \
  -F model=moss-transcribe-diarize-0.9b \
  -F file=@test-fixtures/audio/test_short.wav \
  -F response_format=verbose_json \
  -F max_new_tokens=2048
```

实际服务状态：

- systemd unit：`asr-server.service`
- WorkingDirectory：`/home/fragt/services/asr-server`
- Environment：`ASR_ADAPTER=qwen ASR_ENABLE_MOSS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- `/health`：HTTP 200，GPU 为 `NVIDIA GeForce RTX 5070 Ti`
- `/v1/models`：包含 `moss-transcribe-diarize-0.9b`，默认模型仍为 `qwen3-asr-1.7b`

MOSS verbose_json 转写：

```text
HTTP_STATUS: 200
model: moss-transcribe-diarize-0.9b
backend: transformers
language: auto
text: 你好你好你好，这个是测试音频，主要用于测试千问
ASR模型一点七B的实际转入能力。
segments:
- start: 0.21
  end: 7.35
  speaker: S01
  text: 你好你好你好，这个是测试音频，主要用于测试千问
- start: 7.94
  end: 12.67
  speaker: S01
  text: ASR模型一点七B的实际转入能力。
timings:
  total_ms: 6984.469233000027
  load_ms: 4252.479056000084
  inference_ms: 2694.880934999901
```

补充接口验收：

- `response_format=text`：HTTP 200，返回纯文本。
- `timestamps=word`：HTTP 422，`capability_not_supported`。
- `backend=vllm`：HTTP 422，`capability_not_supported`。
- `DELETE /v1/models/moss-transcribe-diarize-0.9b`：HTTP 200，状态回到 `unloaded`。
- 卸载后 `nvidia-smi` 无 MOSS 推理进程，显存回到约 `1879MiB / 16303MiB`。
