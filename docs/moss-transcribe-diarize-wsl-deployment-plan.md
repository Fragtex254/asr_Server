# MOSS-Transcribe-Diarize WSL Deployment Plan

更新时间：2026-07-09

## 1. 目标

评估并规划在当前 `asr_server` 中接入 Hugging Face 模型 `OpenMOSS-Team/MOSS-Transcribe-Diarize` 的 WSL 侧部署方案。

本计划书只描述方案和实施边界。当前机器是 Mac mini，只能完成文档、schema 设计、mock 测试和客户端验收脚本准备；真实 CUDA、PyTorch GPU wheel、模型下载、MOSS 真实推理和长期后台服务部署，必须在 Windows PC 的 WSL Arch Linux 内完成。

目标不是替换现有 Qwen3-ASR，而是在不破坏 Qwen 主路径的前提下新增一个可验收、可发现、可卸载、可通过局域网调用的 MOSS adapter。

参考来源：

- Hugging Face: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
- GitHub: https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
- 当前项目 PRD: `docs/asr-server-prd.md`

## 2. 第一性原理

### 2.1 公共 API 先于模型实现

`asr_server` 是局域网 ASR 网关，不是某一个模型的临时 demo。任何新模型接入都必须先服从公共 API、错误信封、生命周期状态和 `/v1/models` 能力发现机制。

推论：

- 不允许让客户端硬编码 MOSS 的 Hugging Face repo id。
- 不允许让客户端绕过 `18080` 直接访问内部 worker。
- 不允许在 `/v1/models` 中声明没有通过 WSL 端到端验收的能力。
- 不允许为了 MOSS 改坏现有 Qwen3-ASR 的默认行为。

### 2.2 计算在哪里，依赖就在哪里

Mac mini 是轻量开发机和客户端验收机。RTX 5070 Ti 在 Windows PC，真实推理必须发生在 WSL Arch Linux。

推论：

- Mac 侧不得安装 CUDA、torch GPU 包、MOSS 模型权重或大模型缓存。
- WSL 侧必须先验收 CUDA torch，再安装模型依赖。
- `nvidia-smi` 只能证明 WSL 看得到 GPU，不能证明 Python 环境里的 torch 是 CUDA 版。
- 每次安装模型依赖后，都要重新验证 `torch.version.cuda is not None` 和 `torch.cuda.is_available()`。

### 2.3 模型差异必须隔离在 adapter 内

当前 Qwen3-ASR adapter 使用 `AutoModelForMultimodalLM` 和 Qwen 的 HF native transcription helper。MOSS 使用 `AutoModelForCausalLM`、`AutoProcessor`、`trust_remote_code=True`、官方 helper 的 chat-template/audio 输入流程，并输出紧凑格式：

```text
[start_time][Sxx]transcribed speech[end_time]
```

这两条路径不是同一个 adapter 的参数差异，而是模型协议差异。

推论：

- 新增 `MossTranscribeDiarizeAdapter`，不要改造 `QwenAsrAdapter` 去兼容 MOSS。
- 模型加载、prompt、输出解析、speaker segment 映射都放在 MOSS adapter 内。
- FastAPI 路由、JobManager、生命周期管理和错误信封尽量复用。

### 2.4 能力声明必须保守

MOSS 原生输出段级时间戳和匿名说话人标签，但这不等于当前 PRD 里的 `word`/`char` timestamps，也不等于 forced alignment。

推论：

- 第一版不要声明 `forced_alignment=true`。
- 第一版不要声明 `streaming=true`。
- 第一版不要把 MOSS 段级 diarization 当成 Qwen 的 word/char timestamps。
- 如果扩展 capabilities，使用更精确的字段，例如 `diarization=true`、`segment_timestamps=true`。
- 如果暂时不扩展 capabilities schema，则只在 `verbose_json` 中返回 `segments`，同时在文档中说明发现能力的限制。

### 2.5 长音频切分会改变 speaker label 语义

MOSS 的 `[S01]`、`[S02]` 是输入音频上下文内的匿名 speaker label。当前服务会把长音频切分成 chunk 后逐段转写。跨 chunk 后，`S01` 不一定指同一个真实说话人。

推论：

- 第一版 MOSS 可以复用现有切分和异步 job，但必须返回 warning。
- 如果音频被切分，MOSS speaker label 只保证 chunk 内有效，不保证跨 chunk 身份一致。
- 会议、访谈、播客这类强依赖 diarization 的场景，后续需要专门的 speaker stitching 方案。

## 3. 兼容性结论

### 3.1 兼容项

| 项目 | 当前 asr_server | MOSS 要求 | 结论 |
| --- | --- | --- | --- |
| Python | 3.12 | >=3.10 | 兼容 |
| 依赖管理 | uv | pip/uv 均可 | 兼容 |
| 服务框架 | FastAPI/Uvicorn | 官方 web app 也用 FastAPI/Uvicorn | 兼容 |
| GPU runtime | WSL CUDA torch | PyTorch CUDA | 兼容 |
| 模型库 | transformers | transformers + remote code | 兼容，但要补依赖 |
| 音频工具 | ffmpeg/librosa/soundfile | librosa/soundfile/av/soxr | 需要补 `av`、`soxr` |
| 长任务 | 内存 JobManager + FIFO | 长音频推理可能耗时 | 兼容 |

### 3.2 不兼容项

| 问题 | 原因 | 处理 |
| --- | --- | --- |
| 当前 Qwen adapter 不能直接加载 MOSS | MOSS 使用 `AutoModelForCausalLM` 和 custom remote code | 新增 MOSS adapter |
| 当前 `TranscriptionResult` 不携带 speaker segments | MOSS 的价值在 diarization 和 segment timestamps | 扩展结果结构 |
| 当前 `timestamps` 参数只接受 `none`、`word`、`char` | MOSS 段级 timestamp 不属于这三类 | 增加 `segments` 返回，不把它伪装成 word/char timestamps |
| 当前全局 `ASR_ADAPTER=qwen` 不适合多真实模型 | adapter factory 现在按全局模式返回 adapter | 增加 model-id dispatch 或显式 MOSS gate |
| 当前 chunk merge 不处理 speaker label 跨 chunk 一致性 | MOSS speaker label 是相对标签 | 第一版 warning，后续做 speaker stitching |

## 4. 接入范围

### 4.1 第一版必须做

1. 新增 `MossTranscribeDiarizeAdapter`。
2. 新增模型注册项 `moss-transcribe-diarize-0.9b`，默认关闭或通过环境变量开启。
3. 新增 WSL 独立 smoke 脚本，在不启动 FastAPI 服务的情况下跑通一次 MOSS 转写。
4. 扩展 `TranscriptionResult`，支持 adapter 返回 `segments`。
5. `response_format=verbose_json` 时返回 MOSS parsed segments。
6. 保留现有健康检查、模型列表、加载、卸载、同步转写、异步 job 语义。
7. 保留 Qwen3-ASR 默认模型和现有 Qwen 验收路径。

### 4.2 第一版暂不做

- 不接入 vLLM。
- 不接入 SGLang Omni。
- 不做公网暴露。
- 不做 Web UI。
- 不做多 GPU 调度。
- 不做 speaker identity stitching。
- 不把 MOSS 段级 timestamps 声明为 word/char timestamps。
- 不把 MOSS diarization 声明为 forced alignment。

## 5. 建议模型注册

建议新增内部模型 ID：

```text
moss-transcribe-diarize-0.9b
```

映射到 Hugging Face repo：

```text
OpenMOSS-Team/MOSS-Transcribe-Diarize
```

第一版能力建议：

```json
{
  "transcription": true,
  "streaming": false,
  "timestamps": [],
  "forced_alignment": false,
  "languages": ["auto", "zh", "en"],
  "chinese_dialects": [],
  "backends": ["transformers"],
  "diarization": true,
  "segment_timestamps": true
}
```

如果暂时不扩展 `ModelCapabilities` schema，则使用兼容方案：

```json
{
  "transcription": true,
  "streaming": false,
  "timestamps": [],
  "forced_alignment": false,
  "languages": ["auto", "zh", "en"],
  "chinese_dialects": [],
  "backends": ["transformers"]
}
```

并在接口文档中说明：`moss-transcribe-diarize-0.9b` 在 `verbose_json` 响应中可返回 `segments[].speaker`、`segments[].start`、`segments[].end`。

## 6. 代码实施方案

### 6.1 文件结构

新增：

```text
asr_server/adapters/moss.py
scripts/moss_backend_smoke.py
tests/test_moss_adapter.py
docs/validation-YYYY-MM-DD-wsl-moss.md
```

修改：

```text
asr_server/adapters/base.py
asr_server/config.py
asr_server/main.py
asr_server/registry.py
asr_server/transcription.py
asr_server/audio/merger.py
requirements/wsl-gpu-cu128.txt
docs/docs-endpoint-capabilities.md
```

### 6.2 Adapter dispatch

当前 `ASR_ADAPTER` 是全局模式。为了不破坏现有部署，建议保持兼容：

```text
ASR_ADAPTER=mock  -> 所有模型使用 MockAsrAdapter，用于 macOS 测试
ASR_ADAPTER=qwen  -> 当前生产路径，默认只注册 Qwen 模型
ASR_ENABLE_MOSS=1 -> 额外注册 MOSS，并按 model_id dispatch 到 Moss adapter
```

adapter factory 目标行为：

```text
model_id starts with qwen3-asr -> QwenAsrAdapter
model_id == moss-transcribe-diarize-0.9b -> MossTranscribeDiarizeAdapter
otherwise -> model_not_found 或配置错误
```

这样可以保留当前 `ASR_ADAPTER=qwen` 的生产习惯，同时用 `ASR_ENABLE_MOSS=1` 控制新模型是否进入 `/v1/models`。

### 6.3 MOSS adapter 加载

真实加载必须在 worker 子进程内发生，避免 GPU 内存释放问题污染主 FastAPI 进程。

加载逻辑：

```python
model = AutoModelForCausalLM.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    trust_remote_code=True,
    dtype="auto",
).to(dtype=torch.bfloat16).to("cuda").eval()

processor = AutoProcessor.from_pretrained(
    "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    trust_remote_code=True,
)
```

规则：

- `device` 第一版只支持 `cuda` / `cuda:0`。
- `dtype=auto` 默认映射到 `torch.bfloat16`。
- `dtype=bfloat16` / `bf16` 映射到 `torch.bfloat16`。
- `dtype=float16` / `fp16` 映射到 `torch.float16`。
- 不支持的 dtype 返回 `422 capability_not_supported`。
- `backend` 第一版只支持 `transformers`。

### 6.4 MOSS adapter 推理

优先复用官方 helper：

```python
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
)
```

推理流程：

```python
messages = build_transcription_messages(audio_path, prompt=prompt)
result = generate_transcription(
    model,
    processor,
    messages,
    max_new_tokens=max_new_tokens or 2048,
    do_sample=False,
    device=device,
    dtype=dtype,
)
segments = parse_transcript(result["text"])
```

`context` 和 `hotwords` 映射：

- 如果用户传了 `context`，追加到默认 prompt 后面。
- 如果用户传了 `hotwords`，追加 `热词提示：...`。
- prompt 总长度仍受 `MAX_CONTEXT_CHARS` 约束。

### 6.5 响应映射

建议把 adapter 层返回扩展为：

```python
@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None

@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration: float
    language: str
    warnings: list[str]
    segments: list[TranscriptionSegment] = field(default_factory=list)
    timings: TranscriptionTimings = field(default_factory=TranscriptionTimings)
```

MOSS 返回：

```json
{
  "text": "拼接后的纯文本",
  "segments": [
    {
      "start": 0.48,
      "end": 1.66,
      "speaker": "S01",
      "text": "Welcome everyone"
    }
  ]
}
```

如果请求 `response_format=text`，只返回拼接后的纯文本。

如果请求 `response_format=json`，返回纯文本和基础 metadata，可以不返回完整 segments。

如果请求 `response_format=verbose_json`，返回完整 `segments`。

### 6.6 长音频切分和合并

短音频不切分时，MOSS parsed segments 可直接返回。

被服务端切分时：

1. 每个 chunk 内调用 MOSS。
2. adapter 返回 chunk-local segments。
3. merger 将 segment `start` / `end` 加上 chunk 的原始 `start` offset。
4. 合并所有 segment 到全局时间线。
5. 返回 warning：

```text
moss_speaker_labels_are_chunk_local
```

含义：时间线是全局的，但 speaker label 不保证跨 chunk 身份一致。

## 7. WSL 部署步骤

以下步骤必须在 WSL Arch Linux 内执行，项目目录为：

```bash
cd /home/fragt/services/asr-server
```

### 7.1 同步代码

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
```

### 7.2 基础系统依赖

```bash
sudo pacman -Syu --needed git ffmpeg libsndfile
```

### 7.3 Python 和项目依赖

```bash
uv python install 3.12
uv sync
```

### 7.4 CUDA torch 验收

不要裸装 CPU 版 torch。先使用官方 CUDA wheel 源：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

验收：

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.version.cuda is not None, "installed torch is CPU-only"
assert torch.cuda.is_available(), "torch cannot access CUDA"
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

### 7.5 安装 MOSS 依赖

MOSS 官方包依赖 `transformers>=5.0.0,<6.0.0`、`av`、`librosa`、`soundfile`、`soxr` 等。建议先使用项目 requirements 管理基础 GPU 依赖，再补 MOSS 包。

```bash
uv pip install --torch-backend cu128 -r requirements/wsl-gpu-cu128.txt
uv pip install -U av soxr
uv pip install -U "moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git"
```

安装后再次验收 torch：

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

如果出现 CPU 版 torch，删除 `.venv` 后从 CUDA torch 步骤重建，不要继续在错误环境上叠装。

### 7.6 独立 smoke

在接入 FastAPI 服务前，必须先跑独立 smoke：

```bash
uv run python scripts/moss_backend_smoke.py \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio test-fixtures/audio/test_short.wav \
  --max-new-tokens 2048
```

通过条件：

- 模型成功加载到 CUDA。
- 输出 `text` 非空。
- `parse_transcript()` 能解析出 0 个或多个 segment。
- 不要求测试短音频一定有多 speaker，但如果有 segment，字段必须包含 `start`、`end`、`speaker`、`text`。

验收记录写入：

```text
docs/validation-YYYY-MM-DD-wsl-moss.md
```

至少记录：

- 模型 ID。
- 后端。
- 音频文件。
- torch 版本。
- CUDA 版本。
- GPU 名称。
- transformers 版本。
- MOSS package 版本或 commit。
- `max_new_tokens`。
- 输出文本前 200 字。
- parsed segment 数量。
- 第一条 segment 示例。

### 7.7 服务启动

MOSS 未验收前，不打开 `ASR_ENABLE_MOSS`：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

MOSS 验收通过并完成 adapter 接入后：

```bash
ASR_ADAPTER=qwen ASR_ENABLE_MOSS=1 \
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

`18080` 仍是唯一面向局域网的 API 端口。不要开放 MOSS 内部 worker 端口，不要使用历史测试端口 `8765`。

## 8. Mac 侧验收命令

Mac mini 只作为客户端验收机，请求局域网地址时绕过本机代理。

健康检查：

```bash
curl --noproxy '*' http://192.168.31.137:18080/health
```

模型发现：

```bash
curl --noproxy '*' http://192.168.31.137:18080/v1/models
```

MOSS 转写：

```bash
curl --noproxy '*' -X POST http://192.168.31.137:18080/v1/audio/transcriptions \
  -F model=moss-transcribe-diarize-0.9b \
  -F file=@test-fixtures/audio/test_short.wav \
  -F response_format=verbose_json \
  -F max_new_tokens=2048
```

通过条件：

- HTTP 200。
- `model` 为 `moss-transcribe-diarize-0.9b`。
- `backend` 为 `transformers`。
- `text` 非空。
- 如果实现了正式 segments 映射，`segments` 中对象字段符合 `start`、`end`、`speaker`、`text`。
- 如果音频被切分，`warnings` 包含 speaker label chunk-local 说明。

## 9. 对抗式审查

### 9.1 为什么不直接用官方 SGLang Omni？

反方质疑：模型页推荐 SGLang Omni，为什么不用官方推荐路径？

结论：第一版不采用。

理由：

- 当前项目的核心是单一局域网 ASR 网关和统一模型生命周期。
- SGLang 会引入第二套 serving 进程、端口、并发、显存和日志管理。
- 当前 PRD 明确第一版只走 `transformers`，vLLM/SGLang 属于后续路线。
- 先用 adapter 接入可以复用已有上传、鉴权、job、卸载和错误语义。

后续条件：

- MOSS transformers 路径可用但吞吐不足。
- 有明确长音频批处理吞吐需求。
- 已经完成 torch 依赖锁定，能防止 serving 栈替换 CUDA torch。

### 9.2 为什么不直接用 vLLM？

反方质疑：模型页提供 vLLM OpenAI-compatible transcription API，直接起 vLLM 不更省事吗？

结论：第一版不采用。

理由：

- 模型页要求 pinned vLLM nightly wheel。
- vLLM 安装可能替换 torch，和 RTX 5070 Ti CUDA 环境风险冲突。
- 当前项目 PRD 明确 vLLM 后续再做。
- 直接暴露 vLLM 会绕过现有 `/v1/models` 能力治理和生命周期锁。

### 9.3 `trust_remote_code=True` 是否可接受？

反方质疑：remote code 有供应链风险。

结论：可以接受，但必须收窄边界。

约束：

- 只允许服务端固定加载 `OpenMOSS-Team/MOSS-Transcribe-Diarize`。
- 不允许客户端传任意 Hugging Face repo id。
- WSL 验收记录中写明使用的 model revision 或 commit。
- 后续生产稳定后，可以 pin 到具体 revision。

### 9.4 16GB VRAM 是否够？

反方质疑：RTX 5070 Ti 16GB VRAM 可能不够长音频和大 token 输出。

结论：短音频和分段推理大概率可行，长音频需要限制。

约束：

- 第一版默认 `max_new_tokens=2048` 或 `4096`。
- 服务端继续保留 `MAX_NEW_TOKENS` 上限。
- 长音频走异步 job。
- 如果用户显式要求更大 `max_new_tokens`，必须在 WSL 记录显存和耗时验收。

### 9.5 MOSS 是否会破坏 Qwen？

反方质疑：新增 MOSS 依赖可能影响 Qwen 已跑通路径。

结论：必须通过隔离和回归验收控制。

措施：

- MOSS 用独立 adapter 文件。
- Qwen 默认模型不变。
- MOSS 默认不进入 `/v1/models`，通过 `ASR_ENABLE_MOSS=1` gate。
- MOSS 安装后重新跑 Qwen 0.6B 和 1.7B smoke，确认 torch 没被替换。

### 9.6 diarization segments 是否能跨 chunk 可信？

反方质疑：切分后 speaker label 可能错。

结论：第一版只能保证 chunk 内 speaker label，不保证跨 chunk 一致性。

措施：

- 返回 warning。
- verbose chunks 中保留 chunk index。
- 后续做 speaker stitching 前，不宣传跨 chunk speaker identity。

### 9.7 是否应该把 MOSS 设为默认模型？

反方质疑：MOSS 有 diarization 和 timestamp，功能更多，是否应该默认使用？

结论：不应该。

理由：

- 当前 PRD 初版默认模型是 Qwen3-ASR 1.7B。
- 默认模型变化会影响现有客户端预期。
- MOSS 的输出结构更复杂，长音频 speaker label 语义也不同。
- 应由客户端通过 `/v1/models` 发现后显式选择。

## 10. 验收清单

WSL 环境：

- `nvidia-smi` 可见 RTX 5070 Ti。
- `torch.version.cuda is not None`。
- `torch.cuda.is_available() is True`。
- `torch.cuda.get_device_name(0)` 为 RTX 5070 Ti。
- 安装 MOSS 依赖后 torch 仍是 CUDA 版。

独立模型：

- `scripts/moss_backend_smoke.py` 成功。
- 输出文本非空。
- parsed segments 结构有效。
- 验收记录已写入 `docs/validation-YYYY-MM-DD-wsl-moss.md`。

服务接口：

- `/health` 返回 200。
- `/v1/models` 在 `ASR_ENABLE_MOSS=1` 时出现 `moss-transcribe-diarize-0.9b`。
- `POST /v1/audio/transcriptions` 使用 MOSS 返回非空文本。
- `response_format=text` 返回纯文本。
- `response_format=verbose_json` 返回 segments。
- 请求 `timestamps=word` 或 `timestamps=char` 时，如果未声明支持，返回 `capability_not_supported`。
- 请求 `backend=vllm` 时返回 `capability_not_supported`。

生命周期：

- `POST /v1/models/moss-transcribe-diarize-0.9b/load` 可加载。
- `GET /v1/models/moss-transcribe-diarize-0.9b/status` 能看到 `loaded`。
- `DELETE /v1/models/moss-transcribe-diarize-0.9b` 可卸载。
- 活跃请求中卸载进入 `unloading_scheduled`。
- `unloading_scheduled` 状态下新请求返回 409。

回归：

- Qwen3-ASR 0.6B smoke 仍通过。
- Qwen3-ASR 1.7B smoke 仍通过。
- macOS mock 测试仍通过。

## 11. 回滚方案

如果 MOSS 接入后出现依赖冲突或显存问题：

1. 关闭 `ASR_ENABLE_MOSS`。
2. 重启服务，只保留 Qwen 模型注册。
3. 如果 torch 被依赖污染，删除 `.venv` 后按 CUDA torch 验收步骤重建。
4. 不回滚 Qwen adapter 和公共 API，除非变更直接破坏 Qwen 测试。

最小回滚启动命令：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

## 12. 推荐实施顺序

1. 在 Mac 侧提交本计划书。
2. 在 Mac 侧实现 schema、adapter skeleton、mock 测试和 smoke 脚本。
3. 推送远端。
4. WSL 侧拉取代码。
5. WSL 侧重建或校验 CUDA torch 环境。
6. WSL 侧安装 MOSS 依赖。
7. WSL 侧运行独立 smoke。
8. WSL 侧运行 FastAPI 服务端验收。
9. Mac mini 通过局域网调用 `18080` 做客户端验收。
10. 验收通过后，再考虑把 MOSS 从实验 gate 纳入正式模型列表。
