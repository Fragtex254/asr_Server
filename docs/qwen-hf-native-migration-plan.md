# Qwen3-ASR HF Native Migration Plan

更新时间：2026-07-02

## 1. 目标

把当前 Qwen3-ASR 真实推理接入从 `qwen-asr` 包优先的加载方式，迁移为更规范的 Hugging Face Transformers native adapter，同时保持现有 ASR 网关 API、模型生命周期、异步 job 和错误语义稳定。

这次迁移的第一目标不是新增功能，而是把模型接入边界整理清楚，让后续适配其他 ASR 模型时只新增 adapter 和模型注册，不改公共 API。

官方依据：

- QwenLM/Qwen3-ASR 已在 2026-06-26 公布 Native Transformers support，并说明支持 `torch.compile`。
- 新 HF 模型卡示例使用 `AutoProcessor`、`AutoModelForMultimodalLM` 加载 `Qwen/Qwen3-ASR-0.6B-hf` 和 `Qwen/Qwen3-ASR-1.7B-hf`。
- `Qwen/Qwen3-ForcedAligner-0.6B-hf` 是独立 forced alignment 模型，不等于普通转录模型能力已自动打开。

参考链接：

- https://github.com/QwenLM/Qwen3-ASR
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf
- https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf

## 2. 不变边界

以下内容不在本次迁移中改变：

- 对外 HTTP API 路径和响应结构。
- `docs/asr-server-prd.md` 中定义的错误信封。
- 模型状态枚举：`unloaded`、`loading`、`loaded`、`unloading_scheduled`、`unloading`、`error`。
- 活跃请求计数、生命周期锁、延迟卸载和 `model_unloading_scheduled` 拒绝语义。
- 异步 job 的内存 JobManager、单 worker FIFO 队列、轮询状态和 chunk 级真实进度。
- `18080` 是唯一局域网入口；不开放 worker 端口，不使用历史测试端口 `8765`。
- Mac mini 不安装 CUDA、torch GPU 包、Qwen 模型包或模型缓存。

## 3. 本次范围

必须做：

1. 新增或改造真实 Qwen adapter，使 `transformers` 后端优先使用 HF native 模型：
   - `Qwen/Qwen3-ASR-0.6B-hf`
   - `Qwen/Qwen3-ASR-1.7B-hf`
2. 保留现有 mock adapter，使无 GPU 测试仍能在 macOS 通过。
3. 保留 Qwen worker 子进程隔离，避免 GPU 内存释放问题污染主 FastAPI 进程。
4. 更新模型 repo 映射、smoke 脚本和 WSL 验收文档，使默认真实验收走 `*-hf` 模型。
5. 在 WSL Arch Linux 内验证两个 HF ASR 模型端到端转录成功后，才把 `/v1/models` 中的 `transformers` 后端视为验收通过。

暂不做：

- 不接入 vLLM。
- 不做 Web UI。
- 不做公网暴露。
- 不做数据库队列、Redis、Celery、多 worker 或多 GPU 调度。
- 不把 forced alignment、timestamps、streaming 在 `/v1/models` 中打开。
- 不把 `Qwen/Qwen3-ForcedAligner-0.6B-hf` 混进基础转录 adapter。

## 4. 目标架构

当前项目已经有：

- `asr_server/adapters/base.py`：adapter 协议和转录结果类型。
- `asr_server/adapters/qwen.py`：Qwen worker 子进程和真实 adapter。
- `asr_server/registry.py`：模型能力声明。
- `scripts/qwen_asr_backend_smoke.py`：WSL 侧最小后端验收脚本。

迁移后保持同样的外部形状，但 adapter 内部按后端实现拆清楚：

```text
FastAPI API
  -> Model Lifecycle Manager
  -> AsrAdapter protocol
  -> Qwen adapter parent process
  -> Qwen worker child process
  -> HF Native Transformers backend
  -> CUDA torch on RTX 5070 Ti
```

建议内部命名：

- 对外模型 ID 继续使用 `qwen3-asr-0.6b`、`qwen3-asr-1.7b`。
- 内部 repo ID 改为 `Qwen/Qwen3-ASR-0.6B-hf`、`Qwen/Qwen3-ASR-1.7B-hf`。
- 后端名称仍为 `transformers`，不要新增 `hf` 作为公共 backend 名称。`hf native` 是 adapter 实现细节，不是客户端能力枚举。

## 5. Adapter 实现要求

### 5.1 加载

HF native 加载路径应使用延迟 import，保证基础 import 和 mock 测试不需要 GPU 依赖：

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained(repo_id)
model = AutoModelForMultimodalLM.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
).to("cuda").eval()
```

实现时按项目现有 `dtype` 参数处理：

- `auto`：WSL CUDA 默认 `torch.bfloat16`。
- `bfloat16` / `bf16`：`torch.bfloat16`。
- `float16` / `fp16`：`torch.float16`。
- 其他值返回 `422 capability_not_supported` 或明确参数错误。

`device` 第一阶段只支持 `cuda` / `cuda:0`。CPU fallback 不进入本项目真实 adapter。

### 5.2 转录

优先使用 HF 模型卡上的 processor helper：

```python
inputs = processor.apply_transcription_request(
    audio=audio_input,
    language=language_or_none,
)
inputs = inputs.to(model.device, model.dtype)
output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
decoded = processor.decode(generated_ids, return_format="parsed")
```

adapter 输出仍统一映射到项目的 `TranscriptionResult`：

- `text`：非空转录文本。
- `language`：模型返回语言；没有时回落到请求语言，`auto` 时回落到空字符串或 `auto`。
- `duration`：沿用服务端音频 metadata 或现有测算方式。
- `warnings`：只放可行动的降级信息，不记录完整音频内容。
- `timings`：继续记录加载、推理、后处理耗时。

如果 helper API 在当前 Transformers 版本中不可用，停止实现并记录依赖版本问题；不要用脆弱的字符串 prompt 拼接临时绕过。

### 5.3 Batch / chunk

现有 job 和长音频切分可能会调用 `transcribe_batch`。第一阶段可以保守实现为串行调用单条 `transcribe`，保证语义正确；确认 HF native batch 输入稳定后，再优化为真正 batch。

不要伪造单个 chunk 内部进度。job 进度继续只承诺阶段和 chunk 级真实进度。

### 5.4 卸载

卸载仍在 worker 进程内完成：

1. 删除 model 和 processor 引用。
2. `gc.collect()`。
3. 如请求 `cuda_empty_cache=true`，执行 `torch.cuda.empty_cache()`。
4. 父进程等待 worker 正常退出，超时后按现有逻辑 terminate/kill。

## 6. 依赖安装规则

在 WSL Arch Linux 内执行。不要在 macOS 安装真实推理依赖。

必须先验收 CUDA torch：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.version.cuda is not None, "装到 CPU 版 torch 了"
assert torch.cuda.is_available(), "torch 看不到 CUDA"
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

再安装 HF native 所需依赖。推荐先用当前官方 Transformers 版本；如果 `AutoModelForMultimodalLM` 或 Qwen3-ASR helper 尚未进入 release，再切到 Hugging Face Transformers main：

```bash
uv pip install -U transformers accelerate safetensors soundfile librosa

# 只有 release 不支持 Qwen3-ASR-hf 时才执行：
uv pip install -U "transformers @ git+https://github.com/huggingface/transformers"
```

安装任何模型相关依赖后，都必须再次验收 torch 没被替换成 CPU 版：

```bash
uv run python - <<'PY'
import torch

assert torch.version.cuda is not None, "依赖安装后 torch 变成 CPU 版"
assert torch.cuda.is_available(), "依赖安装后 CUDA 不可用"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

不要裸跑 `pip install torch`。不要为了绕过 RTX 5070 Ti 兼容问题降级到 CPU torch。

## 7. Smoke 脚本改造

更新 `scripts/qwen_asr_backend_smoke.py`：

- 默认模型改为 `Qwen/Qwen3-ASR-0.6B-hf`。
- 新增或改造 `--backend transformers` 路径为 HF native `AutoProcessor` + `AutoModelForMultimodalLM`。
- 可以临时保留旧 `qwen-asr` 包路径，但必须通过参数显式启用，例如 `--loader qwen-asr`；默认不再走旧路径。
- 输出至少包含：
  - model repo ID
  - backend
  - loader
  - torch version
  - torch CUDA version
  - GPU name
  - output language
  - text 前 200 字

建议验收命令：

```bash
uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B-hf \
  --audio test-fixtures/audio/test_short.wav \
  --language auto

uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-1.7B-hf \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

## 8. Registry 与能力声明

`asr_server/registry.py` 的对外能力暂时保持：

```json
{
  "transcription": true,
  "streaming": false,
  "timestamps": [],
  "forced_alignment": false,
  "backends": ["transformers"]
}
```

可以新增内部字段或旁路配置记录 repo ID，但不要改变 API 返回结构，除非 PRD 同步修改。

`/v1/models` 只声明 WSL 真实跑通过的能力。HF 模型卡声称支持 timestamps 或 forced alignment，不等于本服务已经支持。必须等本项目 API、adapter、测试和验收全部完成后再打开。

## 9. ForcedAligner 结论

当前项目没有完成 `Qwen/Qwen3-ForcedAligner-0.6B-hf` 集成。它应作为单独阶段处理。

原因：

- forced alignment 的输入是音频加已有 transcript，不是普通 ASR 的同一条路径。
- 它使用 `AutoModelForTokenClassification`，生命周期、显存占用和返回结构都应独立建模。
- PRD 已把 `POST /v1/audio/alignments` 作为预留接口，并要求未验收时返回 `capability_not_supported`。
- HF 模型卡说明 ForcedAligner 支持 11 种语言、最长约 5 分钟语音的任意单位时间戳预测；这需要额外 API 校验和音频时长限制。

后续 forced alignment 阶段建议另开计划：

1. 新增 `AlignmentAdapter` 或扩展 adapter 协议，不混进基础 `transcribe()`。
2. 新增 forced aligner 模型生命周期，明确是否和 ASR 模型共享加载锁。
3. 实现 `POST /v1/audio/alignments`。
4. 为 `timestamps=word|char` 定义是否调用 ForcedAligner 做二阶段处理。
5. WSL 验收后再把 `forced_alignment=true` 或 `timestamps=["word", "char"]` 写入 `/v1/models`。

## 10. 分阶段执行

### Phase 0：阅读和保护现有行为

1. 阅读：
   - `AGENTS.md`
   - `docs/asr-server-prd.md`
   - `docs/qwen-hf-native-migration-plan.md`
   - `prompts/server-agent.md`
   - `docs/wsl-deployment.md`
   - `docs/validation-template.md`
2. 在 macOS 或 WSL 先跑无 GPU 测试：

```bash
uv run pytest -q
```

不得为了让 HF 迁移通过而删掉生命周期、job、错误处理测试。

### Phase 1：HF native smoke 先跑通

1. 在 WSL CUDA torch 验收通过后安装 HF 依赖。
2. 改造 `scripts/qwen_asr_backend_smoke.py`。
3. 跑通 0.6B-hf 和 1.7B-hf 最小转录。
4. 把结果写入新的 validation 记录或 `docs/validation-YYYY-MM-DD-wsl.md`。

只有 smoke 返回非空文本后，才能继续接入服务 adapter。

### Phase 2：接入服务 adapter

1. 改造 `asr_server/adapters/qwen.py` 内部 repo 映射为 `*-hf`。
2. 将 `transformers` 后端实现替换为 HF native 加载和生成路径。
3. 保持 worker 子进程 IPC payload 不变。
4. 保持 `TranscriptionResult` 对外字段不变。
5. 如保留旧 qwen-asr 包路径，必须隐藏在显式实验配置后面，不作为默认路径。

### Phase 3：测试

必须通过：

```bash
uv run pytest -q
```

重点补充或更新测试：

- registry 仍只暴露 `transformers`，不暴露 `vllm`。
- `timestamps=word|char` 仍返回 `capability_not_supported`。
- forced alignment 未启用时仍不声明能力。
- mock adapter 不依赖 transformers、torch、CUDA。
- Qwen adapter 的 import 是懒加载，不影响无 GPU环境测试。

### Phase 4：WSL 服务验收

启动真实服务：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

本机验收：

```bash
curl --noproxy '*' http://127.0.0.1:18080/health
curl --noproxy '*' http://127.0.0.1:18080/v1/models
ASR_BASE_URL=http://127.0.0.1:18080 uv run pytest tests/test_http_smoke.py -q
```

Mac mini 局域网验收：

```bash
curl --noproxy '*' http://192.168.31.137:18080/health
curl --noproxy '*' http://192.168.31.137:18080/v1/models
```

再分别对 `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b` 发起真实音频转录，记录输出文本前 200 字。

## 11. 验收记录要求

每个模型至少记录：

- 日期时间。
- WSL distro 和内核信息。
- GPU 名称。
- torch 版本。
- CUDA 版本。
- transformers 版本。
- model repo ID。
- audio 文件。
- backend：`transformers`。
- loader：`hf-native`。
- 输出 language。
- 输出 text 前 200 字。
- 是否使用 `torch.compile`。
- 峰值显存，如果方便获取。

`torch.compile` 第一阶段默认关闭。只有 HF native 基础路径稳定后，再作为单独性能优化开关验收。

## 12. 完成定义

本计划完成时必须满足：

- `scripts/qwen_asr_backend_smoke.py` 默认走 HF native，并在 WSL 上跑通 0.6B-hf 和 1.7B-hf。
- `ASR_ADAPTER=qwen` 服务能通过现有 API 转录短音频。
- `/v1/models` 仍只声明已验收能力：offline transcription + `transformers`。
- mock 测试和无 GPU 测试仍通过。
- forced alignment 仍未声明为已支持，相关请求保持明确的 `capability_not_supported` 语义。
- 文档记录 WSL 真实验收结果。
