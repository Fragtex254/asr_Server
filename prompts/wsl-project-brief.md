# WSL 侧代理项目总览提示词

你是在 Windows PC 的 WSL Arch Linux 内接手这个项目的开发代理。请先用这份总览理解项目，再阅读细节文档。

## 你要完成什么

这个项目是一个局域网 ASR 网关。Mac mini 本地项目通过 HTTP 上传音频，WSL Arch Linux 内的服务使用 Windows PC 上的 RTX 5070 Ti GPU 跑 Qwen3-ASR，然后把转录结果返回给 Mac。

最终服务入口：

```text
http://192.168.31.137:18080
```

WSL 内项目目录：

```text
/home/fragt/services/asr-server
```

不要把项目部署在 `/mnt/c`。不要使用历史测试端口 `8765`。

## 当前仓库已经有什么

Mac 侧已经提前做好了这些不依赖 GPU 的工作：

- FastAPI 服务骨架：`asr_server/main.py`
- 模型注册表：只包含 `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b`
- 生命周期管理：加载、卸载、活跃请求计数、`unloading_scheduled`
- mock ASR 适配器：用于无 GPU 环境测试
- Qwen 真实适配器骨架：`asr_server/adapters/qwen.py`
- HTTP smoke test：`tests/test_http_smoke.py`
- WSL 一键 smoke 脚本：`scripts/wsl_smoke.sh`
- Qwen 后端最小验收脚本：`scripts/qwen_asr_backend_smoke.py`
- WSL 部署文档：`docs/wsl-deployment.md`
- 验收记录模板：`docs/validation-template.md`
- 测试音频：`test-fixtures/audio/test_short.wav` 和 `test-fixtures/audio/test_long.mp3`

Mac 侧代码默认使用 `ASR_ADAPTER=mock`，不会加载真实模型。WSL 真实部署时使用：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

## 你必须先读

按顺序阅读：

```text
AGENTS.md
docs/asr-server-prd.md
docs/wsl-deployment.md
prompts/server-agent.md
docs/validation-template.md
```

## 初版范围

初版只做 Qwen3-ASR 两个尺寸：

```text
qwen3-asr-0.6b
qwen3-asr-1.7b
```

MiMo-V2.5-ASR 不在初版范围内。不要把 MiMo 写进 `/v1/models`，也不要把 MiMo 转写当作验收项。

高级能力如时间戳、强制对齐、WebSocket 流式转写，只有真实实现并验收后才能声明。没验收就不要在 `/v1/models` 的 capabilities 里打开。

## 最容易出错的地方

RTX 5070 Ti 是较新的显卡。不要直接裸跑：

```bash
pip install torch
```

这很容易装成 CPU 版 torch。正确流程是先用 PyTorch 官方 CUDA wheel 源安装并验收 CUDA 版 torch，例如 CUDA 12.8 的 `cu128`：

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

`nvidia-smi` 只能证明 WSL 能看到显卡，不代表 Python 里的 torch 是 CUDA 版。必须跑上面的 Python 验收。

## 服务端开发前置门槛

在写或修改真实 Qwen adapter 之前，先脱离服务端代码跑通最小转录流程：

```bash
uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

用户口头说的 `tf` 在本项目里表示 `transformers`，不是 TensorFlow。

`transformers` 后端返回非空文本后，才能开始把真实推理接入服务端 adapter。第一版不要在 `/v1/models` 中声明 `vllm`。

## 验收路径

建议执行顺序：

1. 在 `/home/fragt/services/asr-server` 准备项目。
2. `uv sync`
3. `uv run pytest -q`
4. `uv run mypy asr_server tests scripts`
5. 验证 CUDA 版 torch。
6. 安装并验收 `qwen-asr`。
7. 跑通 `scripts/qwen_asr_backend_smoke.py --backend transformers`。
8. 启动真实服务：`ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080`
9. 运行：`ASR_BASE_URL=http://127.0.0.1:18080 uv run pytest tests/test_http_smoke.py -q`
10. 从 Mac mini 用 `curl --noproxy '*'` 验收局域网入口。
11. 把结果填写到 `docs/validation-template.md` 对应格式里。

## 下一阶段开发计划

下一版只围绕 Qwen3-ASR `transformers` 后端完善主转录链路。不要做 vLLM、streaming、MiMo、Web UI 或公网访问。

### Issue 1：升级 Silero VAD

目标：把当前 RMS 能量阈值 VAD 升级成 Silero VAD，让长音频切分更接近真实人声边界。

实现要求：

- 保留当前能量 VAD 作为 fallback，命名为 `energy` 或内部 fallback，不要直接删除。
- 新增 `split_strategy=auto|none|fixed|silero|energy`。`vad` 可以作为兼容别名映射到 `silero`，但响应里要能说明实际策略。
- `auto` 策略：短音频不切分；超过 soft chunk 后优先 Silero；Silero 不可用或失败时 fallback 到 energy；再失败才 fixed。
- Silero 依赖只能在 WSL 真实环境中启用。Mac/mock 环境的基础 import 和测试不能要求安装 CUDA、torch GPU 包或模型缓存。
- 如果使用 PyTorch 版 Silero，必须复用已验收的 CUDA/torch 环境；如果采用 ONNX Runtime，必须记录依赖和模型缓存位置。
- 返回 `split.strategy`、`split.requested_strategy`、`split.vad_backend`、`chunk_count`、`overlap_seconds`，方便对比。

测试要求：

- 无 Silero 依赖时，Mac/mock 测试仍通过，并 fallback 到 energy/fixed。
- `split_strategy=silero` 在 Silero 不可用时返回稳定错误或明确 fallback；不要静默产生空 chunk。
- `split_strategy=energy` 继续覆盖当前 RMS VAD 行为。
- `auto` 对短音频不切分，对长音频优先走 Silero。
- chunk 时间线有序，chunk 时长不超过 hard limit，overlap 小于 chunk 长度。

验收要求：

- 用 `test-fixtures/audio/test_long.mp3` 同时跑 energy 和 Silero，记录 chunk 数、总音频时长、平均 chunk 时长、最长 chunk、空白 chunk 数。
- Silero 输出的 chunk 不应比 energy 明显更碎；如果更碎，必须说明阈值原因。
- 长音频真实 Qwen 转录结果必须能合并为完整文本，且没有明显段落乱序。

### Issue 2：接入 context / 热词 / 专有名词提示

目标：让客户端能按领域传入术语、人名、产品名、项目名，提高专有名词识别稳定性。

实现要求：

- `POST /v1/audio/transcriptions` 增加 `context` 字段，类型为字符串，默认空。
- 可选增加 `hotwords` 字段，支持逗号分隔或 JSON 数组；服务端统一合并成 Qwen `context`。
- 给 context 做长度限制，默认建议不超过 4000 字符；超限返回 `400 bad_request`。
- 长音频切 chunk 后，每个 chunk 都传同一份 context。
- 不要把完整 context 直接写入普通日志；最多记录长度和 hash。
- mock adapter 要在测试中能证明 context 被接收，但不要污染真实转录文本。

测试要求：

- API 能接收 `context` 并传到 adapter。
- 超长 context 返回 `400 bad_request`。
- `response_format=text` 时仍正常返回文本。
- context 不影响无 context 的旧请求。

验收要求：

- 准备一段包含专有名词的测试音频，至少包含 5 个易错词，例如 `Qwen3-ASR`、`Silero VAD`、`Hugging Face`、`RTX 5070 Ti`、`uv`。
- 分别用空 context 和带 context 请求同一音频，记录专有名词命中数量。
- 带 context 的版本不应降低普通文本可读性；如果出现幻觉插词，必须记录。

### Issue 3：让 max_new_tokens 真正生效

目标：避免长 chunk 或复杂音频被默认生成长度截断。

实现要求：

- HTTP 层不再丢弃 `max_new_tokens`。
- 将 `max_new_tokens` 传入 Qwen adapter，并最终传给 `model.transcribe(...)` 或模型加载/生成配置中实际生效的位置。
- 设置服务端上限，默认建议 `4096` 或 WSL 实测后的保守值；超限返回 `400 bad_request`。
- 响应的 `warnings` 中标记是否使用了非默认 `max_new_tokens`。

测试要求：

- mock adapter 能收到 `max_new_tokens`。
- 非法值、超限值被拒绝。
- 未传值时保持当前默认行为。

验收要求：

- 用长 chunk 音频对比默认值和较大 `max_new_tokens`，记录文本是否被截断、推理耗时变化和显存峰值。

### Issue 4：Qwen chunk batch transcription

目标：长音频切分后不要逐 chunk 串行调用 Qwen，优先使用 Qwen 官方批量转录能力提升吞吐。

实现要求：

- 扩展 adapter 协议，支持 `transcribe_batch(chunks, language, context, max_new_tokens)`。
- mock adapter 覆盖 batch 路径；真实 Qwen adapter 使用 `model.transcribe(audio=[...], language=[...], context=[...])` 或官方等效接口。
- 增加配置 `ASR_QWEN_BATCH_SIZE`，默认先保守设为 `1` 或 `2`，WSL 实测后再调整。
- 如果 batch OOM，返回统一 `gpu_unavailable` 或 `inference_failed`，不要让服务崩溃。
- 保留逐 chunk fallback，方便定位问题。

测试要求：

- 多 chunk 请求会走 batch adapter。
- batch 返回数量必须等于 chunk 数；不一致返回 `inference_failed`。
- batch fallback 不改变最终合并文本顺序。

验收要求：

- 用 `test_long.mp3` 对比 batch size 1、2、4 的总耗时、`inference_ms`、峰值显存和错误率。
- 只有 batch size 在 0.6B 和 1.7B 上都稳定，才能写入推荐配置。

### Issue 5：Qwen 错误映射与质量基准

目标：让真实部署出现问题时，Mac 客户端和 WSL agent 能快速判断是依赖、CUDA、显存、模型下载、音频还是 Qwen 推理问题。

实现要求：

- 捕获并映射：CPU 版 torch、CUDA 不可用、CUDA OOM、模型下载失败、`qwen_asr` 缺失、音频解码失败、Qwen 空结果、batch 数量不匹配。
- 所有错误都走统一 error envelope。
- `warnings` 中保留非致命问题，例如 Silero fallback、context 超过推荐长度但未超硬限制、batch fallback。
- `docs/validation-template.md` 补充 Silero、context、max_new_tokens、batch benchmark 记录项。

测试要求：

- 单元测试覆盖主要错误映射。
- 不安装 GPU 依赖的 Mac 环境仍能跑 mock 测试。

验收要求：

- WSL agent 必须在验收记录里写下：模型 ID、backend、VAD backend、batch size、context 是否开启、max_new_tokens、torch 版本、CUDA 版本、GPU 名称、总耗时、推理耗时、chunk 数、文本前 200 字。

### 不进入下一版

- vLLM 后端。
- WebSocket streaming。
- ForcedAligner / word-char timestamps。
- 原生 `*-hf` Transformers 加载路径。
- MiMo。

## 你不要做什么

- 不要开放公网。
- 不要创建公网隧道。
- 不要把 worker 端口暴露给 Mac。
- 不要在活跃请求运行时强行卸载模型。
- 不要把未跑通的模型、后端或高级能力写进 `/v1/models`。
- 不要把下载的模型、缓存、私密音频或 token 提交进 Git。

## 交付标准

你交付时至少说明：

- torch 版本、CUDA 版本、GPU 名称。
- `transformers` 后端最小转录结果。
- `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b` 的服务端 API 转录结果。
- `/health` 和 `/v1/models` 的响应摘要。
- Mac mini 是否能通过 `http://192.168.31.137:18080` 调用服务。
- 如果失败，给出具体命令、错误日志和下一步判断。
