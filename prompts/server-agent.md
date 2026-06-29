# 服务端 Agent 提示词

你是在 Windows 主机的 WSL Arch Linux 内工作的开发代理。你的任务是实现一个常驻后台的 ASR 服务，让局域网内 Mac mini 可以访问 Windows/WSL 里的 GPU ASR 模型。

工作目录：

```bash
/home/fragt/services/asr-server
```

目标入口：

```text
http://192.168.31.137:18080
```

必须阅读并遵守 PRD：

```text
docs/asr-server-prd.md
```

下一阶段执行计划已合并到 WSL 项目总览：

```text
prompts/wsl-project-brief.md
```

技术约束：

- 使用 Python 3.12。
- 使用 uv 管理环境。
- 使用 FastAPI + Uvicorn 实现 HTTP API。
- 服务监听 `0.0.0.0:18080`。
- 不要把项目放在 `/mnt/c` 下，放在 WSL 原生文件系统 `/home/fragt/services/asr-server`。
- 第一版优先做稳定的离线转写，不要一开始就实现 Web UI。
- 模型包很重，安装前先检查磁盘空间、CUDA、`nvidia-smi`。
- RTX 5070 Ti 较新，必须显式安装并验证 CUDA 版 torch，防止误装 CPU 版 torch。

初版模型只包含：

- `qwen3-asr-1.7b`
- `qwen3-asr-0.6b`

MiMo-V2.5-ASR 不进入初版交付范围；不要把 MiMo 写进 `/v1/models`，也不要把 MiMo 转写作为初版验收项。

## RTX 5070 Ti / Qwen3-ASR 安装要求

真实推理依赖只能在 WSL Arch Linux 内安装。Mac mini 只做客户端验证，不安装 CUDA、torch GPU 包、Qwen 模型包或模型缓存。

先验收 torch CUDA，再安装 Qwen3-ASR。`nvidia-smi` 只能证明 WSL 能看到驱动和显卡，不能证明 Python 环境里的 torch 是 CUDA 版。不要直接裸跑 `pip install torch`，必须使用 PyTorch 官方 CUDA wheel 源，例如 CUDA 12.8 的 `cu128`：

```bash
cd /home/fragt/services/asr-server

uv python install 3.12
uv sync

sudo pacman -Syu --needed ffmpeg libsndfile

nvidia-smi
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

只有上面的 CUDA 验收通过后，才能安装 Qwen3-ASR：

```bash
uv pip install -U qwen-asr

uv run python - <<'PY'
import torch

assert torch.version.cuda is not None, "qwen-asr 安装后 torch 变成 CPU 版"
assert torch.cuda.is_available(), "qwen-asr 安装后 CUDA 不可用"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

如果已经误装 CPU 版 torch，删除 `.venv` 后按上述顺序重建。不要在错误环境上继续叠装。

如果 RTX 5070 Ti 报 `no kernel image is available`、架构不支持或 CUDA capability 不匹配，不要退回 CPU 版 torch；应改用支持该显卡的更新官方 CUDA wheel 或 PyTorch nightly，并重新跑 CUDA 验收脚本。

`/v1/models` 中只能声明真实跑通过的后端。第一版只声明 `transformers`，`vllm` 延后。

## 服务端开发前的后端预验收

开始开发真实 Qwen adapter 前，必须先脱离服务端代码跑通一次最小 Qwen3-ASR 转录流程：

- `transformers` 后端。用户口头说的 `tf` 在本项目里统一理解为 `transformers`，不是 TensorFlow。

优先使用 0.6B 和短音频样本降低首次验收成本：

```bash
uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

`transformers` 命令必须返回非空文本。第一版不要在 `/v1/models` 中声明 `vllm`。

必须实现的 API：

```text
GET /health
GET /v1/models
GET /v1/models/{model_id}/status
POST /v1/models/{model_id}/load
DELETE /v1/models/{model_id}
DELETE /v1/models
POST /v1/audio/transcriptions
```

`POST /v1/audio/alignments`、WebSocket 流式转写、时间戳等高级能力只在真实实现并验收后再打开能力声明；未打开时返回 `capability_not_supported` 或不暴露入口。

模型状态枚举：

```text
unloaded
loading
loaded
unloading_scheduled
unloading
error
```

卸载语义必须正确：

- 每个模型维护活跃请求计数。
- 每个模型维护生命周期锁。
- 收到卸载请求后，如果活跃请求数为 0，立即卸载。
- 如果活跃请求数大于 0，设置 `unloading_scheduled` 和 `rejecting_new_requests=true`。
- `unloading_scheduled` 状态下拒绝新的同模型请求，返回 409 和 `model_unloading_scheduled`。
- 最后一个活跃请求结束后再卸载模型。
- 卸载后调用 CUDA cache 清理。

建议项目结构：

```text
asr_server/
  __init__.py
  main.py
  config.py
  schemas.py
  errors.py
  registry.py
  lifecycle.py
  adapters/
    __init__.py
    base.py
    qwen.py
tests/
  test_health.py
  test_models.py
  test_lifecycle.py
  test_transcription_api.py
pyproject.toml
README.md
```

开发顺序：

1. 阅读现有 Mac 侧实现和测试，保留 API 合约、错误信封、生命周期语义。
2. 在 WSL Arch Linux 的 `/home/fragt/services/asr-server` 部署项目，不要放在 `/mnt/c`。
3. 检查磁盘空间、CUDA、`nvidia-smi`、Python 3.12、uv。
4. 先用 `scripts/qwen_asr_backend_smoke.py` 跑通 `transformers` 最小转录流程。
5. 接入 Qwen3-ASR 适配器，真实跑通 `qwen3-asr-0.6b` 与 `qwen3-asr-1.7b`。
6. 对 `/v1/models` 中声明的每个后端都做端到端转写验收；若某个后端不能跑通，不要声明它。
7. 保持 mock 适配器测试可在无 GPU 环境通过。
8. 增加 systemd user service 或 Windows 启动任务，让服务可后台常驻。
9. 从 Mac mini 验收局域网调用。
10. 按 `prompts/wsl-project-brief.md` 的“下一阶段开发计划”继续做 Silero VAD、context/热词、max_new_tokens、Qwen batch transcription 和错误映射。不要做 vLLM、streaming、MiMo、ForcedAligner 或 `*-hf` 路径。

测试命令：

```bash
uv run pytest -q
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
curl --noproxy '*' http://127.0.0.1:18080/health
curl --noproxy '*' http://192.168.31.137:18080/v1/models
```

Mac mini 验收命令：

```bash
curl --noproxy '*' -v http://192.168.31.137:18080/health
curl --noproxy '*' -v http://192.168.31.137:18080/v1/models
```

交付物：

- 可运行的 FastAPI ASR 服务。
- README 中写明启动、停止、开机自启、Mac 调用方式。
- 测试覆盖健康检查、模型列表、加载、卸载、卸载等待当前请求完成、转写接口参数校验。
- 给出 `scripts/qwen_asr_backend_smoke.py` 在 `transformers` 后端的最小转录验收结果。
- 给出 `qwen3-asr-0.6b` 在所有声明后端上的真实音频转写验收结果。
- 给出 `qwen3-asr-1.7b` 在所有声明后端上的真实音频转写验收结果。

不要做：

- 不要开放公网。
- 不要默认经过代理访问局域网 IP。
- 不要在活跃请求还没结束时强行卸载模型。
- 不要在 `/v1/models` 中声明未真实跑通的模型、后端或高级能力。
