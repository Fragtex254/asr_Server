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
docs/wsl-next-plan.md
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

## 下一阶段开发顺序

完成 Qwen `transformers` 最小链路后，按 `docs/wsl-next-plan.md` 推进：

1. 转录耗时记录。
2. 长音频切分与合并。
3. Qwen `transformers` 能力补全。
4. MiMo `transformers` 调研。

不要先做 MiMo，也不要回到 vLLM。

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
