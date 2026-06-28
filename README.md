# WSL ASR 服务

这个仓库包含一个局域网 ASR 网关的规划文档、代理提示词、FastAPI 服务骨架和测试。目标服务运行在 Windows WSL Arch Linux 内，由 Mac mini 项目通过 HTTP 调用。

## 目录结构

- `docs/asr-server-prd.md`：产品需求和 API 合约。
- `prompts/server-agent.md`：给 WSL Arch Linux 服务端开发代理的实现提示词。
- `prompts/request-client-agent.md`：给 Mac 侧客户端项目的接入提示词。
- `asr_server/`：FastAPI 应用、模型注册表、生命周期管理器和 mock ASR 适配器。
- `tests/`：不依赖 CUDA 的 API 和生命周期行为测试。
- `scripts/asr_client.py`：Mac 侧验证客户端，会绕过本机代理设置。
- `scripts/qwen_asr_backend_smoke.py`：WSL 侧真实 Qwen3-ASR 后端最小验收脚本。
- `scripts/wsl_smoke.sh`：WSL 侧启动服务并运行 HTTP smoke test 的脚本。
- `deploy/`：systemd user service 和 Windows 启动脚本模板。
- `test-fixtures/audio/`：给 WSL 侧 ASR 自测使用的音频样本。

## 部署目标

服务计划运行在 WSL Arch Linux 内：

```text
/home/fragt/services/asr-server
```

局域网公开 API 入口：

```text
http://192.168.31.137:18080
```

Mac 侧请求这个局域网入口时必须绕过本机代理，例如：

```bash
curl --noproxy '*' http://192.168.31.137:18080/health
```

## 当前状态

仓库已经包含可在 macOS 上运行的 FastAPI 服务骨架、mock ASR 适配器、生命周期管理器和 API 测试，不需要 CUDA 或模型下载。

真实 Qwen3-ASR 模型依赖、CUDA 验证和正式部署仍然由 WSL Arch Linux 侧完成。

## 本地开发

Python 和依赖以 uv 为准：

```bash
uv sync
uv run pytest -q
uv run mypy asr_server tests scripts
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

Python 通过 `.python-version` 和 `pyproject.toml` 固定为 3.12；具体 Python 包版本锁定在 `uv.lock`。

如果某台机器必须先用 conda，只用 conda 创建外层 Python/uv 环境：

```bash
conda env create -f environment.yml
conda activate asr-server
uv sync
```

不要在 Mac mini 上安装 CUDA、Qwen 模型包或模型缓存。

## Mac 侧验证客户端

辅助客户端使用 `httpx.Client(trust_env=False)` 禁用环境代理。

```bash
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 check
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 transcribe /path/to/audio.wav --model qwen3-asr-1.7b --backend auto
```
