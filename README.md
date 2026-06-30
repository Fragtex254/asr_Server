# WSL ASR 服务

这个仓库包含一个局域网 ASR 网关、部署脚本、代理提示词和验收测试。目标服务运行在 Windows WSL Arch Linux 内，由 Mac mini 项目通过 HTTP 调用。

## 目录结构

- `docs/asr-server-prd.md`：产品需求和 API 合约。
- `prompts/server-agent.md`：给 WSL Arch Linux 服务端开发代理的实现提示词。
- `prompts/wsl-project-brief.md`：给 WSL 侧代理开工前阅读的项目总览提示词。
- `prompts/request-client-agent.md`：给 Mac 侧客户端项目的接入提示词。
- `asr_server/`：FastAPI 应用、模型注册表、生命周期管理器、异步 job 队列和 ASR 适配器。
- `tests/`：不依赖 CUDA 的 API、生命周期、临时文件清理和 job 行为测试。
- `scripts/asr_client.py`：Mac 侧验证客户端，会绕过本机代理设置。
- `scripts/qwen_asr_backend_smoke.py`：WSL 侧真实 Qwen3-ASR 后端最小验收脚本。
- `scripts/wsl_smoke.sh`：WSL 侧启动服务并运行 HTTP smoke test 的脚本。
- `deploy/`：WSL 一键部署脚本、systemd user service 和 Windows 启动脚本模板。
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

当前代码已经实现 FastAPI 网关、mock 适配器、Qwen 真实适配器、模型生命周期管理、同步转录、异步转录 job、单 worker FIFO 队列、chunk 级进度、长音频切分与合并、上传大小限制和统一错误信封。

转录过程中产生的上传文件、解码文件和 Qwen 临时音频文件只用于当次请求。同步请求在返回前清理临时文件；异步 job 在完成、失败或取消后清理 job 工作目录，只在内存里按 `ASR_JOB_RESULT_TTL_SECONDS` 保留 job 结果。

模型默认 lazy load。转录完成后如果 `ASR_IDLE_UNLOAD_SECONDS` 秒内没有新的同模型转录请求，服务会自动卸载该模型并释放 CUDA cache；默认值为 `180` 秒。

真实 Qwen3-ASR 模型依赖、CUDA 验证和常驻部署仍然只在 WSL Arch Linux 侧完成，Mac mini 只作为轻量客户端和验收机。

## 本地开发

Python 和依赖以 uv 为准：

```bash
uv sync
uv run pytest -q
uv run mypy asr_server tests scripts
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

WSL 真实部署优先使用一键脚本：

```bash
deploy/wsl-deploy.sh
```

常驻 systemd 服务和 Windows 启动任务使用部署目录里的 `.venv/bin/uvicorn` 直接启动，不使用 `uv run uvicorn`，避免服务启动时因依赖同步访问 PyPI 镜像而卡住。

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
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 transcribe /path/to/audio.wav --context "Qwen3-ASR, Silero VAD, Hugging Face" --max-new-tokens 512
```

接口使用原则：

- 短音频或脚本验证可以继续使用 `POST /v1/audio/transcriptions` 同步接口。
- 长音频或需要前端进度展示时使用 `POST /v1/audio/transcription-jobs`。
- job 进度只承诺服务端阶段和 chunk 级真实进度，不承诺单个 Qwen chunk 内部百分比。
