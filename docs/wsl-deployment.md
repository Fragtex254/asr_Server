# WSL Arch Linux 部署说明

## 目标

服务端真实部署只在 Windows PC 的 WSL Arch Linux 内进行，部署目录固定为：

```text
/home/fragt/services/asr-server
```

服务监听：

```text
0.0.0.0:18080
```

Mac 侧访问：

```text
http://192.168.31.137:18080
```

不要把项目放在 `/mnt/c`，不要使用历史测试端口 `8765`。

## 基础准备

```bash
cd /home/fragt/services/asr-server
uv sync
uv run pytest -q
uv run mypy asr_server tests scripts
```

## 一键部署

在 WSL Arch Linux 内，从项目 checkout 目录运行：

```bash
deploy/wsl-deploy.sh
```

默认行为：

- 同步当前项目到 `/home/fragt/services/asr-server`。
- 安装 Arch 系统包 `uv`、`rsync`、`git`、`ffmpeg`、`libsndfile`。
- 创建 Python 3.12 uv 环境并执行 `uv sync --frozen`。
- 运行 `pytest` 和 `mypy`。
- 安装 `requirements/wsl-gpu-cu128.txt` 中固定的 CUDA 12.8 torch、HF native Qwen 运行时依赖和 MOSS 可选运行时依赖。
- 校验 `torch==2.11.0+cu128`、CUDA 可用和 Transformers HF native Qwen/MOSS loading 类可导入。
- 跑一次 `Qwen/Qwen3-ASR-0.6B-hf` + `test-fixtures/audio/test_short.wav` 的 `transformers` 后端 smoke。
- 安装并启动 `systemd --user` 服务 `asr-server.service`。
- 常驻服务通过 `/home/fragt/services/asr-server/.venv/bin/uvicorn` 直接启动，避免 `uv run` 在 systemd 或 Windows 任务计划程序启动时联网同步依赖，导致服务没有监听 `18080`。
- 常驻服务默认设置 `ASR_IDLE_UNLOAD_SECONDS=180`，转录完成后 3 分钟无新增同模型请求就自动卸载模型并释放 CUDA cache。
- 常驻服务默认设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，使用部署 smoke 已拉取并验收过的本地模型缓存，避免运行时被 Hugging Face HEAD 请求超时拖住。
- MOSS 默认不进入 `/v1/models`；只有单独跑通 MOSS smoke 并设置 `ASR_ENABLE_MOSS=1` 后才注册 `moss-transcribe-diarize-0.9b`。
- 对 `http://127.0.0.1:18080` 运行 HTTP smoke test。

如果需要让常驻服务启动后仍可在线拉取 Hugging Face 模型文件：

```bash
deploy/wsl-deploy.sh --online-service
```

如果只想更新代码并重启服务，不重装 GPU 依赖和不重新下载模型：

```bash
deploy/wsl-deploy.sh \
  --skip-system-packages \
  --skip-gpu-deps \
  --skip-cuda-check \
  --skip-qwen-backend-smoke
```

如果只是验证部署流程，不启用真实 Qwen/GPU：

```bash
deploy/wsl-deploy.sh --mock
```

## GPU 运行时依赖

WSL 侧真实运行时统一使用这组 PyTorch CUDA、HF native Qwen 和可选 MOSS 依赖：

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
transformers
accelerate
safetensors
soundfile
librosa
av
soxr
numba==0.65.1
llvmlite==0.47.0
moss-transcribe-diarize
```

安装命令：

```bash
uv pip install --torch-backend cu128 -r requirements/wsl-gpu-cu128.txt
```

不要裸跑 `pip install torch`，也不要让模型相关依赖把 torch 替换成 CPU wheel。安装后必须验收：

```bash
uv run python - <<'PY'
import torch
import torchvision
import torchaudio
import transformers

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.__version__ == "2.11.0+cu128"
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__)
assert hasattr(transformers, "AutoProcessor")
assert hasattr(transformers, "AutoModelForMultimodalLM")
assert hasattr(transformers, "AutoModelForCausalLM")
print("HF native Qwen and MOSS loading classes import ok")
PY
```

交互式调试真实 Qwen adapter 时可以使用：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

常驻服务和 Windows 启动任务不要使用 `uv run uvicorn` 作为启动命令。`uv run` 可能在启动时尝试同步依赖并访问 PyPI 镜像；如果网络超时，systemd 会显示 service 已启动过但 API 端口没有监听。部署脚本和 service 模板统一使用已创建好的虚拟环境入口：

```bash
ASR_ADAPTER=qwen \
ASR_QWEN_BATCH_SIZE=1 \
ASR_IDLE_UNLOAD_SECONDS=180 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/fragt/services/asr-server/.venv/bin/uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

## 可选 MOSS 说话人分离

`moss-transcribe-diarize-0.9b` 是可选模型。依赖可以随 `requirements/wsl-gpu-cu128.txt` 安装，但模型默认不注册，避免在未验收环境里让客户端看到不可用能力。

启用前先在 WSL 侧跑独立 smoke：

```bash
uv run python scripts/moss_backend_smoke.py \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

smoke 必须返回非空文本和可解析 `segments`，并确认安装 MOSS 依赖后 `torch.version.cuda is not None` 且 `torch.cuda.is_available()` 仍为 `True`。

交互式启动 MOSS gate：

```bash
ASR_ADAPTER=qwen ASR_ENABLE_MOSS=1 \
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

常驻 systemd 服务启用 MOSS 时，使用 drop-in，不要直接手改部署脚本生成的 unit：

```bash
mkdir -p ~/.config/systemd/user/asr-server.service.d
cat > ~/.config/systemd/user/asr-server.service.d/10-enable-moss.conf <<'EOF'
[Service]
Environment=ASR_ENABLE_MOSS=1
EOF
systemctl --user daemon-reload
systemctl --user restart asr-server.service
```

启用后验证模型发现和一次转录：

```bash
curl --noproxy '*' -sS http://127.0.0.1:18080/v1/models
curl --noproxy '*' -sS \
  -F file=@test-fixtures/audio/test_short.wav \
  -F model=moss-transcribe-diarize-0.9b \
  -F response_format=verbose_json \
  http://127.0.0.1:18080/v1/audio/transcriptions
```

MOSS 当前只声明 `transformers` 后端。`verbose_json` 会返回 `segments[].speaker`、`segments[].start` 和 `segments[].end`；这不是 PRD 里的 `word`/`char` timestamps，也不是 forced alignment。长音频被切分时，MOSS 的 speaker label 只保证 chunk 内相对一致，响应 `warnings` 会包含 `moss_speaker_labels_are_chunk_local`。

## 可选 Silero VAD

长音频 `split_strategy=auto` 会优先尝试 Silero VAD；如果 WSL 环境没有安装 Silero 依赖，服务会明确 fallback 到 energy VAD，再失败才使用 fixed window。Mac/mock 环境不需要安装 Silero、CUDA torch 或模型缓存。

Silero 依赖只在 WSL 真实环境中安装，并且应在 CUDA 版 torch 验收通过后安装：

```bash
uv pip install silero-vad
```

当前实现通过 `silero_vad` Python 包懒加载模型，不在基础 import 或普通 mock 测试阶段加载；响应中的 `split.vad_backend` 和 `split.warnings` 会记录实际使用的 VAD 后端与 fallback 原因。

## 转录调优参数

同步转录接口和异步 job 接口支持以下调优字段：

- `context`：专有名词、领域背景或热词提示，服务端硬限制 4000 字符。
- `hotwords`：逗号分隔字符串或 JSON 字符串数组，服务端会合并到本次模型提示，普通日志不记录完整内容。
- `max_new_tokens`：可选生成长度上限，Qwen 默认 512，MOSS 默认 2048，服务端硬限制 4096；响应 `warnings` 会标记非默认值。
- `split_strategy`：`auto`、`none`、`fixed`、`silero`、`energy`、`vad`；`vad` 是兼容别名，实际优先走 Silero。

长音频默认按 WSL 实测后的稳定组合执行：

```text
max_chunk_seconds=120
max_new_tokens=512
ASR_QWEN_BATCH_SIZE=1
```

Qwen chunk batch size 通过环境变量配置，默认保守为 `1`：

```bash
ASR_QWEN_BATCH_SIZE=2 ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

只有 batch size 在 `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b` 上都稳定后，才把更大的值写入常驻服务配置。

## 服务端限制参数

以下限制通过环境变量控制：

```text
ASR_MAX_UPLOAD_MB=512
ASR_MAX_QUEUED_JOBS=20
ASR_JOB_RESULT_TTL_SECONDS=3600
ASR_IDLE_UNLOAD_SECONDS=180
```

- `ASR_MAX_UPLOAD_MB`：单次上传音频大小上限；同步转录和异步 job 创建都会执行，超限返回 `413 audio_too_large`。
- 单文件音频时长硬上限为 6 小时；超过返回 `422 duration_limit_exceeded`。
- `ASR_MAX_QUEUED_JOBS`：queued/running job 总数上限；超限返回 `429 job_queue_full`。
- `ASR_JOB_RESULT_TTL_SECONDS`：job 完成、失败或取消后结果保留时间；过期后客户端应重新提交。
- `ASR_IDLE_UNLOAD_SECONDS`：模型最后一次转录结束后等待多少秒自动卸载；默认 `180`，设为 `0` 可关闭自动空闲卸载。

## 异步 job 与进度查询

长音频或需要前端展示进度时，优先使用异步 job：

```text
POST /v1/audio/transcription-jobs
GET /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
```

设计边界：

- 服务端使用内存 JobManager 和单 worker FIFO 队列。
- 可以提交多个 job，但同一时间只运行一个真实模型转录；后续 job 显示 `queued` 和 `queue_position`。
- 进度是服务端阶段和 chunk 级真实进度，包括 `total_chunks`、`completed_chunks`、`current_chunk`。
- 不承诺单个模型 chunk 内部 token/帧级百分比。
- 进程重启后内存 job 可以丢失；客户端应重新提交。
- job 完成、失败或取消后清理上传音频和中间临时文件；TTL 只保留内存中的 job 状态和结果，不保留音频文件。
- job 结果默认保留 1 小时，可用 `ASR_JOB_RESULT_TTL_SECONDS` 调整。

示例：

```bash
curl --noproxy '*' -sS \
  -F file=@test-fixtures/audio/test_long.mp3 \
  -F model=qwen3-asr-1.7b \
  -F backend=auto \
  -F language=auto \
  http://127.0.0.1:18080/v1/audio/transcription-jobs
```

返回 `job_id` 后轮询：

```bash
curl --noproxy '*' -sS http://127.0.0.1:18080/v1/jobs/<job_id>
```

Mac 侧局域网验收同样使用 `http://192.168.31.137:18080`，并且必须绕过本机代理。

当前 WSL 实测记录见：

```text
docs/validation-2026-07-02-wsl-hf-native.md
docs/validation-2026-07-10-wsl-moss.md
```

## 后端预验收

开发或启用真实 Qwen adapter 前，先跑：

```bash
uv run python scripts/qwen_asr_backend_smoke.py --backend transformers --model Qwen/Qwen3-ASR-0.6B-hf --audio test-fixtures/audio/test_short.wav
```

默认 `transformers` smoke 走 HF native `AutoProcessor` + `AutoModelForMultimodalLM`。返回非空文本后，再接入或启用服务端真实 Qwen adapter。当前不在 `/v1/models` 中声明 `vllm`。

启用 MOSS 前还必须跑：

```bash
uv run python scripts/moss_backend_smoke.py --model OpenMOSS-Team/MOSS-Transcribe-Diarize --audio test-fixtures/audio/test_short.wav
```

## HTTP smoke test

服务已启动后运行：

```bash
ASR_BASE_URL=http://127.0.0.1:18080 uv run pytest tests/test_http_smoke.py -q
```

或者一键启动 mock/qwen 服务并跑 HTTP smoke：

```bash
ASR_ADAPTER=qwen scripts/wsl_smoke.sh
```

## 一键启动与停止

正式部署完成后，日常启动和停止统一操作 `systemd --user` 服务，不直接手写 `uvicorn` 命令：

```bash
scripts/start_asr_server.sh
scripts/shutdown_asr_server.sh
```

`scripts/start_asr_server.sh` 会：

- 检查 `asr-server.service` 是否已安装。
- 启动已部署的 `/home/fragt/services/asr-server/.venv/bin/uvicorn` 服务。
- 等待 `http://127.0.0.1:18080/health` 返回成功。
- 打印本机和局域网访问地址。

`scripts/shutdown_asr_server.sh` 会：

- 停止 `asr-server.service`。
- 等待服务进入 inactive。
- 确认 `http://127.0.0.1:18080/health` 不再响应。

可用环境变量覆盖默认值：

```bash
ASR_SERVICE_NAME=asr-server.service
ASR_PORT=18080
ASR_BASE_URL=http://127.0.0.1:18080
ASR_START_WAIT_SECONDS=90
ASR_SHUTDOWN_WAIT_SECONDS=90
```

如果还要在同一个脚本里跑 Qwen `transformers` 后端预验收：

```bash
ASR_ADAPTER=qwen ASR_RUN_QWEN_BACKEND_SMOKE=1 scripts/wsl_smoke.sh
```

如果要让该 smoke 脚本启动的服务暴露 MOSS，可额外设置 `ASR_ENABLE_MOSS=1`。这只影响服务注册，不替代独立的 `scripts/moss_backend_smoke.py`。

## systemd user service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/asr-server.service ~/.config/systemd/user/asr-server.service
systemctl --user daemon-reload
systemctl --user enable --now asr-server.service
systemctl --user status asr-server.service
```

`deploy/asr-server.service` 的 `ExecStart` 必须指向部署目录里的 `.venv/bin/uvicorn`，不要改回 `uv run uvicorn`。服务重启后用下面两个命令同时确认 systemd 状态和端口健康：

```bash
systemctl --user status asr-server.service --no-pager
curl --noproxy '*' http://127.0.0.1:18080/health
```

## Windows 启动任务

`deploy/windows-start-asr.ps1` 可作为 Windows 任务计划程序调用脚本。任务应在用户登录后运行，并确保 WSL 发行版名称与脚本中的 `$Distro` 一致。

## 防火墙

Windows 只需要为专用网络开放 TCP `18080`。不要开放 `8001`、`8765` 或公网入口。
