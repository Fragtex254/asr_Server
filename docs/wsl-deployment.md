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

## GPU 运行时依赖

WSL 侧真实 Qwen3-ASR 运行时统一使用这组 PyTorch CUDA 版本：

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
qwen-asr==0.0.6
numba==0.65.1
llvmlite==0.47.0
```

安装命令：

```bash
uv pip install --torch-backend cu128 -r requirements/wsl-gpu-cu128.txt
```

不要裸跑 `pip install torch`，也不要让 `qwen-asr` 安装过程把 torch 升级到另一组 CUDA wheel。安装后必须验收：

```bash
uv run python - <<'PY'
import torch
import torchvision
import torchaudio
import qwen_asr

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
print("qwen_asr import ok:", qwen_asr.Qwen3ASRModel)
PY
```

真实 Qwen adapter 启动时使用：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

## 可选 Silero VAD

长音频 `split_strategy=auto` 会优先尝试 Silero VAD；如果 WSL 环境没有安装 Silero 依赖，服务会明确 fallback 到 energy VAD，再失败才使用 fixed window。Mac/mock 环境不需要安装 Silero、CUDA torch 或模型缓存。

Silero 依赖只在 WSL 真实环境中安装，并且应在 CUDA 版 torch 验收通过后安装：

```bash
uv pip install silero-vad
```

当前实现通过 `silero_vad` Python 包懒加载模型，不在基础 import 或普通 mock 测试阶段加载；响应中的 `split.vad_backend` 和 `split.warnings` 会记录实际使用的 VAD 后端与 fallback 原因。

## 转录调优参数

同步转录接口支持以下调优字段：

- `context`：专有名词、领域背景或热词提示，服务端硬限制 4000 字符。
- `hotwords`：逗号分隔字符串或 JSON 字符串数组，服务端会合并到 Qwen `context`，普通日志不记录完整内容。
- `max_new_tokens`：可选生成长度上限，默认 512，服务端硬限制 4096；响应 `warnings` 会标记非默认值。
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

本轮 WSL 实测记录见：

```text
docs/validation-2026-06-29-wsl.md
```

## 后端预验收

开发或启用真实 adapter 前，先跑：

```bash
uv run python scripts/qwen_asr_backend_smoke.py --backend transformers --model Qwen/Qwen3-ASR-0.6B --audio test-fixtures/audio/test_short.wav
```

`transformers` 后端返回非空文本后，再接入或启用服务端真实 adapter。第一版不在 `/v1/models` 中声明 `vllm`。

## HTTP smoke test

服务已启动后运行：

```bash
ASR_BASE_URL=http://127.0.0.1:18080 uv run pytest tests/test_http_smoke.py -q
```

或者一键启动 mock/qwen 服务并跑 HTTP smoke：

```bash
ASR_ADAPTER=qwen scripts/wsl_smoke.sh
```

如果还要在同一个脚本里跑 Qwen `transformers` 后端预验收：

```bash
ASR_ADAPTER=qwen ASR_RUN_QWEN_BACKEND_SMOKE=1 scripts/wsl_smoke.sh
```

## systemd user service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/asr-server.service ~/.config/systemd/user/asr-server.service
systemctl --user daemon-reload
systemctl --user enable --now asr-server.service
systemctl --user status asr-server.service
```

## Windows 启动任务

`deploy/windows-start-asr.ps1` 可作为 Windows 任务计划程序调用脚本。任务应在用户登录后运行，并确保 WSL 发行版名称与脚本中的 `$Distro` 一致。

## 防火墙

Windows 只需要为专用网络开放 TCP `18080`。不要开放 `8001`、`8765` 或公网入口。
