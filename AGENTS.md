# 代理开发指南

## 项目背景

这个仓库用于实现一个局域网 ASR 网关：Mac mini 客户端通过 HTTP 调用运行在 Windows WSL Arch Linux 内的 GPU ASR 服务。

主要部署目录：

```text
/home/fragt/services/asr-server
```

局域网公开入口：

```text
http://192.168.31.137:18080
```

Mac mini 只作为轻量开发机和客户端验收机。真实 GPU 推理、CUDA 验证、模型包安装、长期后台服务部署，都必须在 Windows PC 的 WSL Arch Linux 内完成。

## 必读文档

实现服务行为前，必须先读：

```text
docs/asr-server-prd.md
```

把任务交给专门代理时，使用这些提示词：

```text
prompts/server-agent.md
prompts/request-client-agent.md
```

## 开发原则

- 公共 API 必须和 `docs/asr-server-prd.md` 保持一致。
- 使用 Python 3.12、uv、FastAPI 和 Uvicorn 实现服务。
- 服务入口监听 `0.0.0.0:18080`。
- WSL 项目位置不要放在 `/mnt/c`；正式部署目录是 `/home/fragt/services/asr-server`。
- 除非 PRD 明确变更，不要把 worker 端口 `8001` 暴露到局域网。
- 不要把历史测试端口 `8765` 写入实现、部署、防火墙或启动配置。
- API、生命周期管理、测试和第一条转录路径完成前，不要做 Web UI。

## 跨平台边界

- macOS 侧可以创建项目骨架、schema、测试、mock 适配器、客户端验证脚本和文档。
- macOS 侧不得要求 CUDA、NVIDIA 驱动、模型下载或大型本地模型缓存。
- WSL Arch Linux 侧负责 CUDA 检查、`nvidia-smi`、磁盘空间检查、真实 Qwen 依赖安装和模型推理验证。
- 大模型依赖必须在适配器内懒加载，保证基础 import 和测试在无 GPU 环境也能运行。
- 路径处理保持 POSIX 兼容，不要在服务代码里硬编码 macOS 专用路径。

## RTX 5070 Ti 与 Qwen3-ASR 环境规则

WSL 侧安装真实 Qwen3-ASR 前，必须先把 RTX 5070 Ti 的 PyTorch CUDA 环境验收通过。这个显卡较新，最常见错误是 agent 直接执行普通 `pip install torch` 或被依赖解析成 CPU 版 torch。

硬性规则：

- 只在 WSL Arch Linux 内安装真实推理依赖；Mac mini 不安装 CUDA、torch GPU 包、Qwen 模型包或模型缓存。
- 使用 Python 3.12 和 uv。不要混用多个 Python、conda 环境和系统 pip。
- 安装 torch 时必须使用 PyTorch 官方 CUDA wheel 源，例如 CUDA 12.8 对应的 `cu128` 源；不要使用没有 CUDA wheel 源的裸 `pip install torch`。
- `nvidia-smi` 只能证明 WSL 能看到驱动和显卡，不能证明 Python 环境里的 torch 是 CUDA 版；必须运行下面的 Python 验收脚本。
- 如果新显卡报 `no kernel image is available`、架构不支持或 CUDA capability 不匹配，不要退回 CPU 版 torch；应改用支持 RTX 5070 Ti 的更新官方 CUDA wheel 或 PyTorch nightly，并重新验收。
- 第一版只安装并验收 `qwen-asr` 的 `transformers` 路径；不要安装 `qwen-asr[vllm]` 作为第一版前置步骤。
- vLLM 后续再做。只有明确切到 vLLM 任务时，才安装 `qwen-asr[vllm]`，并且安装后必须再次验收 torch，防止依赖把 torch 替换成 CPU 版。
- `/v1/models` 中只能声明真实跑通过的后端。第一版只声明 `transformers`，不要声明 `vllm`。

推荐 WSL 安装顺序：

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

uv pip install -U qwen-asr

uv run python - <<'PY'
import torch

assert torch.version.cuda is not None, "qwen-asr 安装后 torch 变成 CPU 版"
assert torch.cuda.is_available(), "qwen-asr 安装后 CUDA 不可用"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

如果已经误装 CPU 版 torch，优先删除 `.venv` 后按上述顺序重建环境；不要在错误环境上继续叠装：

```bash
rm -rf .venv
uv sync
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

## Qwen3-ASR 后端预验收门槛

开发 WSL 服务端真实 adapter 前，必须先脱离本项目服务代码，跑通一次最小 Qwen3-ASR 转录流程：

- `transformers` 后端，用户口头说的 `tf` 在本项目里统一理解为 `transformers`，不是 TensorFlow。

推荐先用 `Qwen/Qwen3-ASR-0.6B` 和 `test-fixtures/audio/test_short.wav` 验收，降低首次下载和显存压力。`transformers` 返回非空文本后，才能开始接入 `asr_server` adapter。第一版不要在 `/v1/models` 中声明 `vllm`。

验收命令：

```bash
uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

验收记录至少写下：模型 ID、后端、音频文件、torch 版本、CUDA 版本、GPU 名称、输出语言和文本前 200 字。

## API 与生命周期规则

- 模型能力发现必须来自 `GET /v1/models`；客户端不要硬编码模型能力。
- 模型状态必须使用 PRD 枚举：`unloaded`、`loading`、`loaded`、`unloading_scheduled`、`unloading`、`error`。
- 每个模型必须维护活跃请求计数和生命周期锁。
- 如果卸载请求到来时仍有活跃请求，设置 `unloading_scheduled`，新的同模型请求返回 `409 model_unloading_scheduled`，等活跃请求完成后再卸载。
- 推理仍在执行时，不要强制卸载模型。
- 异步转录使用 `POST /v1/audio/transcription-jobs`、`GET /v1/jobs/{job_id}`、`DELETE /v1/jobs/{job_id}`。
- 下一阶段异步 job 采用内存 JobManager 和单 worker FIFO 队列；允许排队多个 job，但同一时间只运行一个真实 Qwen 转录。
- job 进度只承诺服务端阶段和 chunk 级真实进度，不要伪造单个 Qwen chunk 内部百分比。
- 错误必须使用 PRD 错误信封：

```json
{
  "error": {
    "code": "model_not_found",
    "message": "unknown model: xxx",
    "details": {}
  }
}
```

## 网络与安全

- 唯一面向局域网的 API 端口是 `18080`。
- Mac 客户端请求 `192.168.31.137` 时必须绕过本机代理，例如使用 `curl --noproxy '*'`。
- 支持可选 Bearer Token 鉴权，但不要假设服务会暴露到公网。
- 不要添加公网隧道、端口映射或互联网暴露。
- 上传音频应写入临时位置，并在推理结束后清理。
- 日志默认不要保存完整音频内容。

## 测试要求

至少覆盖：

- 健康检查。
- 模型列表和单模型状态。
- 模型加载和卸载行为。
- 卸载等待活跃请求完成。
- `unloading_scheduled` 状态下拒绝新请求。
- 转录接口参数校验。
- 异步 job 创建、状态轮询、队列位置、chunk 进度、完成结果、失败错误和取消语义。
- 未声明能力的错误处理，例如 timestamps、forced alignment 或 streaming 返回 `capability_not_supported`。
- Qwen 两个尺寸和 `/v1/models` 中声明的所有后端的转录路径。

先用 mock 适配器覆盖生命周期和 API 行为，再在 WSL Arch Linux 内接入真实 Qwen 模型。

## Git 规范

- 生成缓存、虚拟环境、模型缓存、上传文件和本地运行数据不要进 Git。
- 文档和提示词保持在稳定路径，方便代理交接。
- 提交要聚焦，提交信息要清楚。
- 不要提交机器专用 secret、token、下载模型或包含隐私内容的音频样本。
