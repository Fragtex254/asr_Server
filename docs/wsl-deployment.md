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

真实 Qwen adapter 启动时使用：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
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
