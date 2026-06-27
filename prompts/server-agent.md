# 服务端 Agent 提示词

你是在 Windows 主机的 WSL Arch Linux 内工作的开发 agent。你的任务是实现一个常驻后台的 ASR 服务，让局域网内 Mac mini 可以访问 Windows/WSL 里的 GPU ASR 模型。

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

技术约束：

- 使用 Python 3.12。
- 使用 uv 管理环境。
- 使用 FastAPI + Uvicorn 实现 HTTP API。
- 服务监听 `0.0.0.0:18080`。
- 不要把项目放在 `/mnt/c` 下，放在 WSL 原生文件系统 `/home/fragt/services/asr-server`。
- 第一版优先做稳定的离线转写，不要一开始就实现 Web UI。
- 模型包很重，安装前先检查磁盘空间、CUDA、`nvidia-smi`。

首批模型：

- `qwen3-asr-1.7b`
- `qwen3-asr-0.6b`
- `mimo-v2.5-asr`

必须实现的 API：

```text
GET /health
GET /v1/models
GET /v1/models/{model_id}/status
POST /v1/models/{model_id}/load
DELETE /v1/models/{model_id}
DELETE /v1/models
POST /v1/audio/transcriptions
POST /v1/audio/alignments
```

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

- 每个模型维护 active request 计数。
- 每个模型维护 lifecycle lock。
- 收到卸载请求后，如果 active request 为 0，立即卸载。
- 如果 active request 大于 0，设置 `unloading_scheduled` 和 `rejecting_new_requests=true`。
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
    mimo.py
tests/
  test_health.py
  test_models.py
  test_lifecycle.py
  test_transcription_api.py
pyproject.toml
README.md
```

开发顺序：

1. 创建 FastAPI 项目骨架和 `/health`。
2. 实现模型注册表和 `/v1/models`。
3. 实现模型生命周期状态机，并写测试覆盖并发卸载语义。
4. 先用 mock adapter 打通 `POST /v1/audio/transcriptions`。
5. 接入 Qwen3-ASR adapter。
6. 接入 MiMo-V2.5-ASR adapter。
7. 补充 `/v1/audio/alignments`，如果模型不支持则返回 `capability_not_supported`。
8. 增加 systemd user service 或 Windows 启动任务，让服务可后台常驻。
9. 从 Mac mini 验收局域网调用。

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
- 给出一次真实音频的 Qwen 转写验收结果。
- 给出一次真实音频的 MiMo 转写验收结果，或明确说明 MiMo 未能部署的具体阻塞点。

不要做：

- 不要开放公网。
- 不要默认经过代理访问局域网 IP。
- 不要在活跃请求还没结束时强行卸载模型。
- 不要把 Qwen 和 MiMo 的差异暴露给请求端，必须由统一 API 屏蔽。
