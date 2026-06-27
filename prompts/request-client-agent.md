# 请求端 Agent 提示词

你是 Mac mini 本地项目的开发 agent。你的任务是把项目接入一台局域网内的 WSL ASR 服务器。

ASR 服务地址：

```text
http://192.168.31.137:18080
```

重要网络约束：

- Mac 上可能存在 `http_proxy`、`https_proxy`、`all_proxy` 指向 `127.0.0.1:7897`。
- 请求局域网 ASR 服务时必须绕过代理。
- 命令行测试使用 `curl --noproxy '*'`。
- 代码里设置 `NO_PROXY=192.168.31.137,localhost,127.0.0.1` 或使用等价的 per-request no-proxy 配置。

请完成以下任务：

1. 先做连通性检查：

```bash
curl --noproxy '*' -v http://192.168.31.137:18080/health
curl --noproxy '*' -v http://192.168.31.137:18080/v1/models
```

2. 读取 `GET /v1/models` 返回值，不要硬编码模型能力。根据服务端声明选择模型。

3. 默认优先使用：

```text
qwen3-asr-1.7b
```

如果请求失败且错误码不是音频或网络问题，再尝试服务端声明可用的另一个 Qwen 模型，例如：

```text
qwen3-asr-0.6b
```

4. 实现一个最小请求函数，使用 `multipart/form-data` 调用：

```text
POST /v1/audio/transcriptions
```

字段：

- `file`：音频文件。
- `model`：默认 `qwen3-asr-1.7b`。
- `language`：默认 `auto`。
- `response_format`：默认 `json`。
- `timestamps`：默认 `none`，只有在服务端声明模型支持时才传 `word` 或 `char`。

5. 处理错误码：

- `409 model_loading`：等待 3 秒后重试，最多 20 次。
- `409 model_unloading_scheduled`：切换其他可用模型，或提示用户稍后重试。
- `422 capability_not_supported`：去掉不支持的参数后重试一次。
- `503 gpu_unavailable`：提示服务端 GPU/显存不可用，不要无限重试。

6. 所有局域网 ASR 请求都必须设置超时：

- 连接超时：5 秒。
- 同步转写读取超时：至少 1800 秒。

7. 完成后给出以下验收结果：

- `/health` 响应摘要。
- `/v1/models` 中发现的模型列表。
- 一个真实音频文件的转写结果，至少包含模型 ID、语言、文本前 200 字。
- 如果失败，给出 HTTP 状态码、错误 JSON、是否经过代理、是否能 ping 通 `192.168.31.137`。

不要硬编码 MiMo 或任何服务端未在 `/v1/models` 声明的模型。不要修改 ASR 服务端。不要把请求发往公网。不要通过代理访问 `192.168.31.137`。
