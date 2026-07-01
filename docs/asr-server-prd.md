# WSL ASR Server PRD

更新时间：2026-06-30

## 1. 背景

当前 Windows 主机已通过 WSL2 Arch Linux 暴露局域网服务，Mac mini 已验证可以访问 WSL mirrored networking 下的局域网服务。目标是在 WSL 内常驻一个 ASR 服务，让 Mac mini 上的本地项目可以通过 HTTP API 调用 Windows/WSL 内的 GPU 模型完成音频转录。

初版支持模型只包含 Qwen3-ASR 两个尺寸：

- QwenLM/Qwen3-ASR：https://github.com/QwenLM/Qwen3-ASR

MiMo-V2.5-ASR 不进入初版交付范围，保留为后续扩展候选。

硬件前提：

- Windows 主机：RTX 5070 Ti 16GB VRAM
- WSL：Arch Linux，已启用 mirrored networking
- WSL 资源：32GB RAM，12 CPUs，16GB swap
- 局域网入口建议：`http://192.168.31.137:18080`

## 2. 目标

构建一个常驻 WSL 后台的统一 ASR 网关，让 Mac mini 可以通过稳定、可探测、可管理的 API 调用 WSL Arch Linux 上的 Qwen3-ASR GPU 转录能力。

必须支持：

- 查询服务健康状态。
- 查询 ASR 服务端支持哪些模型、每个模型的能力与加载状态。
- 请求加载指定模型。
- 请求卸载指定模型或全部模型。
- 卸载模型时保证当前正在执行的请求完成后再卸载，并拒绝新的同模型请求。
- 上传音频并获取转写结果。
- 创建异步转录任务，并通过轮询获取阶段状态和 chunk 级真实进度。
- 跑通 `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b` 的端到端转录流程。
- 对每个 Qwen3-ASR 模型跑通所有在 `/v1/models` 中声明支持的推理后端；第一版只声明 `transformers` 后端，`vllm` 后端延后到后续版本。
- `/v1/models` 只声明已经实现并通过验收的能力；时间戳、强制对齐、流式转写等高级能力不作为初版发布验收前提。

暂不做：

- 公网访问。
- 多用户计费。
- 分布式多机调度。
- Web 管理后台。
- Windows Docker 部署。
- MiMo-V2.5-ASR 适配器与验收。

## 3. 用户与场景

主要用户是 Mac mini 上的本地开发项目。

典型场景：

1. Mac 项目启动时调用 `GET /v1/models`，发现服务端可用模型与能力。
2. Mac 项目上传音频到 `POST /v1/audio/transcriptions`，指定 `model=qwen3-asr-1.7b` 或 `model=qwen3-asr-0.6b`。
3. 长音频或批处理时，Mac 项目创建异步任务，再轮询任务状态。
4. 长时间不用某个模型时，Mac 项目或运维脚本调用卸载接口释放显存。
5. 如果服务端正在处理请求，卸载动作进入排队状态，当前请求结束后再释放模型。

## 4. 总体架构

采用单一 HTTP 网关加模型适配器结构。

```mermaid
flowchart LR
  Mac["Mac mini 项目"] --> Gateway["ASR Gateway :18080"]
  Gateway --> Registry["Model Registry"]
  Gateway --> Manager["Model Lifecycle Manager"]
  Gateway --> Qwen["Qwen3-ASR Adapter"]
  Qwen --> GPU["NVIDIA GPU in WSL"]
```

建议进程：

- `asr-gateway`：对 Mac 暴露 HTTP API，监听 `0.0.0.0:18080`。
- `qwen-adapter`：可先内嵌在 gateway 进程内，后续如需 vLLM serving 再拆成独立 worker。

端口规划：

- `18080`：唯一对 Mac mini 暴露的正式 ASR API 入口。服务进程运行在 WSL Arch Linux 内，监听 `0.0.0.0:18080`；Windows 侧只需要为这个端口添加专用网络入站防火墙规则。
- `8001`：可选的 WSL 内部 Qwen3-ASR worker 端口。只有当后续把 Qwen 适配器拆成独立 worker 进程时才使用；默认第一版不需要开放。若启用，必须只绑定 WSL 内部 loopback 或 Unix socket，不为 Windows 防火墙添加入站规则，不对 Mac mini 暴露。

历史测试端口 `8765` 不属于正式设计，不应写入部署、启动、自启或防火墙配置。正式验收只使用 `18080`。

## 5. 模型能力矩阵

| 模型 | 模型 ID | 首选用途 | 初版验收能力 | 限制 |
| --- | --- | --- | --- | --- |
| Qwen3-ASR 0.6B | `qwen3-asr-0.6b` | 轻量转写、较低显存占用 | 离线转写、语言提示、`transformers` 后端转写 | 质量和复杂音频能力弱于 1.7B；高级能力后续按实测结果逐项打开 |
| Qwen3-ASR 1.7B | `qwen3-asr-1.7b` | 默认主力模型 | 离线转写、语言提示、`transformers` 后端转写 | 流式、时间戳、强制对齐、vLLM 等高级能力不阻塞初版发布 |

## 6. API 设计

### 6.1 健康检查

`GET /health`

返回：

```json
{
  "status": "ok",
  "version": "0.1.0",
  "host": "archlinux-wsl",
  "gpu": {
    "available": true,
    "name": "NVIDIA GeForce RTX 5070 Ti",
    "vram_total_mb": 16303
  }
}
```

### 6.2 查询模型

`GET /v1/models`

返回：

```json
{
  "models": [
    {
      "id": "qwen3-asr-1.7b",
      "provider": "QwenLM",
      "status": "unloaded",
      "default": true,
      "capabilities": {
        "transcription": true,
        "streaming": false,
        "timestamps": [],
        "forced_alignment": false,
        "languages": ["auto", "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro"],
        "chinese_dialects": ["Anhui", "Dongbei", "Fujian", "Gansu", "Guizhou", "Hebei", "Henan", "Hubei", "Hunan", "Jiangxi", "Ningxia", "Shandong", "Shaanxi", "Shanxi", "Sichuan", "Tianjin", "Yunnan", "Zhejiang", "Cantonese-Hong-Kong-accent", "Cantonese-Guangdong-accent", "Wu", "Minnan"],
        "backends": ["transformers"]
      }
    },
    {
      "id": "qwen3-asr-0.6b",
      "provider": "QwenLM",
      "status": "unloaded",
      "default": false,
      "capabilities": {
        "transcription": true,
        "streaming": false,
        "timestamps": [],
        "forced_alignment": false,
        "languages": ["auto", "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro"],
        "chinese_dialects": ["Anhui", "Dongbei", "Fujian", "Gansu", "Guizhou", "Hebei", "Henan", "Hubei", "Hunan", "Jiangxi", "Ningxia", "Shandong", "Shaanxi", "Shanxi", "Sichuan", "Tianjin", "Yunnan", "Zhejiang", "Cantonese-Hong-Kong-accent", "Cantonese-Guangdong-accent", "Wu", "Minnan"],
        "backends": ["transformers"]
      }
    }
  ]
}
```

### 6.3 查询单个模型状态

`GET /v1/models/{model_id}/status`

状态枚举：

- `unloaded`
- `loading`
- `loaded`
- `unloading_scheduled`
- `unloading`
- `error`

返回：

```json
{
  "id": "qwen3-asr-1.7b",
  "status": "loaded",
  "active_requests": 1,
  "rejecting_new_requests": false,
  "backend": "transformers",
  "loaded_at": "2026-06-26T21:30:00+08:00",
  "last_used_at": "2026-06-26T21:34:12+08:00",
  "vram_allocated_mb": 7420
}
```

### 6.4 加载模型

`POST /v1/models/{model_id}/load`

请求：

```json
{
  "backend": "auto",
  "device": "cuda",
  "dtype": "auto"
}
```

返回：

```json
{
  "id": "qwen3-asr-1.7b",
  "status": "loading",
  "message": "model loading started"
}
```

### 6.5 卸载单个模型

`DELETE /v1/models/{model_id}`

请求：

```json
{
  "mode": "after_current_requests",
  "reject_new_requests": true,
  "cuda_empty_cache": true
}
```

语义：

- 如果模型没有正在运行的请求，立即卸载。
- 如果模型有正在运行的请求，立刻把状态改为 `unloading_scheduled`。
- 状态为 `unloading_scheduled` 后，新的同模型转写请求返回 `409 model_unloading_scheduled`。
- 已经开始的请求正常完成。
- 最后一个活跃请求结束后执行卸载并清理 CUDA cache。

返回：

```json
{
  "id": "qwen3-asr-1.7b",
  "status": "unloading_scheduled",
  "active_requests": 2,
  "rejecting_new_requests": true
}
```

### 6.6 卸载全部模型

`DELETE /v1/models`

请求：

```json
{
  "mode": "after_current_requests",
  "reject_new_requests": true,
  "cuda_empty_cache": true
}
```

返回：

```json
{
  "status": "accepted",
  "models": [
    {
      "id": "qwen3-asr-1.7b",
      "status": "unloading_scheduled",
      "active_requests": 1
    },
    {
      "id": "qwen3-asr-0.6b",
      "status": "unloaded",
      "active_requests": 0
    }
  ]
}
```

### 6.7 同步转写

`POST /v1/audio/transcriptions`

Content-Type: `multipart/form-data`

字段：

- `file`：音频文件，必填。
- `model`：模型 ID，默认 `qwen3-asr-1.7b`。
- `language`：`auto`、`zh`、`en` 等，默认 `auto`。
- `response_format`：`json`、`text`、`verbose_json`，默认 `json`。
- `timestamps`：`none`、`word`、`char`，默认 `none`。
- `backend`：`auto`、`transformers`、`vllm`，默认 `auto`。
- `temperature`：可选。
- `max_new_tokens`：可选；用于控制 Qwen 生成长度，服务端必须设置上限，避免长音频请求失控。
- `context`：可选；本次转录的领域提示、术语、人名、产品名、项目名等，服务端应限制长度并传给 Qwen adapter。
- `hotwords`：可选；热词列表，可用逗号分隔字符串或 JSON 数组表达，服务端可合并到 `context`。

返回：

```json
{
  "id": "tr_01JZ0000000000000000000000",
  "model": "qwen3-asr-1.7b",
  "backend": "transformers",
  "language": "zh",
  "text": "这是转写结果。",
  "duration": 12.35,
  "timestamps": [],
  "segments": [],
  "usage": {
    "audio_seconds": 12.35
  },
  "warnings": []
}
```

### 6.8 强制对齐

`POST /v1/audio/alignments`

预留接口，后续仅 Qwen3-ASR 模型支持。只有当对应模型和后端已通过能力验收时，`forced_alignment` 才能在 `/v1/models` 中标记为 `true`。

Content-Type: `multipart/form-data`

字段：

- `file`：音频文件，必填。
- `text`：需要对齐的文本，必填。
- `model`：默认 `qwen3-asr-1.7b`。
- `language`：默认 `auto`。
- `granularity`：`word` 或 `char`。

返回：

```json
{
  "id": "al_01JZ0000000000000000000000",
  "model": "qwen3-asr-1.7b",
  "language": "zh",
  "granularity": "char",
  "items": [
    {
      "text": "这",
      "start": 0.0,
      "end": 0.18
    }
  ]
}
```

如果请求不支持强制对齐的模型或后端：

```json
{
  "error": {
    "code": "capability_not_supported",
    "message": "requested model/backend does not support forced alignment in this server"
  }
}
```

### 6.9 异步任务与进度查询

- `POST /v1/audio/transcription-jobs`
- `GET /v1/jobs/{job_id}`
- `DELETE /v1/jobs/{job_id}`

适合长音频、批处理、前端需要进度展示、或 Mac 项目不希望 HTTP 长连接阻塞的场景。

第一版采用内存 job manager 和单 worker FIFO 队列：

- 服务端可以接收多个 job，但同一时间只运行一个转录 job。
- 后提交的 job 保持 `queued`，通过 `queue_position` 暴露排队位置。
- 不引入 Redis、Celery、数据库或多 worker 推理。
- 进程重启后内存 job 可以丢失；客户端应将 `job_not_found` 视为需要重新提交。
- job 结果默认保留 1 小时，之后可过期清理。

进度边界：

- 服务端必须暴露真实阶段：`queued`、`preprocessing`、`splitting`、`loading_model`、`transcribing`、`merging`、`completed`、`failed`。
- 切分完成后，必须暴露真实 chunk 级进度：`total_chunks`、`completed_chunks`、`current_chunk`、`percent`。
- Qwen adapter 当前拿不到单个 chunk 内部推理百分比，不能伪造模型内部 token/帧级进度。

`POST /v1/audio/transcription-jobs`

Content-Type: `multipart/form-data`

字段与同步转写接口保持一致，第一版只要求支持单个 `file`。服务端收到请求后应快速返回 `202 Accepted`，不在该 HTTP 请求内执行真实转录。

返回：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "queued",
  "model": "qwen3-asr-1.7b",
  "backend": "transformers",
  "queue_position": 1,
  "created_at": "2026-06-29T12:00:00Z",
  "status_url": "/v1/jobs/job_01JZ0000000000000000000000"
}
```

`GET /v1/jobs/{job_id}`

队列中返回：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "queued",
  "model": "qwen3-asr-1.7b",
  "backend": "transformers",
  "queue_position": 2,
  "progress": {
    "phase": "queued",
    "percent": 0.0,
    "message": "waiting for previous transcription jobs"
  },
  "created_at": "2026-06-29T12:00:00Z",
  "started_at": null,
  "completed_at": null,
  "elapsed_seconds": 0.0
}
```

转录中返回：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "transcribing",
  "model": "qwen3-asr-1.7b",
  "backend": "transformers",
  "queue_position": 0,
  "progress": {
    "phase": "transcribing",
    "percent": 37.5,
    "total_chunks": 32,
    "completed_chunks": 12,
    "current_chunk": 13,
    "current_chunk_start": 1412.4,
    "current_chunk_end": 1530.1,
    "message": "transcribing chunk 13 of 32"
  },
  "split": {
    "strategy": "silero",
    "requested_strategy": "auto",
    "vad_backend": "silero",
    "chunk_count": 32,
    "soft_chunk_seconds": 120,
    "hard_chunk_seconds": 300,
    "overlap_seconds": 2
  },
  "created_at": "2026-06-29T12:00:00Z",
  "started_at": "2026-06-29T12:00:03Z",
  "completed_at": null,
  "elapsed_seconds": 74.3
}
```

完成后返回：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "completed",
  "progress": {
    "phase": "completed",
    "percent": 100.0,
    "total_chunks": 32,
    "completed_chunks": 32
  },
  "result": {
    "id": "tr_01JZ0000000000000000000000",
    "model": "qwen3-asr-1.7b",
    "backend": "transformers",
    "language": "zh",
    "text": "完整合并后的转写文本。",
    "duration": 3630.05,
    "split": {},
    "chunks": [],
    "usage": {
      "audio_seconds": 3630.05
    },
    "timings": {},
    "warnings": []
  },
  "error": null,
  "created_at": "2026-06-29T12:00:00Z",
  "started_at": "2026-06-29T12:00:03Z",
  "completed_at": "2026-06-29T12:04:12Z",
  "elapsed_seconds": 249.0
}
```

失败后返回：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "failed",
  "progress": {
    "phase": "failed",
    "percent": 37.5,
    "total_chunks": 32,
    "completed_chunks": 12
  },
  "result": null,
  "error": {
    "code": "gpu_unavailable",
    "message": "CUDA out of memory during Qwen ASR",
    "details": {
      "phase": "transcribing",
      "chunk_index": 12
    }
  }
}
```

`DELETE /v1/jobs/{job_id}`

取消语义：

- `queued` job 可以直接取消为 `cancelled`。
- `preprocessing`、`splitting`、`merging` 阶段尽量在阶段边界取消。
- `transcribing` 阶段不要强杀 Qwen 推理，不要卸载模型；设置 `cancel_requested`，等当前 chunk 完成后停止后续 chunk。
- 已结束 job 的 DELETE 返回当前状态，不应报 500。

### 6.10 流式转写

第一版可以先不实现，但接口预留如下：

- `WebSocket /v1/audio/transcriptions/stream`

Qwen3-ASR 官方支持流式推理，但当前边界需要在 API 中明确：

- Qwen3-ASR streaming 当前依赖 vLLM backend；transformers backend 不作为流式实现路径。
- 流式请求不支持 batch inference。
- 流式请求不返回 timestamps；需要 timestamps 时应使用非流式转写或强制对齐接口。
- 流式转写面向“实时输入”，例如麦克风、通话、边录边转；不替代长音频文件上传后的服务端切分。
- 如果流式转写未在初版实现或未通过验收，`/v1/models` 中对应模型的 `streaming` 必须为 `false`。

建议流式 API：

```text
WebSocket /v1/audio/transcriptions/stream?model=qwen3-asr-1.7b&language=auto
```

客户端发送 16kHz PCM 音频帧，服务端返回增量文本事件：

```json
{
  "type": "partial",
  "text": "正在识别中的临时文本",
  "is_final": false
}
```

```json
{
  "type": "final",
  "text": "稳定后的最终片段文本",
  "is_final": true
}
```

如果请求端上传的是已经存在的完整音频文件，即使使用 Qwen3-ASR，默认仍走 `POST /v1/audio/transcriptions`，由 ASR server 执行切分、批处理、合并和可选时间戳。

### 6.11 服务端音频切分与合并

请求端可以一次上传一个非无限长音频，或一次上传一个音频数组。切分、批处理、模型窗口适配、结果合并由 ASR server 负责；请求端不需要理解不同模型的推荐输入长度。

设计原则：

- 请求端只表达业务意图，例如模型、语言、是否需要时间戳、是否允许长音频异步处理。
- 服务端根据模型能力、音频时长、显存状态、是否需要时间戳，决定切分窗口。
- 对外返回原始音频级别的结果，同时保留 chunk 级别元数据，方便调试。
- 对长音频优先使用异步任务接口；同步接口只适合短音频和中等长度音频。

输入字段扩展：

- `files`：音频数组，可选；与单个 `file` 二选一。
- `split_strategy`：`auto`、`none`、`fixed`、`silero`、`energy`、`vad`，默认 `auto`；`vad` 是兼容别名，当前默认语义指向 Silero VAD，Silero 不可用时 fallback 到 energy VAD。
- `max_chunk_seconds`：可选；用户给上限时不得超过服务端模型上限。
- `overlap_seconds`：可选；默认由服务端决定。
- `preserve_segments`：是否返回 chunk 级别结果，默认 `false`。

服务端默认切分策略：

1. 使用 FFmpeg 解码并规范化音频元数据，生成服务端临时 WAV/PCM。
2. 统计总时长、采样率、声道数、文件大小。
3. 如果音频短于当前模型的 `soft_chunk_seconds`，默认不切分。
4. 如果音频较长，优先使用 Silero VAD 在人声边界切分；Silero 不可用时 fallback 到 energy VAD，再失败才固定窗口切分。
5. 每个 chunk 保留少量 overlap，降低切断词、切断句的概率。
6. 合并时按原始时间线排序，去掉 overlap 中重复的文本或时间戳。
7. 返回 `chunks` 调试信息，包括每段起止时间、模型、耗时、错误与警告。

默认模型窗口：

| 模型 | 默认 soft chunk | 默认 hard chunk | overlap | 说明 |
| --- | --- | --- | --- | --- |
| `qwen3-asr-1.7b` | 120 秒 | 300 秒 | 2 秒 | WSL RTX 5070 Ti 实测后默认按 2 分钟软切分，配合默认 `max_new_tokens=512` 降低显存峰值。请求时间戳或强制对齐时，单段不得超过 300 秒。 |
| `qwen3-asr-0.6b` | 120 秒 | 300 秒 | 2 秒 | 与 1.7B 保持一致；后续可根据吞吐测试放宽。 |

同步与异步阈值：

- 单个请求总音频时长小于等于 10 分钟时，允许走同步 `POST /v1/audio/transcriptions`。
- 单个请求总音频时长超过 10 分钟时，同步接口应返回 `202 accepted` 并创建异步 job，响应包含 `job_id` 和 `status_url`；如果暂未实现自动创建 job，必须返回明确错误并提示客户端使用 `POST /v1/audio/transcription-jobs`。
- 单文件默认最大 6 小时。
- 单请求数组默认最多 50 个文件，总时长默认最多 6 小时。
- 超过服务端限制返回 `413 audio_too_large` 或 `422 duration_limit_exceeded`。

返回示例：

```json
{
  "id": "tr_01JZ0000000000000000000000",
  "model": "qwen3-asr-1.7b",
  "text": "完整合并后的转写文本。",
  "duration": 742.3,
  "split": {
    "strategy": "silero",
    "requested_strategy": "auto",
    "vad_backend": "silero",
    "chunk_count": 5,
    "soft_chunk_seconds": 120,
    "hard_chunk_seconds": 300,
    "overlap_seconds": 2
  },
  "chunks": [
    {
      "index": 0,
      "start": 0.0,
      "end": 178.4,
      "text": "第一段文本。",
      "warnings": []
    }
  ],
  "warnings": []
}
```

## 7. 错误码

统一错误格式：

```json
{
  "error": {
    "code": "model_not_found",
    "message": "unknown model: xxx",
    "details": {}
  }
}
```

错误码：

| HTTP | code | 场景 |
| --- | --- | --- |
| 400 | `bad_request` | 参数缺失、非法枚举值 |
| 404 | `model_not_found` | 模型 ID 不存在 |
| 404 | `job_not_found` | job ID 不存在、已过期或服务重启后丢失 |
| 409 | `model_loading` | 模型正在加载，暂不能处理请求 |
| 409 | `model_unloading_scheduled` | 模型已安排卸载，拒绝新请求 |
| 409 | `job_cancel_requested` | job 已请求取消，不能重复启动或修改 |
| 413 | `audio_too_large` | 音频超出服务限制 |
| 415 | `unsupported_audio_format` | 音频格式不支持 |
| 422 | `capability_not_supported` | 请求了模型不支持的能力 |
| 429 | `job_queue_full` | 异步转录队列达到服务端上限 |
| 500 | `inference_failed` | 模型推理失败 |
| 503 | `gpu_unavailable` | GPU 不可用或显存不足 |

## 8. 运行与部署

服务部署位置：

- WSL 内：`/home/fragt/services/asr-server`

建议技术栈：

- Python 3.12
- FastAPI
- Uvicorn
- uv 管理 Python 环境
- systemd user service 或 Windows 启动任务触发 `wsl -d archlinux`

启动命令建议：

```bash
cd /home/fragt/services/asr-server
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

Windows 防火墙：

- 需要允许专用网络入站 TCP `18080`。
- 不需要为 `8001` 添加 Windows 防火墙入站规则；它只是未来可选的 WSL 内部 Qwen worker 通信端口。
- 不需要保留或开放 `8765`；它只是此前连通性测试端口。
- Mac 请求局域网服务时需要绕过本机代理，例如 `curl --noproxy '*' http://192.168.31.137:18080/health`。

## 9. 后台常驻要求

服务应满足：

- Windows 开机后可自动启动 WSL 服务。
- 服务崩溃后可自动重启。
- 模型不必开机即加载，可 lazy load。
- 首次请求某个模型时可自动加载，但返回时间可能较长。
- 转录完成后 3 分钟内没有新的同模型转录请求时，自动卸载模型并清理 CUDA cache；新请求到来时取消上一轮空闲卸载计时，直到该请求结束后重新计时。

建议配置：

```yaml
server:
  host: 0.0.0.0
  port: 18080
  public_base_url: http://192.168.31.137:18080

models:
  default: qwen3-asr-1.7b
  auto_load_on_request: true
  idle_unload_seconds: 180
  max_loaded_models: 1

limits:
  max_upload_mb: 512
  max_audio_seconds_sync: 600
  request_timeout_seconds: 3600
  job_result_ttl_seconds: 3600
  max_queued_jobs: 20
```

## 10. 安全要求

由于服务只给局域网使用，第一版可使用轻量认证：

- 支持可选 `Authorization: Bearer <token>`。
- 默认只监听局域网地址或 `0.0.0.0` 配合 Windows 专用网络防火墙。
- 不允许公网端口映射。
- 上传音频写入临时目录，推理结束后清理。
- 同步转录、异步 job、音频解码、切分和模型适配器产生的临时文件必须在请求完成、失败或取消后清理；job 结果 TTL 只允许保留内存中的结果和状态，不保留上传音频或中间音频文件。
- 日志不默认保存完整音频内容。

## 11. 验收标准

连通性：

- Mac mini 执行 `curl --noproxy '*' http://192.168.31.137:18080/health` 返回 `status=ok`。
- Mac mini 执行 `curl --noproxy '*' http://192.168.31.137:18080/v1/models` 能看到 `qwen3-asr-0.6b` 与 `qwen3-asr-1.7b`。
- `/v1/models` 不展示未进入初版交付范围的 MiMo 模型，也不声明未验收通过的高级能力。

转写：

- Mac mini 通过局域网向 `POST /v1/audio/transcriptions` 上传一段 10 秒中文音频，`qwen3-asr-0.6b` + `transformers` 返回非空 `text`。
- Mac mini 通过局域网向 `POST /v1/audio/transcriptions` 上传一段 10 秒中文音频，`qwen3-asr-1.7b` + `transformers` 返回非空 `text`。
- 默认模型 `qwen3-asr-1.7b` 在不显式传 `backend` 时可以用 `backend=auto` 完成转录。
- 前端 Mac 客户端只依赖 `GET /v1/models` 的模型与后端发现结果，不硬编码服务端未声明的能力。

异步 job：

- Mac mini 能创建 `POST /v1/audio/transcription-jobs`，拿到 `202`、`job_id` 和 `status_url`。
- `GET /v1/jobs/{job_id}` 能看到 `queued`、运行中状态和最终 `completed`。
- 长音频转录中，`progress.total_chunks`、`completed_chunks`、`current_chunk` 会随真实 chunk 完成而变化。
- 多个 job 同时提交时，服务端只运行一个，其余 job 显示 `queued` 和正确 `queue_position`。
- job 完成后 `result.text` 非空，结构与同步转写结果兼容。
- job 失败时返回统一 error 对象，包含错误 code、message、details 和失败阶段。

模型管理：

- 模型处于 `loaded` 时，`DELETE /v1/models/{model_id}` 能将其卸载。
- 模型有活跃请求时，卸载请求进入 `unloading_scheduled`。
- `unloading_scheduled` 状态下，新请求返回 409。
- 活跃请求完成后，模型进入 `unloaded`。

能力探测：

- 请求未声明的高级能力，例如 timestamps、forced alignment 或 streaming，返回 `capability_not_supported`，或在能力未声明时由客户端避免发起该请求。
- 任意出现在 `/v1/models` capabilities 中的能力都必须有对应的 Mac 到 WSL 端到端验收记录。

## 12. 后续路线

当前 WSL 服务端已经实现异步转录 job、FIFO 串行队列、可轮询状态和 chunk 级真实进度。后续阶段不做 vLLM、WebSocket streaming、MiMo、ForcedAligner、数据库队列、多 worker 并发推理或 `*-hf` 路径，除非 PRD 明确扩展范围。

优先级 P0：

- FastAPI 网关。
- 模型注册表。
- Qwen3-ASR 0.6B 与 1.7B 适配器最小可用转写。
- Qwen3-ASR 两个尺寸在 `transformers` 后端上完成端到端转录验收。
- 模型加载/卸载状态机。
- Mac 局域网访问验收。

优先级 P1：

- 转录耗时记录，包括总耗时、加载耗时、推理耗时和后处理耗时。
- 长音频切分与合并，先支持固定切分和 chunk 元数据。
- 异步转录 job 和 chunk 级进度查询。
- 异步任务。
- 上传文件大小、时长、格式限制。
- systemd user service 或 Windows 开机启动。
- Qwen 时间戳与强制对齐能力，按后端实测结果逐项声明。

优先级 P2：

- Qwen vLLM 独立 worker。
- WebSocket 流式转写。
- MiMo-V2.5-ASR 适配器调研与转写验收。
- 简单 Web 管理页。
- 多模型队列与优先级。

## 13. 参考资料

- Qwen3-ASR GitHub：https://github.com/QwenLM/Qwen3-ASR
- MiMo-V2.5-ASR GitHub：https://github.com/XiaomiMiMo/MiMo-V2.5-ASR
- WSL mirrored networking 已在本机验证，正式 ASR 服务入口规划为 `192.168.31.137:18080`
