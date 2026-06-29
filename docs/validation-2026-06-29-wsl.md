# WSL ASR 验收记录 2026-06-29

## 环境

- 日期：2026-06-29
- WSL 发行版：WSL2 Linux `6.6.87.2-microsoft-standard-WSL2`
- 项目路径：`/home/fragt/project/asr_Server`
- Python 管理：uv
- torch 版本：`2.11.0+cu128`
- torch CUDA 版本：`12.8`
- GPU 名称：`NVIDIA GeForce RTX 5070 Ti`
- GPU capability：`(12, 0)`
- `qwen-asr`：可 import，`qwen_asr.Qwen3ASRModel`
- `silero-vad`：`6.2.1`

## 功能验收

- Silero VAD：`split_strategy=silero` 可用，长音频无 fallback。
- energy fallback：未安装 Silero 时已验证 fallback 到 `energy` 并返回 warning。
- context / hotwords：mock HTTP 验证 adapter 收到 context hash，不污染转录文本。
- `max_new_tokens`：真实 Qwen 验证后改为加载期参数，传入 `Qwen3ASRModel.from_pretrained(...)`。
- Qwen batch：0.6B 短音频固定切分后，真实 batch size 2 路径跑通。
- overlap 去重：新增 fuzzy tail-prefix 去重，保留 `raw_text`，更新 `chunks[].text` 和顶层 `text`。

## 1.7B 长音频对比

音频文件：`test-fixtures/audio/test_long.mp3`

### `max_chunk_seconds=180` + `max_new_tokens=1024`

- 模型：`qwen3-asr-1.7b`
- 后端：`transformers`
- VAD backend：`silero`
- chunk 数：21
- 音频时长：`3630.053875s`
- wall time：`259.548s`
- `total_ms`：`234994.19`
- `load_ms`：`14004.26`
- `decode_ms`：`4199.15`
- `inference_ms`：`216753.51`
- 平均 chunk 推理：`10321.60ms`
- 最大 chunk 推理：`13618.20ms`
- 文本长度：`9719`

### `max_chunk_seconds=180` + `max_new_tokens=768`

- 模型：`qwen3-asr-1.7b`
- 后端：`transformers`
- VAD backend：`silero`
- chunk 数：21
- 音频时长：`3630.053875s`
- wall time：`270.007s`
- `total_ms`：`240836.86`
- `load_ms`：`14301.18`
- `decode_ms`：`4200.97`
- `inference_ms`：`222295.32`
- 平均 chunk 推理：`10585.49ms`
- 最大 chunk 推理：`13070.84ms`
- 运行中显存观察：约 `15861-15882 MiB / 16303 MiB`
- 文本长度：`9662`

### `max_chunk_seconds=120` + `max_new_tokens=512`

- 模型：`qwen3-asr-1.7b`
- 后端：`transformers`
- VAD backend：`silero`
- chunk 数：32
- 音频时长：`3630.053875s`
- wall time：`231.232s`
- `total_ms`：`202800.38`
- `load_ms`：`11905.46`
- `decode_ms`：`4287.87`
- `inference_ms`：`186567.64`
- 平均 chunk 时长：`117.019s`
- 最大 chunk 时长：`119.948s`
- 平均 chunk 推理：`5830.24ms`
- 最大 chunk 推理：`8262.95ms`
- 运行中显存观察：约 `11351-13440 MiB / 16303 MiB`
- 服务停止后显存：约 `1347 MiB / 16303 MiB`
- 文本长度：`9827`
- 输出文件：`/tmp/asr_long_1_7b_512_120s_response.json`

## 默认参数结论

当前 WSL 实测后采用以下默认值：

```text
max_chunk_seconds=120
max_new_tokens=512
ASR_QWEN_BATCH_SIZE=1
```

理由：

- 比 `180s + 768` 峰值显存低约 2.4GB。
- 总 wall time 从约 `270s` 降到约 `231s`。
- chunk 最大推理耗时从约 `13070ms` 降到约 `8263ms`。
- 文本长度未下降，未观察到明显截断。

## 异步 job 与进度验收

### 对抗式发现与修复

- 初次真实长音频 job 验收时发现：`QwenAsrAdapter` 是 `async def`，但内部执行同步 `from_pretrained(...)` 和 `model.transcribe(...)`，会阻塞 Uvicorn event loop。
- 影响：长音频 job 运行期间 `GET /v1/jobs/{job_id}` 和 `/health` 不能及时响应，违背“可轮询进度”的核心目标。
- 修复：生命周期管理器把 adapter `load` / `unload` / `transcribe` / `transcribe_batch` 调用放入 `asyncio.to_thread(...)`；job 的音频 decode、split、merge 也放入线程执行。
- 修复后验证：长音频 job 转录期间 `/health` 持续响应，轮询可见 `preprocessing`、`splitting`、`transcribing`、`merging`、`completed`。

### `qwen3-asr-1.7b` 长音频 job

- 接口：`POST /v1/audio/transcription-jobs`
- 模型：`qwen3-asr-1.7b`
- 后端：`transformers`
- 音频文件：`test-fixtures/audio/test_long.mp3`
- 音频时长：`3630.053875s`
- split：`silero`
- chunk 数：`32`
- 状态流转：`queued -> preprocessing -> splitting -> transcribing -> merging -> completed`
- 轮询观察：`completed_chunks` 从 `0` 增长到 `32`，`current_chunk` 从 `1` 增长到 `32`
- 长 job 期间 `/health` 成功响应次数：`112`
- wall time：约 `222.7s`
- `total_ms`：`193340.19`
- `load_ms`：`0.0`（1.7B 已由 HTTP smoke 预加载）
- `decode_ms`：`4244.08`
- `inference_ms`：`189046.05`
- 文本前 200 字：`别紧张，是我。嗯，是熟悉的味道呢。看来命运再一次将你交到了我的手上。刚刚听到那个音乐，是有应急反应吗？还是说唤起了你的一些快乐的回忆？是呢，是呢，是我将你带到了这里，可是你也收获了很多美好的经历，不是吗？一脸幸福的回味表情呢。而且上次在我这里的时候，我记得你也很享受呢。还是说主人对你太客气了，妾奴？我说过的吧，主人一定会找过你的，我的小奴隶。走吧，我们换个地方好好玩。好了，就是这里了。这里是舞台的`

### `qwen3-asr-0.6b` 短音频 job

- 接口：`POST /v1/audio/transcription-jobs`
- 模型：`qwen3-asr-0.6b`
- 后端：`transformers`
- 音频文件：`test-fixtures/audio/test_short.wav`
- 状态流转：`queued -> preprocessing -> loading_model -> transcribing -> completed`
- chunk 数：`1`
- `total_ms`：`9481.25`
- `load_ms`：`8049.89`
- `decode_ms`：`44.13`
- `inference_ms`：`1386.46`
- 文本前 200 字：`你好，你好，你好。这个是测试音频，主要用于测试千问 ASR 模型 1.7B 的实际转录能力。`

### job 行为测试覆盖

- 创建 job 返回 `202`、`queued`、`status_url`
- `GET /v1/jobs/{job_id}` 返回 queued/running/completed
- 单 worker FIFO 队列与 `queue_position`
- chunk 级进度：`total_chunks`、`completed_chunks`、`current_chunk`
- completed result 与同步接口结构兼容
- adapter 错误进入 `failed`，错误对象为 `code/message/details`
- queued job 取消为 `cancelled`
- running job 设置 `cancel_requested`，chunk 边界后进入 `cancelled`
- 未知 job 返回 `404 job_not_found`
- 超过同步阈值时同步接口返回 `202` job

## 验证命令

```bash
uv run pytest -q
uv run mypy asr_server tests scripts
ASR_BASE_URL=http://127.0.0.1:18080 uv run pytest tests/test_http_smoke.py -q
```

结果：

```text
58 passed, 2 skipped
Success: no issues found in 33 source files
HTTP smoke: 2 passed
```
