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

## 验证命令

```bash
uv run pytest -q
uv run mypy asr_server tests scripts
```

结果：

```text
48 passed, 2 skipped
Success: no issues found in 29 source files
```
