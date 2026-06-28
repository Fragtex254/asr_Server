# WSL 服务端下一阶段执行计划

## 当前决策

第一版继续聚焦 Qwen3-ASR + `transformers` 后端。`vllm` 暂不作为第一版目标，MiMo 也暂不进入第一版实现。

WSL agent 接手后，按下面顺序推进。不要跳到 MiMo 或 vLLM，除非前面的 Qwen transformers 路径已经稳定验收。

## 1. 转录耗时记录

先实现转录耗时记录。这一项风险最小，但能为后续长音频和真实推理性能分析提供基础数据。

建议响应字段：

```json
{
  "timings": {
    "total_ms": 12345,
    "load_ms": 0,
    "decode_ms": 120,
    "inference_ms": 11800,
    "postprocess_ms": 200
  }
}
```

要求：

- `total_ms` 必须覆盖一次转录请求的端到端服务端耗时。
- 自动加载模型时记录 `load_ms`；模型已加载时 `load_ms` 可以为 0。
- 真实 Qwen adapter 至少记录 `inference_ms`。
- 不要把完整音频内容写入日志。
- 保持原有 `usage.audio_seconds` 字段。

测试：

- mock adapter 下返回 `timings.total_ms`，且数值大于等于 0。
- 自动加载模型时有 `load_ms` 字段。
- 已加载模型再次转录时 `load_ms` 为 0 或接近 0。
- 错误响应仍使用统一 error envelope。

## 2. 长音频切分与合并

第二步做长音频切分与合并。先用纯 Python 模块和 mock adapter 测透，不要一开始就绑定真实 Qwen。

建议模块：

```text
asr_server/audio/
  metadata.py
  splitter.py
  merger.py
```

第一版先支持：

- `split_strategy=auto|none|fixed`
- `max_chunk_seconds`
- `overlap_seconds`
- `preserve_segments`
- 返回 chunk 级元数据

先不要急着做 VAD。固定切分和时间线合并先跑通。

测试：

- 短音频不切分。
- 长音频按 chunk 秒数切分。
- overlap 不超过 chunk 长度。
- chunk 时间线连续且不乱序。
- 合并后返回完整文本。
- `preserve_segments=true` 时返回 chunk 级结果。
- 超过服务端限制返回 `422 duration_limit_exceeded` 或 PRD 指定错误码。

验收音频：

```text
test-fixtures/audio/test_short.wav
test-fixtures/audio/test_long.mp3
```

## 3. Qwen transformers 能力补全

第三步只围绕 Qwen `transformers` 后端补能力。不要碰 vLLM。

优先补：

- `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b` 都能真实转录。
- `language=auto|zh|en` 参数映射正确。
- 临时音频文件推理结束后清理。
- Qwen 异常转换成统一 error envelope。
- 真实模型加载耗时和推理耗时记录到 `timings`。
- `test_short.wav` 和 `test_long.mp3` 都有验收记录。

高级能力策略：

- timestamps、forced alignment、streaming 不要默认打开。
- 只有 transformers 后端真实跑通并测试后，才能在 `/v1/models` capabilities 中声明。
- 没跑通的能力必须返回 `capability_not_supported`，或不暴露入口。

## 4. MiMo transformers 后续调研

MiMo 放到 Qwen transformers 稳定之后。

开始 MiMo 前必须满足：

- Qwen 0.6B / 1.7B transformers 已稳定。
- 长音频切分在 mock 和 Qwen 下都能跑。
- `timings` 能记录真实耗时。
- `/v1/models` 能准确声明模型与能力。

MiMo 第一阶段只做调研和最小 adapter，不默认加入 `/v1/models`。只有真实转录验收通过后，才允许声明。

## 不做事项

- 第一版不做 vLLM。
- 第一版不做 MiMo。
- 不做公网访问。
- 不做 Web UI。
- 不声明未验收的模型、后端或高级能力。

