# MOSS 长音频播客基建优化交接

## 给下一位 AI Agent 的任务提示词

你现在接手本仓库的 MOSS 长音频转录基建优化。请直接完成调查、设计、实现、测试、部署验证和文档更新，不要只给方案。开始前先完整阅读：

- `AGENTS.md`
- `docs/asr-server-prd.md`
- `docs/wsl-deployment.md`
- `docs/validation-2026-07-10-wsl-moss.md`
- `docs/moss-transcribe-diarize-wsl-deployment-plan.md`

## 背景与实测证据

真实测试音频位于 Mac mini：

`/Users/jinghao/Downloads/【十字路口】AI时代是谁的黄金时代？｜和张咋啦聊：文科生、积极行动、爆款的规律、普通人也能赢【视频播客】/1-【十字路口】AI时代是谁的黄金时代？｜和张咋啦聊：文科生、积极行动、爆款的规律、普通人也能赢【视频播客】-1080P 高码率-AVC.mp3`

音频约 5406 秒、86,498,322 bytes、128 kbps。通过局域网服务 `http://192.168.31.137:18080` 使用 `moss-transcribe-diarize-0.9b`、`response_format=verbose_json`、`preserve_segments=true`、`language=auto` 实测成功：

- job：`job_355c79cb2d774bb3abff2de8b4ecd516`
- 总耗时约 764 秒，推理约 740 秒
- 服务端按通用 `auto` 策略切成 47 个约 120 秒 chunk
- 返回约 37,371 字、546 个 speaker segments
- warnings 包含 `moss_speaker_labels_are_chunk_local`、`moss_segment_clamped_to_chunk`、`silero_streaming_not_validated_fallback_to_energy`
- 说话人在每个 chunk 内基本能正确分成两人，但跨 chunk 的 `S01/S02` 会翻转。约 115 秒边界处，同一个嘉宾从 `chunk-0000:S02` 变成 `chunk-0001:S01`，主持人随后成为 `chunk-0001:S02`
- 546 个 segment 中有大量很短的“对 / 嗯”等反馈；结构适合逐字稿，但需要后处理后再用于总结
- `language=zh` 会返回 422 `capability_not_supported`；该模型当前只声明 `auto`

原始音频、完整逐字稿与响应 JSON 不进入 Git。需要复测时重新提交上述音频，并把原始结果保存为临时验收产物。

官方模型卡：<https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>。官方将该模型定义为原生 long-form、多说话人、一次生成全局 speaker-aware transcript；长音频示例建议提高 `max_new_tokens`，可到 65536。当前服务端却使用 MOSS 默认 2048、全局硬上限 4096，并在 `asr_server/transcription.py` 中复用 Qwen 的通用切片流程。这很可能破坏了 MOSS 的长程 speaker memory。

## 核心目标

把执行策略从“所有 ASR 模型共用一种切片方式”升级为“根据模型能力选择执行策略”：

- Qwen 保持现有 chunked ASR 行为和稳定默认值，不能回归。
- MOSS 默认优先走 native long-form：尽可能一次处理完整音频，保留整期节目全局一致的 speaker ID。
- `split_strategy=auto` 应变成真正的 model-aware policy，而不是固定进入同一个 120 秒 splitter。
- MOSS 使用模型专属、可配置且有安全边界的 `max_new_tokens` 策略；不要简单提高所有模型的全局上限。
- MOSS 结果继续支持 `verbose_json`，完整保留 `segments`、起止时间、speaker、chunks、warnings 和 timings。
- 如果受 16GB VRAM、模型上下文或稳定性限制必须降级切片，必须明确返回 speaker scope 为 chunk-local，不能伪装成全局 diarization；同时设计可演进的 global speaker stitching 边界。

## 必须先回答并用证据验证的问题

1. RTX 5070 Ti 16GB + 当前 transformers adapter 是否能对 10、30、60、90 分钟音频进行单次推理？
2. 不切片时峰值显存、生成 token 数、总耗时和尾部完整性分别如何？
3. 当前 4096 token 硬上限应如何改成模型专属限制？如何避免输出被静默截断？
4. MOSS 原生一次推理后，整期是否保持稳定的全局 `S01/S02`？
5. 如果 90 分钟单次推理不可行，合理的降级窗口应是多少？能否通过重叠区或声纹 embedding 做可靠的全局 speaker stitching？不要凭猜测实现，先用实测证明。

## 实施要求

- 先理解现有 `transcription.py`、audio splitter/merger、jobs、registry capabilities、MOSS adapter 和测试，不做散落的 `if model == moss` 补丁堆积；建立清楚的模型执行策略边界。
- 保留 OpenAI-compatible API；如果新增字段或 capability，更新 schema、`/v1/models`、客户端脚本、PRD 和端点能力文档。
- 异步 job 必须继续提供真实阶段、进度、取消、失败和过期语义。native long-form 单次模型调用无法提供虚假 chunk 百分比，应只报告真实阶段。
- 必须检测并显式报告生成截断、OOM、上下文超限和不支持参数，不能返回表面 `completed` 的残缺正文。
- `timestamps=none` 对 MOSS 不表示没有 segment 时间戳；MOSS 的 segment timestamps 来自模型输出解析。不要错误地改成 `word` 或 `char`，当前模型会拒绝。
- 真实 GPU、CUDA、模型依赖和长期服务部署只在 Windows PC 的 WSL Arch Linux `/home/fragt/services/asr-server` 进行；Mac mini 只做代码、mock/unit tests 和局域网客户端验收。
- 局域网请求必须绕过本机代理。不要输出、提交或记录 API Key；不要把音频、完整逐字稿、模型缓存或临时结果加入 Git。
- 遵循既有部署脚本和 systemd 约定，不新增公网暴露，不开放额外端口。

## 验收标准

- 现有单元测试全部通过，并补充 model-aware execution policy、MOSS token limit、native long-form job、降级语义、能力声明和错误处理测试。
- Qwen 0.6B/1.7B 原有切片、job 和 smoke 行为不回归。
- 在 WSL RTX 5070 Ti 上按 10 → 30 → 60 → 90 分钟阶梯实测，不要一开始盲目塞 90 分钟。
- 对上述 90 分钟播客给出可复查的验收记录：是否单 chunk、峰值显存、耗时、生成 token、文本是否覆盖到结尾、segment 数、全局 speaker 数、跨原 120 秒边界是否仍发生身份翻转。
- 理想验收：整期稳定为两位全局 speaker，时间轴覆盖完整，结尾未截断，不再出现 `moss_speaker_labels_are_chunk_local`。
- 若硬件无法完成理想路径，必须给出失败证据和明确降级设计，不能悄悄恢复 120 秒切片并宣称支持全局 diarization。
- 更新 `docs/` 下相应规范和新的真实验收记录，区分“代码支持”与“WSL 实测通过”。

完成后请汇报：根因、架构决策、修改文件、测试结果、WSL 实测数据、尚存限制，以及 tts-broadcast 客户端下一步应消费的稳定契约。除非遇到必须由用户决定的高风险分歧，否则持续推进到真实验收完成。

## Suggested skills

- `implement`：按现有规范完成可验证的基建改造。
- `tdd`：先锁定 model-aware policy、token 上限和降级契约，再实现。
- `diagnosing-bugs`：用于定位长音频截断、OOM、speaker 漂移或 job 状态异常。
- `web-access`：若需核对模型行为，只使用官方模型卡、官方仓库或论文等一手资料。
