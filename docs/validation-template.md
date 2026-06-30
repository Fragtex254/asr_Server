# WSL ASR 验收记录模板

## 环境

- 日期：
- WSL 发行版：
- 项目路径：`/home/fragt/services/asr-server`
- Git commit：
- Python 版本：
- uv 版本：
- torch 版本：
- torch CUDA 版本：
- GPU 名称：
- GPU capability：
- `nvidia-smi` 摘要：

## 后端预验收

### transformers

- 命令：
- 模型 ID：
- 音频文件：
- 输出语言：
- 文本前 200 字：
- 是否通过：
- 失败日志：

## 服务端 API 验收

- 服务地址：
- `/health` 摘要：
- `/v1/models` 模型和后端列表：
- `qwen3-asr-0.6b` + `transformers` 文本前 200 字：
- `qwen3-asr-1.7b` + `transformers` 文本前 200 字：
- `backend=vllm` 未声明后端错误码：
- `timestamps=word` 未声明能力错误码：
- 卸载活跃请求时的状态：
- `unloading_scheduled` 新请求错误码：

## 转录耗时记录

- 是否返回 `timings.total_ms`：
- 是否返回 `timings.load_ms`：
- 是否返回 `timings.inference_ms`：
- 短音频总耗时：
- 长音频总耗时：
- 是否避免记录完整音频内容：
- 同步请求结束后是否清理临时音频文件：
- 转录结束 3 分钟无新请求后是否自动卸载模型：
- 新请求是否会重置空闲卸载计时：

## 长音频切分与合并

- 使用音频文件：`test-fixtures/audio/test_long.mp3`
- 切分策略：
- VAD backend：
- chunk 数量：
- overlap 秒数：
- 平均 chunk 时长：
- 最长 chunk 时长：
- 空白/低质量 chunk 数：
- 是否返回 chunk 元数据：
- 合并文本前 200 字：
- 是否通过：

## 异步 job 与进度验收

- `POST /v1/audio/transcription-jobs` 是否返回 `202`：
- 返回的 `job_id`：
- 返回的 `status_url`：
- 短音频 job 模型 ID：
- 短音频 job backend：
- 短音频 job 最终状态：
- 短音频 job 文本前 200 字：
- 长音频 job 使用音频：`test-fixtures/audio/test_long.mp3`
- 长音频 job 模型 ID：
- 长音频 job backend：
- 长音频 job 状态流转时间线：
- queued/preprocessing/splitting 阶段轮询样例 JSON：
- transcribing 阶段轮询样例 JSON：
- completed 阶段轮询样例 JSON：
- `progress.total_chunks`：
- `progress.completed_chunks` 是否真实增长：
- `progress.current_chunk` 是否真实变化：
- `queue_position` 验收结果：
- 同时提交两个 job 时是否只运行一个：
- 第二个 job 是否显示 queued：
- job 失败错误对象是否使用统一 `code/message/details`：
- queued job 取消结果：
- running job 取消是否等待当前 chunk 结束：
- job 完成/失败/取消后临时文件是否清理：
- job result TTL 配置：
- job TTL 是否只保留内存结果、不保留音频文件：
- 进程重启后内存 job 丢失行为是否记录：

## Silero VAD 对比

- energy VAD chunk 数：
- energy VAD 总转录耗时：
- Silero VAD chunk 数：
- Silero VAD 总转录耗时：
- Silero 是否 fallback：
- Silero fallback 原因：
- 观察到的切断句/重复文本问题：
- 结论：

## context / 热词验收

- 测试音频：
- context 内容摘要：
- context 字符数：
- 易错专有名词清单：
- 无 context 命中数：
- 有 context 命中数：
- 是否出现 context 幻觉插词：
- 结论：

## max_new_tokens 验收

- 测试音频：
- 默认 `max_new_tokens`：
- 对照 `max_new_tokens`：
- 默认输出是否截断：
- 对照输出是否截断：
- 推理耗时变化：
- 峰值显存变化：
- 结论：

## Qwen batch transcription 验收

- 测试音频：`test-fixtures/audio/test_long.mp3`
- batch size 1 总耗时：
- batch size 2 总耗时：
- batch size 4 总耗时：
- 0.6B 稳定 batch size：
- 1.7B 稳定 batch size：
- batch fallback 次数：
- CUDA OOM 或异常日志：
- 推荐配置：

## Mac 局域网验收

- Mac 请求是否绕过代理：
- `curl --noproxy '*' http://192.168.31.137:18080/health` 结果：
- `curl --noproxy '*' http://192.168.31.137:18080/v1/models` 结果：
- Mac 上传真实音频转录摘要：
- Mac 创建 job 请求摘要：
- Mac 轮询 job 到 completed 的状态摘要：
- Mac 侧观察到的 chunk 进度变化：

## 结论

- 通过/不通过：
- 阻塞点：
- 后续动作：
