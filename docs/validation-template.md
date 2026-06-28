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

## 长音频切分与合并

- 使用音频文件：`test-fixtures/audio/test_long.mp3`
- 切分策略：
- chunk 数量：
- overlap 秒数：
- 是否返回 chunk 元数据：
- 合并文本前 200 字：
- 是否通过：

## Mac 局域网验收

- Mac 请求是否绕过代理：
- `curl --noproxy '*' http://192.168.31.137:18080/health` 结果：
- `curl --noproxy '*' http://192.168.31.137:18080/v1/models` 结果：
- Mac 上传真实音频转录摘要：

## 结论

- 通过/不通过：
- 阻塞点：
- 后续动作：
