# 代理开发指南

## 项目背景

这个仓库用于实现一个局域网 ASR 网关：Mac mini 客户端通过 HTTP 调用运行在 Windows WSL Arch Linux 内的 GPU ASR 服务。

主要部署目录：

```text
/home/fragt/services/asr-server
```

局域网公开入口：

```text
http://192.168.31.137:18080
```

Mac mini 只作为轻量开发机和客户端验收机。真实 GPU 推理、CUDA 验证、模型包安装、长期后台服务部署，都必须在 Windows PC 的 WSL Arch Linux 内完成。

## 必读文档

实现服务行为前，必须先读：

```text
docs/asr-server-prd.md
```

把任务交给专门代理时，使用这些提示词：

```text
prompts/server-agent.md
prompts/request-client-agent.md
```

## 开发原则

- 公共 API 必须和 `docs/asr-server-prd.md` 保持一致。
- 使用 Python 3.12、uv、FastAPI 和 Uvicorn 实现服务。
- 服务入口监听 `0.0.0.0:18080`。
- WSL 项目位置不要放在 `/mnt/c`；正式部署目录是 `/home/fragt/services/asr-server`。
- 除非 PRD 明确变更，不要把 worker 端口 `8001` 暴露到局域网。
- 不要把历史测试端口 `8765` 写入实现、部署、防火墙或启动配置。
- API、生命周期管理、测试和第一条转录路径完成前，不要做 Web UI。

## 跨平台边界

- macOS 侧可以创建项目骨架、schema、测试、mock 适配器、客户端验证脚本和文档。
- macOS 侧不得要求 CUDA、NVIDIA 驱动、模型下载或大型本地模型缓存。
- WSL Arch Linux 侧负责 CUDA 检查、`nvidia-smi`、磁盘空间检查、真实 Qwen 依赖安装和模型推理验证。
- 大模型依赖必须在适配器内懒加载，保证基础 import 和测试在无 GPU 环境也能运行。
- 路径处理保持 POSIX 兼容，不要在服务代码里硬编码 macOS 专用路径。

## API 与生命周期规则

- 模型能力发现必须来自 `GET /v1/models`；客户端不要硬编码模型能力。
- 模型状态必须使用 PRD 枚举：`unloaded`、`loading`、`loaded`、`unloading_scheduled`、`unloading`、`error`。
- 每个模型必须维护活跃请求计数和生命周期锁。
- 如果卸载请求到来时仍有活跃请求，设置 `unloading_scheduled`，新的同模型请求返回 `409 model_unloading_scheduled`，等活跃请求完成后再卸载。
- 推理仍在执行时，不要强制卸载模型。
- 错误必须使用 PRD 错误信封：

```json
{
  "error": {
    "code": "model_not_found",
    "message": "unknown model: xxx",
    "details": {}
  }
}
```

## 网络与安全

- 唯一面向局域网的 API 端口是 `18080`。
- Mac 客户端请求 `192.168.31.137` 时必须绕过本机代理，例如使用 `curl --noproxy '*'`。
- 支持可选 Bearer Token 鉴权，但不要假设服务会暴露到公网。
- 不要添加公网隧道、端口映射或互联网暴露。
- 上传音频应写入临时位置，并在推理结束后清理。
- 日志默认不要保存完整音频内容。

## 测试要求

至少覆盖：

- 健康检查。
- 模型列表和单模型状态。
- 模型加载和卸载行为。
- 卸载等待活跃请求完成。
- `unloading_scheduled` 状态下拒绝新请求。
- 转录接口参数校验。
- 未声明能力的错误处理，例如 timestamps、forced alignment 或 streaming 返回 `capability_not_supported`。
- Qwen 两个尺寸和 `/v1/models` 中声明的所有后端的转录路径。

先用 mock 适配器覆盖生命周期和 API 行为，再在 WSL Arch Linux 内接入真实 Qwen 模型。

## Git 规范

- 生成缓存、虚拟环境、模型缓存、上传文件和本地运行数据不要进 Git。
- 文档和提示词保持在稳定路径，方便代理交接。
- 提交要聚焦，提交信息要清楚。
- 不要提交机器专用 secret、token、下载模型或包含隐私内容的音频样本。
