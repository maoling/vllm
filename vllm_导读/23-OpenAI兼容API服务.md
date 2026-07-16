# 23 · OpenAI 兼容 API 服务

> 对应目录：`vllm/entrypoints/openai/`

## 这个模块负责什么

把 vLLM 包装成一个 **HTTP 服务**，而且 API 格式和 OpenAI 一模一样。这样原本用 OpenAI 的代码，改个 base_url 就能切到自己部署的 vLLM，无需改其他代码。

类比 Web 后端：这是完整的 **Controller 层 + 路由 + 请求/响应 DTO**，基于 FastAPI。

## 为什么"OpenAI 兼容"很重要

OpenAI 的 API 格式已经成了事实标准，大量工具、SDK、框架都按它来。vLLM 兼容它，意味着：

- 现有生态直接可用（LangChain、各种客户端 SDK）
- 迁移成本几乎为零
- 自己部署开源模型，替换闭源 API

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="任意值")
client.chat.completions.create(model="...", messages=[...])  # 和用 OpenAI 一样
```

## 启动方式

```bash
vllm serve meta-llama/Llama-3-8B
# 启动一个监听 8000 端口的 HTTP 服务
```

## 主要 API 端点

| 端点 | 用途 | 对应代码目录 |
|------|------|-------------|
| `/v1/chat/completions` | 聊天补全（最常用） | `openai/chat_completion/` |
| `/v1/completions` | 文本补全（老式） | `openai/completion/` |
| `/v1/embeddings` | 文本向量化 | （见 24-Pooling模型） |
| `/v1/responses` | Responses API（新，你打开的文件） | `openai/responses/` |
| `/v1/models` | 列出可用模型 | `openai/models/` |
| `/health` | 健康检查 | `api_server.py` |
| `/metrics` | Prometheus 指标 | （见 22-性能指标） |

## 代码结构与分层

```
openai/
├── api_server.py        ← FastAPI 应用主入口（路由、生命周期、启动）
├── cli_args.py          ← 命令行参数
├── chat_completion/     ← 聊天补全的请求处理逻辑
│   ├── protocol.py      ← 请求/响应的数据结构（DTO）
│   └── serving.py       ← 业务逻辑：调用引擎、组装响应
├── completion/          ← 文本补全
├── responses/           ← Responses API
└── models/              ← 模型管理
```

每种 API 大致都有两类文件：

- `protocol.py`：定义请求/响应格式（用 Pydantic，相当于 DTO + 校验）
- `serving.py`：处理逻辑（接收请求 → 调引擎 → 流式/非流式返回）

## 一个请求在 API 层的流程

以 `/v1/chat/completions` 为例：

```
HTTP 请求 (JSON)
  ↓ FastAPI 路由 (api_server.py)
  ↓ Pydantic 解析成 ChatCompletionRequest (protocol.py)
  ↓ serving.py:
  │   · 应用聊天模板（messages → prompt 文本，见 chat_utils.py）
  │   · 构造 SamplingParams
  │   · 调用引擎 engine.generate()（接入 01~07 的核心流程）
  │   · 收集输出
  │   · （可选）调用 reasoning/tool parser 解析（见 13）
  ↓ 组装成 OpenAI 格式的响应
HTTP 响应 (JSON 或 SSE 流)
```

> 关键衔接点：API 层是核心流程（`01~07`）的"外壳"。它负责协议转换（OpenAI 格式 ↔ vLLM 内部格式），真正的推理还是交给引擎。

## 流式响应 (Streaming / SSE)

聊天接口支持 `stream=True`，服务端用 **SSE（Server-Sent Events）** 逐 token 推送：

```
data: {"choices":[{"delta":{"content":"你"}}]}
data: {"choices":[{"delta":{"content":"好"}}]}
data: [DONE]
```

这要求引擎逐 token 产出（回顾 `step()` 每步吐一个 token），detokenizer 增量解码（见 `18`），parser 也得支持流式解析（见 `13`）。你打开的 `responses/streaming_events.py` 就是处理这种流式事件的。

## Chat Template（聊天模板）

聊天接口收到的是 `messages`（角色+内容的列表），但模型吃的是一段纯文本。**聊天模板**负责把对话拼成模型期望的格式：

```
[{"role":"user","content":"你好"}]
  ↓ 应用模型的 chat template
"<|user|>\n你好<|assistant|>\n"
```

每个模型的模板不同（来自 tokenizer 配置）。逻辑在 `entrypoints/chat_utils.py`。

## 认证与安全

> 注意：vLLM 默认的 API 服务**不带强认证**（只有一个可选的简单 API key）。如果对外暴露，务必自己加一层网关/鉴权/限流，不要把裸服务直接暴露到公网。

## 小结

- 这层把 vLLM 包装成 OpenAI 兼容的 HTTP 服务，生态无缝衔接
- 基于 FastAPI，每种 API 有 protocol（DTO）+ serving（逻辑）两部分
- 职责是"协议转换"，真正推理交给核心引擎（01~07）
- 支持 SSE 流式输出，依赖引擎逐 token 产出 + 增量解码
- 生产部署注意自己加鉴权
