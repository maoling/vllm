# 39 · 工具调用与 MCP 集成

> 对应目录：`vllm/entrypoints/mcp/`、`vllm/tool_parsers/`

## 什么是工具调用 (Tool Calling)

让模型不只是聊天，还能**调用外部工具/函数**来完成任务。比如问"今天北京天气"，模型自己不知道，但它能输出"我要调用 `get_weather(city='北京')`"，由你的程序执行后把结果喂回给它。

也叫 Function Calling，是构建 AI Agent 的基础能力。

类比：给模型配了一套"API 调用能力"，它能决定何时调哪个 API。

## 完整的工具调用循环

```
① 你告诉模型有哪些工具可用（tools 定义：函数名、参数 schema）
   ↓
② 用户提问："北京天气怎么样？"
   ↓
③ 模型判断需要调工具，输出：调用 get_weather(city="北京")
   ↓ vLLM 用 tool parser 解析出结构化调用（见 13）
④ 你的程序实际执行 get_weather("北京") → "晴，25度"
   ↓ 把结果作为新消息喂回模型
⑤ 模型基于工具结果回答："北京今天晴，25度。"
```

vLLM 负责的是 ③（让模型生成调用）和"解析调用"这步；实际执行工具（④）通常由调用方做。

## 两个关键部分

### 1. Tool Parser（解析模型的工具调用输出）

详见 `13-推理解析与工具调用.md`。模型输出里的工具调用是用特殊格式包裹的文本，`vllm/tool_parsers/` 把它解析成标准结构（函数名 + 参数）。每个模型格式不同，各有 parser。

### 2. 结构化输出保证格式正确

工具调用的参数必须是合法 JSON（符合函数的参数 schema）。这里会用到 `15-结构化输出.md` 的能力——约束模型只能生成合法的参数 JSON，避免解析失败。

## MCP 是什么

MCP = **Model Context Protocol**（模型上下文协议），是一个让 AI 模型和外部工具/数据源标准化对接的协议（由 Anthropic 提出，正在成为生态标准）。

简单说：MCP 定义了一套标准接口，让"工具提供方"和"模型使用方"能即插即用，不用为每个工具写定制对接。

类比：MCP 之于 AI 工具，就像 USB 之于硬件——统一接口，互相兼容。

## vLLM 的 MCP 集成

`vllm/entrypoints/mcp/` 让 vLLM 能对接 MCP 工具：

| 文件 | 作用 |
|------|------|
| `mcp/tool_server.py` | MCP 工具服务端对接 |
| `mcp/tool.py` | 工具的表示与调用 |

这让 vLLM 部署的模型能直接使用遵循 MCP 协议的工具生态，简化了 Agent 应用的搭建。

## 怎么用（工具调用示例）

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }
}]

response = client.chat.completions.create(
    model="...",
    messages=[{"role": "user", "content": "北京天气?"}],
    tools=tools,           # 告诉模型有哪些工具
)
# response 里会包含模型决定的工具调用
```

启动服务时需指定对应的 tool parser：

```bash
vllm serve <model> --enable-auto-tool-choice --tool-call-parser <模型对应的parser>
```

## 和其他模块的关系

工具调用是多个模块的"合奏"：

- `13` tool parser：解析调用
- `15` 结构化输出：保证参数 JSON 合法
- `23` API 服务：通过 `tools` 参数接收工具定义
- `38` 输入处理：把工具定义和结果渲染进 prompt

## 小结

- 工具调用让模型能决定调用外部函数，是 Agent 的基础
- 循环：声明工具 → 模型生成调用 → 解析 → 执行 → 喂回结果 → 模型作答
- vLLM 负责"生成调用 + 解析"，靠 tool parser（13）+ 结构化输出（15）
- MCP 是工具对接的标准协议（USB 类比），vLLM 有专门集成
