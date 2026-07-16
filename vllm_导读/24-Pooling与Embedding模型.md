# 24 · Pooling 与 Embedding 模型

> 对应目录：`vllm/model_executor/layers/pooler/`、`vllm/v1/pool/`、`vllm/entrypoints/pooling/`

## 两类不同的模型任务

到目前为止，前面的导读都在讲**生成式**模型（generate）：给 prompt，逐 token 生成文本。

但还有另一大类任务，不生成文本，而是输出一个**向量或分数**。vLLM 把这类统称为 **pooling**（池化）任务。

| 任务类型 | 输出 | 例子 |
|----------|------|------|
| generate（生成） | 文本 | 聊天、补全 |
| pooling（池化） | 向量 / 分数 | 向量检索、重排序、分类 |

## 什么是 Embedding（向量化）

把一段文本变成一个固定长度的向量（比如 1024 个数字）。语义相近的文本，向量也相近。

用途：

- **语义搜索 / RAG**：把文档和查询都转成向量，算相似度找最相关的
- **聚类、去重、推荐**

```
"今天天气很好"  → [0.12, -0.45, 0.88, ...]（1024 维向量）
"今儿天气不错"  → [0.13, -0.43, 0.85, ...]（很接近 → 语义相似）
```

## 为什么生成模型和 pooling 共用一套引擎

它们的前半部分是一样的：都要把输入喂给模型跑 forward，得到 hidden states（隐藏状态）。区别只在最后一步：

```
输入 → 模型 forward → hidden states
                          ├─ 生成任务：→ 算 logits → 采样出 token（见 06）
                          └─ pooling 任务：→ 池化成一个向量/分数
```

所以 vLLM 复用了调度、KV Cache、worker 等全部基础设施，只是把"采样"换成"池化"。

## Pooler：池化层

`model_executor/layers/pooler/` 定义了怎么把一串 hidden states 浓缩成一个结果。常见池化方式：

| 方式 | 做法 |
|------|------|
| last token | 取最后一个 token 的隐藏状态（常用于 decoder 模型的 embedding） |
| mean | 所有 token 隐藏状态求平均 |
| CLS | 取特殊 `[CLS]` token 的隐藏状态（BERT 类模型） |

## 几种 pooling 任务

| 任务 | 输出 | 说明 |
|------|------|------|
| embedding | 向量 | 文本向量化 |
| classify | 类别分数 | 文本分类（如情感分析） |
| score / rerank | 相关性分数 | 给"查询-文档对"打分，用于重排序 |

## 怎么用

```python
from vllm import LLM

# 指定 runner 为 pooling
llm = LLM(model="intfloat/e5-mistral-7b-instruct", runner="pooling")
outputs = llm.embed(["今天天气很好"])
print(outputs[0].outputs.embedding)  # 得到向量
```

HTTP 服务对应 `/v1/embeddings` 端点（见 `23-OpenAI兼容API服务.md`）。

## 代码结构

| 目录 | 作用 |
|------|------|
| `model_executor/layers/pooler/` | 各种池化层实现 |
| `v1/pool/` | pooling 任务在 v1 引擎里的处理 |
| `entrypoints/pooling/` | pooling 的离线接口 |
| `pooling_params.py` | pooling 参数（类比 SamplingParams） |

回顾 `05-Worker模型执行.md` 的 `execute_model`，里面有一句 `if self.is_pooling_model: return self._pool(...)`——这就是 pooling 任务的分叉点：算完 hidden states 后走池化而不是采样。

## 没有 KV Cache 复用的差异

很多 embedding 模型是 encoder（如 BERT 类），一次性处理整个输入，不像生成模型那样逐 token。所以它们通常没有 decode 阶段，KV Cache 的使用方式也不同（甚至不需要）。引擎会据此调整调度（回顾 `02-引擎.md` 里"没有 KV Cache 的 encoder 模型不支持 chunked prefill"那段）。

## 小结

- vLLM 不只能生成文本，还能做 pooling 任务：输出向量或分数
- 典型是 embedding（文本向量化），用于语义搜索、RAG、重排序
- 与生成共用引擎，区别只在最后一步：池化 vs 采样
- Pooler 决定怎么把 hidden states 浓缩成结果（last/mean/CLS）
- 对应 `/v1/embeddings` 等端点
