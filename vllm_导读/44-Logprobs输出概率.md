# 44 · Logprobs 输出概率

> 对应文件：`vllm/logprobs.py`、`vllm/v1/engine/logprobs.py`、`vllm/v1/sample/ops/logprobs.py`

## 什么是 logprobs

logprob = log probability，对数概率。除了返回生成的文字，模型还能告诉你**它对每个 token 有多确定**——即每个被选中的 token 的概率（取对数）。

类比：模型不只给答案，还给"信心分数"。

## 为什么有用

| 用途 | 说明 |
|------|------|
| 置信度评估 | 概率高 = 模型很确定；概率低 = 模型在"猜" |
| 检测幻觉 | 低概率片段可能是编造的 |
| 重排序 / 打分 | 用整句的概率给多个候选排序 |
| 研究分析 | 观察模型的决策过程 |
| 评测 | 很多 benchmark 靠 logprobs 算准确率（见 `41` 的 lm-eval） |

## 两种 logprobs

| 类型 | 含义 | 参数 |
|------|------|------|
| 采样 logprobs | 实际生成的每个 token 的概率（外加 top-k 候选的概率） | `logprobs=k` |
| prompt logprobs | 输入 prompt 里每个 token 的概率 | `prompt_logprobs=k` |

```python
SamplingParams(logprobs=5)         # 每步返回 top-5 候选 token 及其概率
SamplingParams(prompt_logprobs=5)  # prompt 各 token 的概率
```

## 从哪来：采样阶段顺带算

回顾 `06-采样.md`：模型 forward 产出 logits（每个 token 的原始分数）。对 logits 做一次 softmax 就得到概率，取对数就是 logprob。所以 logprobs 是在采样阶段顺带算出来的。

```
logits（原始分数）
  ↓ softmax
概率分布
  ↓ log + 取 top-k
logprobs（返回的对数概率 + 候选）
```

代码在 `v1/sample/ops/logprobs.py`（GPU 上计算）和 `v1/engine/logprobs.py`（引擎侧组装成返回结构）。

## 一个返回示例

```python
outputs = llm.generate("天空是", SamplingParams(logprobs=3, max_tokens=1))
# 模型生成 "蓝色的"，并告诉你它考虑过的 top-3：
# {
#   "蓝色的": -0.12  (概率约 89%，很确定)
#   "灰色的": -2.34  (概率约 10%)
#   "晴朗的": -4.56  (概率约 1%)
# }
```

负数越接近 0，概率越高（log(1)=0，log(小数)是负数）。

## 性能代价

返回 logprobs 有额外开销：

- `prompt_logprobs` 尤其贵——它要求模型计算 prompt 每个位置的概率，相当于不能跳过 prompt 的部分计算。回顾 `04-KVCache管理.md` 里 `skip_reading_prefix_cache` 的注释——要 prompt logprobs 时会影响前缀缓存的使用。
- 所以不需要时别开，开了会拖慢、增加返回体积。

## 在 API 里用

OpenAI 兼容接口（见 `23`）支持：

```python
client.chat.completions.create(
    model="...",
    messages=[...],
    logprobs=True,
    top_logprobs=5,   # 每个位置返回 top-5 候选
)
```

## 对新手的意义

- 知道"模型能返回每个 token 的概率/信心"这个能力
- 做评测、置信度分析、检测低质量输出时会用到
- 注意它有性能代价，按需开启

## 小结

- logprobs = 每个 token 的对数概率，反映模型的"信心"
- 两种：采样 logprobs（生成的 token）+ prompt logprobs（输入的 token）
- 在采样阶段由 logits 经 softmax + log 顺带算出
- 用于置信度评估、幻觉检测、重排序、评测
- 有性能代价（尤其 prompt logprobs，影响前缀缓存），按需使用
