# 10 · LoRA 适配器

> 对应目录：`vllm/lora/`

## 什么是 LoRA

LoRA（Low-Rank Adaptation）是一种**高效微调**技术。它不改动原始大模型的权重，而是额外训练一个很小的"补丁"（适配器），插在模型里改变其行为。

类比：原模型是一台标准机器，LoRA 是一个小插件。换插件就能让机器做不同的活，但机器主体不变。

## 为什么有用

| 好处 | 说明 |
|------|------|
| 训练便宜 | 只训练小补丁（参数量约为原模型的 0.1%~1%），省显存省时间 |
| 存储小 | 一个 LoRA 适配器通常几十 MB，而全量模型几十 GB |
| 一基多用 | 一个基础模型 + 多个 LoRA = 多个"专精模型"，共享同一份基础权重 |

## vLLM 的杀手锏：多 LoRA 同时服务

vLLM 能在**同一个服务里同时加载多个 LoRA**，不同请求用不同的适配器，还能动态加载/卸载。

```
基础模型（共享，只加载一份）
  ├─ 请求 A 用 LoRA-客服
  ├─ 请求 B 用 LoRA-翻译
  └─ 请求 C 不用 LoRA（用原始模型）
```

这意味着你不用为每个微调版本单独起一个服务，极大节省资源。

## LoRA 的数学原理（一句话版）

普通线性层算的是 `y = W·x`。LoRA 加一个低秩旁路：

```
y = W·x + (B·A)·x · scaling
```

其中 `W` 是冻结的原始权重，`A` 和 `B` 是两个很小的矩阵（这就是适配器）。因为 `A`、`B` 很小，所以参数量和存储都很省。

## 怎么用

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3-8B", enable_lora=True)

outputs = llm.generate(
    "翻译这句话",
    lora_request=LoRARequest("翻译适配器", 1, "/path/to/lora")
)
```

## 代码结构

| 文件/目录 | 作用 |
|-----------|------|
| `lora/request.py` | `LoRARequest`，请求里指定用哪个适配器 |
| `lora/layers/` | LoRA 版的各种层（在原层基础上加旁路） |
| `lora/models.py` | LoRA 适配器的加载与管理 |
| `lora/lora_weights.py` | 适配器权重的表示 |
| `worker/lora_model_runner_mixin.py` | 在模型执行时注入 LoRA |

## 批处理里的挑战

难点在于：同一个 batch 里，不同请求可能用不同的 LoRA。vLLM 用专门的 batched LoRA kernel，让一次 forward 能同时处理"用不同适配器的请求"，而不用拆成多次。这也是 vLLM LoRA 高效的关键。

## 小结

- LoRA = 给大模型加小补丁，高效微调，不动原始权重
- vLLM 能同时服务多个 LoRA，共享一份基础模型，省资源
- 原理：在线性层上加一个低秩旁路 `B·A`
- 代码核心在 `lora/`，执行时由 worker 注入
