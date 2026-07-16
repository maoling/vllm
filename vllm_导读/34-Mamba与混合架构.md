# 34 · Mamba 与混合架构模型

> 对应目录：`vllm/model_executor/layers/mamba/`、`vllm/v1/core/kv_cache_coordinator.py`

## 背景：Transformer 的痛点

回顾 `04-KVCache管理.md`：Transformer 的注意力要"回看"前面所有 token，所以 KV Cache 随序列长度**线性增长**。序列越长，显存占用越大、计算越慢。处理超长文本（几十万 token）时，这成了大问题。

## Mamba：另一条技术路线

Mamba 是一种 **状态空间模型（SSM）**，思路和 Transformer 不同：

- Transformer：保存所有历史 token 的 KV（占用随长度增长）
- Mamba：维护一个**固定大小的"状态"**，像 RNN 一样，把历史压缩进这个状态里

```
Transformer: [token1][token2]...[token1000] 全都要存 KV  → 占用越来越大
Mamba:       一个固定大小的状态 S，每来一个 token 更新 S  → 占用恒定
```

好处：处理长序列时，显存和计算开销**不随长度暴涨**，更适合超长上下文。

## 混合架构 (Hybrid)

实践中，纯 Mamba 在某些任务上不如 Transformer。于是出现了**混合架构**：在一个模型里同时用 Mamba 层和注意力层，取长补短。

代表模型：Jamba、部分新版模型、MiniMax 等。

```
混合模型的层堆叠（示意）：
  Mamba 层
  Mamba 层
  注意力层    ← 偶尔插入注意力
  Mamba 层
  ...
```

## 给 vLLM 带来的挑战：两种"缓存"并存

这是混合架构在 vLLM 里最有意思的地方。两种层需要缓存的东西完全不同：

| 层类型 | 缓存什么 | 大小特点 |
|--------|----------|----------|
| 注意力层 | KV Cache（每个 token 一份） | 随长度增长 |
| Mamba 层 | 状态 state（固定大小） | 恒定 |

vLLM 的 KV Cache 管理器原本是为注意力的 KV 设计的。要支持混合模型，就得让缓存系统**同时管理两种不同性质的缓存**。

## HybridKVCacheCoordinator

这就是 `v1/core/kv_cache_coordinator.py` 里 `HybridKVCacheCoordinator` 的职责：协调多种缓存类型。回顾 `03-调度器.md` 里出现的 `HybridKVCacheCoordinator`、`has_mamba_layers`、`need_mamba_block_aligned_split`——都是在处理混合模型的特殊性。

代码里还有很多 `mamba` 相关的特殊处理，比如：

- `mamba_block_aligned_split`：Mamba 状态要按块对齐切分
- `mamba_cache_mode`：Mamba 缓存的管理模式

这些都是为了让混合模型也能享受 vLLM 的连续批处理、分页管理等机制。

## 代码结构

| 目录/文件 | 作用 |
|-----------|------|
| `model_executor/layers/mamba/` | Mamba 层实现 |
| `v1/core/kv_cache_coordinator.py` | 协调多种缓存（含混合） |
| `v1/worker/mamba_utils.py` | worker 侧 Mamba 状态处理 |
| `config/mamba.py` | Mamba 相关配置 |

## 对新手的意义

- 大多数主流模型还是纯 Transformer，用不到这些
- 但理解"Mamba 用固定状态替代增长的 KV Cache"这个核心思想，能帮你理解长上下文优化的方向
- 混合模型是较前沿的方向，vLLM 对它的支持体现了缓存系统的灵活性

## 小结

- Transformer 的 KV Cache 随长度增长，是长序列的瓶颈
- Mamba 用固定大小的状态压缩历史，开销不随长度暴涨
- 混合架构同时用 Mamba 层和注意力层，取长补短
- vLLM 用 HybridKVCacheCoordinator 同时管理"增长的 KV"和"固定的状态"两种缓存
