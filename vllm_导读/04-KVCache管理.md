# 04 · KV Cache 管理 (PagedAttention)

> 对应文件：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`

## 先搞懂：KV Cache 是什么

大模型生成文本是一个 token 一个 token 蹦出来的。每生成一个新 token，都要"回顾"前面所有 token。为了不重复计算，模型把前面每个 token 的中间结果（叫 Key 和 Value，简称 KV）缓存起来。这就是 **KV Cache**。

问题：KV Cache 很占显存。比如一个请求生成 2000 个 token，可能要占几百 MB 显存。同时服务几十个请求，显存很快就爆了。

## 传统做法的浪费

传统框架给每个请求预留"最大可能长度"的连续显存。比如设定最大 2048 token，那不管请求实际生成多少，都先占 2048 的空间。

结果：一个只生成 50 token 的请求，也占了 2048 的坑，**97% 的显存被浪费**。

## PagedAttention：像操作系统管内存一样

vLLM 的解法借鉴了操作系统的**虚拟内存分页**：

- 把 KV Cache 显存切成固定大小的小块，叫 **block**（比如每块存 16 个 token）
- 请求需要多少就给多少块，用完再要
- 块不需要物理连续（就像内存页可以散落在各处）
- 请求结束就回收它的块，给别人用

类比 JVM：

| JVM 内存管理 | vLLM KV Cache 管理 |
|--------------|---------------------|
| 堆切成 Region | 显存切成 Block |
| 对象分配 | `allocate_slots()` |
| GC 回收 | `free()` |
| Full GC（STW） | 抢占（preemption） |
| 对象共享/引用 | 前缀缓存（prefix cache） |

## 核心类

| 类/文件 | 作用 |
|---------|------|
| `KVCacheManager` | 对外的管理接口（分配、释放、查缓存） |
| `BlockPool`（`block_pool.py`） | 物理块池子，管理所有空闲/占用的块 |
| `KVCacheBlocks` | 一个请求持有的块的集合 |

## 关键方法 1：allocate_slots() —— 分配块

`kv_cache_manager.py` 第 244 行。调度器每次想给请求安排新 token 时调用它：

```python
def allocate_slots(self, request, num_new_tokens, ...):
    # 算一算这些新 token 需要几个块
    # 从 block_pool 里取出空闲块分配给请求
    # 如果池子空了（显存满了），返回 None
    #   → 调度器收到 None 就知道要抢占别的请求
```

返回 `None` 是一个重要信号：**显存不够了**。调度器据此触发抢占（见 `03-调度器.md`）。

## 关键方法 2：get_computed_blocks() —— 前缀缓存

`kv_cache_manager.py` 第 202 行。这是一个重要优化：

```python
def get_computed_blocks(self, request):
    # 查找：这个请求的 prompt 前缀，是不是之前已经算过、缓存还在？
    computed_blocks, num_computed = self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
    return computed_blocks, num_computed
```

**前缀缓存（Prefix Caching）** 的价值：

假设很多请求都以同样的系统提示词开头（比如"你是一个有帮助的助手..."）。这部分前缀只需算一次，后续请求直接复用缓存的块，省下大量计算。

实现原理：给每个块算一个 hash（基于它包含的 token），hash 相同就能复用。多个请求可以**共享**同一个物理块（类似引用计数）。

## 关键方法 3：free() —— 回收

请求生成完毕后，释放它占用的所有块，归还给 `block_pool`：

```python
def free(self, request):
    # 把 request 占用的块还回池子
    # 如果开了前缀缓存，块不会立刻清空，
    #   而是先留着，万一后面有请求能复用
```

## Block 大小的权衡

block 大小（一块存几个 token）是个权衡：

- 块太大：浪费多（最后一块可能填不满）
- 块太小：管理开销大（块数量太多）

常见默认值是 16 个 token 一块。

## 水位线（Watermark）

`__init__` 里有个 `watermark_blocks`：

```python
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
```

它是一条"安全线"——保留一部分块不分配给新请求，避免刚接纳一个请求马上就得抢占，减少抖动。类似内存管理里的保留区。

## 小结

- KV Cache = 模型生成时缓存的中间结果，很占显存
- PagedAttention = 把显存切成固定大小的块，按需分配、用完回收
- 核心方法：`allocate_slots()`（分配）、`free()`（回收）、`get_computed_blocks()`（前缀缓存复用）
- 这是 vLLM 高吞吐的根本原因：显存利用率从约 20% 提升到接近 100%
- 下一站：分配好块后，GPU 怎么用这些块跑模型（见 `05-Worker模型执行.md`）
