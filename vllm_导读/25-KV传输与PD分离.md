# 25 · KV 传输与 PD 分离 (KV Transfer & Disaggregation)

> 对应目录：`vllm/distributed/kv_transfer/`、`vllm/v1/kv_offload/`

## 先理解 Prefill 和 Decode 的不同性格

回顾 `03-调度器.md`：一个请求分两个阶段。它俩对硬件的需求其实很不一样：

| 阶段 | 特点 | 瓶颈 |
|------|------|------|
| Prefill（处理 prompt） | 一次算很多 token，计算量大 | **算力**（compute-bound） |
| Decode（逐字生成） | 每次算 1 个 token，反复读 KV Cache | **访存/显存带宽**（memory-bound） |

把两个性格不同的活混在同一批 GPU 上跑，会互相干扰：一个长 prompt 的 prefill 会卡住其他请求的 decode，导致延迟抖动。

## PD 分离 (Prefill-Decode Disaggregation)

思路：**让 prefill 和 decode 在不同的 GPU 池子上跑**，各自优化。

```
                 ┌─────────────┐   KV Cache    ┌─────────────┐
请求 ──prompt──▶ │ Prefill 节点 │ ───传输────▶ │ Decode 节点  │ ──逐字生成──▶ 用户
                 │ (算力优化)   │              │ (访存优化)   │
                 └─────────────┘              └─────────────┘
```

- Prefill 节点：专注处理 prompt，算出 KV Cache
- 把 KV Cache 传给 Decode 节点
- Decode 节点：专注逐字生成

好处：两类任务不再互相干扰，延迟更稳定，各自的硬件配置可以单独优化。

## 关键问题：KV Cache 怎么传

Prefill 算出的 KV Cache 要快速、高效地传到 Decode 节点。这正是 `kv_transfer/` 的职责。它对接各种高速传输后端：

- **NIXL**：NVIDIA 的高速传输库（代码里有 `nixl_utils.py`）
- **共享存储 / RDMA / NVLink** 等

传输的是显存里的 KV Cache 块（见 `04`），数据量不小，所以传输效率是 PD 分离能否划算的关键。

## KV Connector：可插拔的传输接口

vLLM 把"KV 怎么传"抽象成 **KV Connector**，不同的传输方案实现统一接口。回顾 `03-调度器.md` 的代码里出现的 `self.connector`、`WAITING_FOR_REMOTE_KVS` 状态——就是调度器在配合 KV 传输：

```
请求状态: WAITING_FOR_REMOTE_KVS
含义: 这个请求在等远端（prefill 节点）把 KV Cache 传过来，传完才能开始 decode
```

## 另一个相关功能：KV Offload（KV 卸载）

`v1/kv_offload/` 做的是另一件事：当 GPU 显存不够时，把暂时用不到的 KV Cache **挪到 CPU 内存**（甚至硬盘），需要时再挪回来。

类比操作系统的 swap（内存换页到磁盘）。这样能服务更长的上下文或更多并发，代价是挪动数据的开销。

```
GPU 显存（快，小）  ⇄  CPU 内存（慢，大）  ⇄  磁盘（更慢，更大）
        热数据放 GPU，冷数据往下挪
```

## 这是进阶/生产特性

- 单机单卡、普通使用**完全不需要**关心这些
- 它们是**大规模生产部署**的优化：超大模型、超高并发、超长上下文、严格的延迟 SLA
- 属于 vLLM 较新、较前沿的方向

## 小结

- Prefill（吃算力）和 Decode（吃访存）性格不同，混跑会互相干扰
- PD 分离：让两者在不同 GPU 池子上跑，各自优化，延迟更稳
- 难点是 KV Cache 跨节点传输，靠可插拔的 KV Connector（NIXL 等）
- KV Offload：显存不够时把 KV 挪到 CPU/磁盘，类似系统 swap
- 都是面向大规模生产的进阶特性，普通使用无需关心
