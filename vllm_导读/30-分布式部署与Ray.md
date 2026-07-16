# 30 · 分布式部署与 Ray

> 对应目录：`vllm/ray/`、`vllm/v1/executor/`

## 区分两个概念：并行 vs 部署

- `08-分布式与并行.md` 讲的是**算法层面**：模型怎么拆到多卡（张量/流水线/数据并行）
- 本篇讲**工程层面**：这些分布在多卡多机上的进程，靠什么管理、协调、通信

类比：08 是"怎么把活分给多个工人"，30 是"怎么招募、调度、协调这些工人"。

## Executor：执行器抽象

回顾 `02-引擎.md`，`EngineCore` 里有个 `self.model_executor`。**Executor** 就是"管理一组 worker 进程"的抽象。引擎只管对 executor 说"执行这批请求"，executor 负责把命令分发到所有 worker、收集结果。

`v1/executor/` 下有不同的 executor 实现：

| Executor | 用途 |
|----------|------|
| 单进程 executor | 单卡，最简单 |
| 多进程 executor | 单机多卡（用 Python multiprocessing） |
| Ray executor | 多机多卡（用 Ray 框架） |

引擎代码不变，换 executor 就能从单卡扩展到多机集群——又是策略模式。

## 为什么需要 Ray

单机多卡用 Python 多进程就够了。但**跨多台机器**时，需要一个分布式框架来管理"哪台机器、哪个进程、怎么通信、怎么容错"。[Ray](https://www.ray.io/) 就是干这个的——一个流行的分布式计算框架。

vLLM 用 Ray 来：

- 在多台机器上启动和管理 worker 进程
- 分配 GPU 资源（哪个 worker 用哪台机器的哪块卡）
- 进程间协调

```
                ┌──────── 机器 A ────────┐   ┌──────── 机器 B ────────┐
Ray 集群管理 → │ worker0(GPU0) worker1(GPU1)│   │ worker2(GPU0) worker3(GPU1)│
                └────────────────────────┘   └────────────────────────┘
                         ↑ vLLM 引擎通过 Ray executor 统一调度
```

## 代码结构

| 目录/文件 | 作用 |
|-----------|------|
| `ray/` | Ray 相关的封装（启动、资源分配等） |
| `v1/executor/` | 各种 executor 实现 |

## 注意：进程内的计算通信还是 NCCL

容易混淆：Ray 负责"管理进程和资源"（控制面），但 GPU 之间真正传张量数据（如张量并行的 all-reduce）走的还是 NCCL 等高速通信库（见 `08`），**不是** Ray。Ray 管"编排"，NCCL 管"数据搬运"。

## 怎么用（多机示例概念）

多机部署大致是：

1. 在所有机器上启动 Ray，组成一个 Ray 集群
2. 在头节点用 `vllm serve` 并指定并行规模（比如 `--tensor-parallel-size 8 --pipeline-parallel-size 2`）
3. vLLM 通过 Ray 把 worker 铺到各机器的 GPU 上

## 对新手的意义

- 单机使用完全不用碰 Ray，vLLM 自动用多进程 executor
- 只有跨机器部署超大模型时才需要 Ray
- 理解"引擎 → executor → workers"这条链，就明白 vLLM 怎么从单卡扩展到集群

## 小结

- 并行（08）是算法层拆分，部署（30）是工程层的进程管理与协调
- Executor 抽象管理一组 worker，换实现即可从单卡扩到多机
- 多机部署用 Ray 框架管理进程和资源
- Ray 管编排（控制面），NCCL 管 GPU 间数据传输（数据面）
