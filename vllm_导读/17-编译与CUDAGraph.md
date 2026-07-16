# 17 · 编译优化与 CUDA Graph

> 对应目录：`vllm/compilation/`

## 要解决的问题

模型 forward 时，Python 要一条一条地向 GPU 发送指令（"算这个矩阵乘法"、"做这个加法"……）。每条指令都有 CPU→GPU 的沟通开销。在 decode 阶段（每步只算 1 个 token），GPU 计算本身很快，这些"发指令"的开销反而变得显眼。

类比：厨师（GPU）切菜很快，但如果每切一刀都要等经理（CPU）口头下一次指令，整体就被拖慢了。

## 两个层次的优化

### 1. torch.compile（图编译）

把模型的一连串运算"编译"成优化后的代码：

- 把多个小操作**融合**成一个大 kernel（减少 kernel 启动次数和访存）
- 提前规划好计算图，减少运行时的 Python 开销

vLLM 基于 PyTorch 的 `torch.compile` 机制，并加了自己的定制（`compilation/backends.py`、`compilation/decorators.py`）。

### 2. CUDA Graph（图捕获/重放）

更进一步：把整个 forward 的 GPU 指令序列"录制"成一张图，之后**一次性重放**，几乎消除 CPU 发指令的开销。

```
普通模式：CPU 逐条发指令 → GPU 执行（每条都有沟通开销）
CUDA Graph：录制一次 → 之后重放整张图（一次性提交，开销极小）
```

类比：与其每切一刀都口头下令，不如把"切菜全套流程"录成视频，厨师照着视频一气呵成。

## 为什么需要"录制"

CUDA Graph 有个前提：每次执行的操作序列和张量形状要**固定**。但实际请求 batch 大小是变化的。vLLM 的做法是：

- 预先为几种常见的 batch 大小（比如 1, 2, 4, 8, 16...）各录制一张图
- 运行时把实际 batch 补齐（padding）到最接近的那个尺寸，用对应的图

这就是代码里频繁出现的 `cudagraph_mode`、`batch_descriptor`、`capture` 等概念的由来。

## 代码结构

| 文件 | 作用 |
|------|------|
| `compilation/backends.py` | 编译后端，对接 torch.compile |
| `compilation/cuda_graph.py` | CUDA Graph 的捕获与重放 |
| `compilation/decorators.py` | 装饰器，标记哪些模块参与编译 |
| `compilation/piecewise_backend.py` | 分段编译（把模型切成可编译的片段） |
| `compilation/passes/` | 编译优化 pass（如算子融合规则） |

## 分段编译 (Piecewise)

完整模型整个编译有时不可行（比如 attention 部分要用特殊的 PagedAttention kernel，不适合被通用编译器处理）。vLLM 用"分段"策略：

```
模型 = [可编译片段] + [attention(特殊处理)] + [可编译片段] + ...
```

把能编译的部分编译，attention 等特殊算子单独走优化好的 kernel。这就是 `piecewise_backend.py` 和 `partition_rules.py` 在做的事。

## 何时生效

这些优化主要加速 **decode 阶段**（小 batch、形状固定、对延迟敏感）。prefill 阶段计算量大，发指令开销占比小，收益相对没那么明显。

## 怎么控制

通过编译配置控制优化等级：

```python
llm = LLM(model="...", compilation_config={"level": 3})
```

等级越高优化越激进，但启动（编译/捕获）时间也越长。

## 小结

- 问题：Python 逐条发 GPU 指令有开销，在 decode 阶段显眼
- torch.compile：融合算子、优化计算图
- CUDA Graph：录制整个指令序列，重放时几乎零开销
- 因 batch 大小变化，预先为多种尺寸录图 + padding 适配
- 分段编译：可编译部分编译，attention 等特殊算子单独优化
- 这是让"Python 写的引擎"依然极快的关键优化之一
