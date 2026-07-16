# 05 · Worker / 模型执行

> 对应目录：`vllm/v1/worker/`、`vllm/model_executor/`

## 这个模块负责什么

Worker 是真正在 GPU 上干活的部分：拿到调度器的安排，准备好输入数据，调用模型 forward，算出每个 token 的概率分布（logits）。

类比 Web 后端：这是真正执行业务逻辑的**工作线程**，只不过这里的"线程"是 GPU。

## 关键文件

| 文件 | 作用 |
|------|------|
| `v1/worker/gpu_worker.py` | GPU worker，管理设备、显存 |
| `v1/worker/gpu_model_runner.py` | 模型运行器，准备输入 + 跑 forward |
| `v1/worker/block_table.py` | 块表，记录每个请求的 KV Cache 块位置 |
| `model_executor/models/` | 各种模型的具体实现（Llama、Qwen 等） |
| `model_executor/layers/` | 模型的基础层（attention、linear 等） |

## 核心方法：execute_model()

`gpu_model_runner.py` 第 4056 行附近。这个方法很长，但骨架就三步：

```python
def execute_model(self, scheduler_output, ...):
    # ① 准备阶段（preprocess）
    self._update_states(scheduler_output)        # 更新批次状态：哪些请求进、哪些出
    input_ids, positions, ... = self._preprocess(...)   # 拼接 token、位置编码
    attn_metadata = self._build_attention_metadata(...) # 构建 attention 元数据（关键！）

    # ② 跑模型（forward）
    hidden_states = self._model_forward(
        input_ids=input_ids, positions=positions, ...
    )

    # ③ 算 logits（postprocess）
    logits = self.model.compute_logits(hidden_states[logits_indices])
    # logits 之后交给采样模块选 token（见 06-采样.md）
```

## 重点：attention metadata 与 block table

这是 PagedAttention 在 GPU 侧的落地点。

调度器分配好了每个请求的 KV Cache 块（散落在显存各处），但 GPU 上的 attention kernel 需要知道：**"这个请求的 KV 缓存到底在哪几个块里？"**

`_build_attention_metadata` 就负责把这个信息打包成元数据，里面最重要的是 **block table**（块表）——本质是个映射：

```
请求 A 的逻辑位置 0,1,2,3...  →  物理块 [块5, 块2, 块9, 块1...]
```

attention kernel 拿着 block table，就能从散落的物理块里正确读取每个请求的 KV，完成注意力计算。这正是"分页"思想在 GPU 计算里的体现。

## 模型 forward 里发生了什么

`_model_forward` 调用具体模型（如 Llama）的 forward。一个 Transformer 层大致是：

```
输入 token
  → Attention（注意力，用 PagedAttention kernel 读 KV Cache）
  → 残差连接 + LayerNorm
  → FFN / MoE（前馈网络）
  → 残差连接 + LayerNorm
  → 下一层...
最后 → hidden_states
```

这些层在 `model_executor/layers/` 里：

- `attention/`：注意力（最核心、最耗时）
- `linear.py`：线性层（支持张量并行，把大矩阵切到多卡）
- `layernorm.py`：归一化
- `fused_moe/`：混合专家（MoE 模型用）
- `quantization/`：量化（用低精度省显存、提速）

## 真正的计算在 C++/CUDA

Python 层的 `_model_forward` 最终会调到 C++/CUDA 写的 kernel（在 `csrc/` 目录）：

- `csrc/attention/`：PagedAttention 的 CUDA 实现
- `csrc/quantization/`：量化算子
- `csrc/moe/`：MoE 路由

Python 通过 `vllm/_custom_ops.py` 这个桥梁调用它们。

> 这就是为什么"Python 写的也不慢"：耗时的矩阵运算全在 CUDA kernel 里跑，Python 只是发号施令。类比 Java 的 JNI 调 native 方法。

## logits 是什么

模型 forward 的最终产物。对词表里的**每一个**可能的 token，给出一个分数（概率的前身）。

比如词表有 5 万个词，logits 就是 5 万个数字，分数最高的那些就是模型认为最可能的下一个词。

logits 接下来交给采样模块（见 `06-采样.md`）选出实际的下一个 token。

## CUDA Graph 优化

代码里频繁出现 `cudagraph`。简单理解：把一连串 GPU 操作"录制"成一个图，之后重放时省去逐条发射指令的开销，减少 CPU-GPU 之间的沟通延迟。这是又一个让 Python 开销变得无关紧要的优化。

## 小结

- Worker 在 GPU 上真正执行模型，三步：准备输入 → forward → 算 logits
- 关键是 block table：告诉 attention kernel 每个请求的 KV 散落在哪些物理块
- 模型层在 `model_executor/layers/`，真正的重计算在 `csrc/` 的 CUDA kernel
- 产物是 logits（每个候选 token 的分数）
- 下一站：怎么从 logits 选出下一个 token（见 `06-采样.md`）
