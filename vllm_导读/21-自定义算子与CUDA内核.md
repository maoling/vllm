# 21 · 自定义算子与 CUDA 内核 (Custom Ops & Kernels)

> 对应目录：`csrc/`、`vllm/_custom_ops.py`、`CMakeLists.txt`

## 这个模块负责什么

这是 vLLM 真正"高性能"的部分：用 C++/CUDA 手写的计算内核（kernel），处理那些 PyTorch 自带算子不够快、或干脆没有的运算。

类比 Java 后端：当 Java 跑得不够快时用 JNI 调 native C 代码。这里 Python 调的是 CUDA。

## 为什么需要手写 kernel

PyTorch 提供了通用算子（矩阵乘法、加法等），但 vLLM 有些需求是它满足不了的：

- **PagedAttention**：要从散落的内存块里读 KV（见 `04`、`05`），PyTorch 没有现成的
- **算子融合**：把多个操作合成一个 kernel，减少访存（比如 RMSNorm + 量化一起做）
- **低精度计算**：INT4/FP8 的高效矩阵乘法（见 `09`）

手写 CUDA 能榨干 GPU 性能。

## 代码结构

```
csrc/
├── attention/          ← PagedAttention 内核（最核心）
├── quantization/       ← 量化矩阵乘法等
├── moe/                ← MoE 专家路由
├── cache.h             ← KV Cache 操作（如块拷贝）
├── custom_all_reduce.cuh ← 优化的多卡通信
├── torch_bindings.cpp  ← 关键！把 C++ 函数注册给 PyTorch/Python
└── ops.h               ← 算子声明
```

## Python 和 C++ 怎么连起来

三步桥接：

```
① C++/CUDA 实现 kernel（csrc/attention/*.cu）
       ↓
② torch_bindings.cpp 用 PyTorch 的机制注册这些函数
       ↓ 编译成 .so 动态库（vllm/_C.abi3.so）
③ Python 端 vllm/_custom_ops.py 包装成易用的函数
```

之后 Python 代码就能像调普通函数一样调用 CUDA kernel：

```python
from vllm import _custom_ops as ops
ops.paged_attention_v1(out, query, key_cache, value_cache, ...)  # 实际跑 CUDA
```

## 注册机制（PyTorch Custom Op）

`torch_bindings.cpp` 用 PyTorch 的 `TORCH_LIBRARY` 宏注册算子。每个自定义算子需要：

1. **schema**：声明函数签名（输入输出类型）
2. **实现**：CUDA 的具体计算
3. **meta-function**：告诉 PyTorch 输出的形状（用于编译/图优化，见 `17`）

贡献指南（`CONTRIBUTING.md`）特别强调：改 kernel 要更新 schema、要用 `torch.library.opcheck()` 测试。

## 编译

这些 C++/CUDA 代码通过 `CMakeLists.txt` + `setup.py` 编译。这也是为什么：

- 只改 Python：用 `VLLM_USE_PRECOMPILED=1` 装预编译版本（跳过编译，秒装）
- 改了 C++/CUDA：必须重新编译（慢，几分钟到几十分钟）

（参考 `AGENTS.md` 的安装说明。）

## 一个具体例子：PagedAttention kernel

它要做的事（呼应 `05-Worker模型执行.md`）：

```
输入：query + block table（KV 散落在哪些物理块）
  ↓ CUDA kernel：
  · 根据 block table 找到每个请求的 KV 块
  · 从这些不连续的块里读 K、V
  · 计算注意力分数、加权求和
输出：注意力结果
```

正是这个 kernel 让"分页的 KV Cache"能被高效计算，是 vLLM 核心创新的底层落地。

## 对新手的意义

- 这是最硬核的部分，需要 CUDA 编程知识，门槛较高
- 不写 kernel 也能理解 vLLM：知道"Python 调 CUDA 做重活"这个分工就够了
- 想深入 GPU 编程的，`csrc/attention/` 是最值得读的起点

## 小结

- `csrc/` 是 C++/CUDA 手写内核，承担最耗时的计算
- 桥接三步：CUDA 实现 → torch_bindings 注册 → `_custom_ops.py` 包装
- PagedAttention、量化、MoE、通信等关键性能点都在这
- 这是"Python 框架也能极快"的根本：重活全在编译好的 native kernel 里
