# 33 · MoE 混合专家与专家并行

> 对应目录：`vllm/model_executor/layers/fused_moe/`、`vllm/distributed/eplb/`

## 什么是 MoE

MoE（Mixture of Experts，混合专家）是一种模型结构。普通模型每个 token 都过同一个前馈网络（FFN）；MoE 把这个 FFN 换成**多个"专家"**（多个小 FFN），每个 token 只走其中几个。

像 DeepSeek-V3、Mixtral、Qwen-MoE 等都是 MoE 模型。

类比：普通模型是"全科医生"，每个问题都找同一个人；MoE 是"专科医院"，每个问题被路由到对口的几个专科医生。

## 为什么用 MoE

| 好处 | 说明 |
|------|------|
| 参数多但算得少 | 总参数量巨大（容量大），但每个 token 只激活一小部分专家（计算省） |
| 性价比高 | 用"激活一部分"的代价，获得"超大模型"的能力 |

比如一个 MoE 模型可能有 64 个专家，但每个 token 只激活 8 个——参数量是大模型，计算量却接近小模型。

## 核心机制：路由 (Routing)

每个 token 来了，一个"路由器（router/gate）"决定它该去哪几个专家：

```
token "苹果"
  ↓ router 打分
专家分数: [专家3: 0.8, 专家7: 0.6, 专家1: 0.1, ...]
  ↓ 选 top-k（比如 top-2）
送往: 专家3 和 专家7
  ↓ 两个专家各自计算
  ↓ 按分数加权合并结果
输出
```

代码在 `fused_moe/router/`（路由）+ `fused_moe/experts/`（专家计算）。

## 工程挑战：为什么有个 fused_moe

MoE 计算很"碎"：一批 token 被分散到不同专家，每个专家处理的 token 数还不一样。朴素实现会有大量小而零碎的运算，效率低。

`fused_moe`（融合 MoE）用专门优化的 kernel，把"分发 token → 各专家计算 → 合并结果"高效地融合处理，避免碎片化。这是 vLLM 跑 MoE 模型快的关键。

## 专家并行 (Expert Parallelism, EP)

MoE 模型专家多，可以把**不同专家放到不同 GPU** 上，这叫专家并行。

```
GPU0: 专家 0~15
GPU1: 专家 16~31
GPU2: 专家 32~47
GPU3: 专家 48~63
```

token 路由到哪个专家，就把它的数据发到那张卡。这需要卡间的 **all-to-all 通信**（每张卡都要把 token 发给其他卡上的目标专家），代码见 `fused_moe/all2all_utils.py`。

## 负载均衡问题与 EPLB

专家并行有个麻烦：**热门专家**会被更多 token 选中，导致它所在的 GPU 累死，其他 GPU 闲着。这叫负载不均。

`distributed/eplb/`（Expert Parallelism Load Balancer，专家并行负载均衡器）解决这个：动态调整专家在 GPU 上的分布，甚至把热门专家**复制多份**到不同卡，让负载更均匀。

```
问题: 专家3 太火 → GPU0 过载
解法(EPLB): 把专家3 复制到 GPU0 和 GPU1，分摊流量
```

## 和前面模块的关系

- MoE 是模型结构的一种，融入 `14-模型加载与定义.md` 的模型定义
- 专家并行是 `08-分布式与并行.md` 之外的第四种并行维度
- MoE 的 kernel 在 `csrc/moe/`（见 `21-自定义算子与CUDA内核.md`）

## 对新手的意义

- 用 MoE 模型（如 DeepSeek、Mixtral）时，vLLM 自动处理这些，无需手动配置
- 理解"MoE = 多专家 + 路由 + 只激活一部分"这个核心即可
- 多机跑超大 MoE 时才需要关心专家并行和 EPLB

## 小结

- MoE = 把 FFN 换成多个专家，每个 token 只走其中几个，参数大但算得少
- 核心是路由器：决定每个 token 去哪些专家，加权合并结果
- fused_moe 用融合 kernel 解决 MoE 计算碎片化、提速
- 专家并行把专家铺到多卡（需 all-to-all 通信），EPLB 解决负载不均
