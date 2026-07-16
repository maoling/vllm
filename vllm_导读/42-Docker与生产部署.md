# 42 · Docker 与生产部署

> 对应目录：`docker/`、`.buildkite/image_build/`

## 这个模块负责什么

提供把 vLLM 打包成容器镜像、部署到生产环境的能力。这是从"本地能跑"到"线上稳定服务"的最后一公里。

类比 Web 后端：和你熟悉的 Dockerfile + K8s 部署完全一致，只是多了 GPU 相关的考量。

## Dockerfile 结构

`docker/` 下针对不同硬件有不同的 Dockerfile：

| 文件 | 用途 |
|------|------|
| `docker/Dockerfile` | 默认（NVIDIA CUDA），Python 3.12 |
| `docker/Dockerfile.cpu` | CPU 版 |
| `docker/Dockerfile.rocm` | AMD GPU |
| `docker/Dockerfile.xpu` | Intel GPU |
| `docker/Dockerfile.tpu` | Google TPU |

呼应 `20-平台抽象与硬件后端.md`——不同硬件不仅代码层适配，连镜像都分开构建。

## 多阶段构建

vLLM 的 Dockerfile 用**多阶段构建（multi-stage build）** 来控制镜像大小和构建效率：

```
阶段1 (builder)：装编译工具，编译 C++/CUDA 代码（见 21）→ 产出 wheel
阶段2 (runtime)：只装运行时依赖 + 上一阶段的 wheel → 最终镜像
```

好处：最终镜像不含庞大的编译工具链，体积更小。

## 怎么用官方镜像

最简单的部署方式：

```bash
docker run --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3-8B
```

要点：

- `--gpus all`：把 GPU 暴露给容器（需要 NVIDIA Container Toolkit）
- `-v ...huggingface`：挂载模型缓存，避免每次重新下载
- 容器内启动的就是 `vllm serve`（见 `23`、`29`）

## 生产部署的关键考量

| 方面 | 说明 |
|------|------|
| GPU 调度 | K8s 上用 device plugin 分配 GPU；多卡注意 NCCL 通信配置 |
| 模型加载 | 大模型加载慢，考虑模型缓存卷、预热 |
| 健康检查 | 用 `/health` 端点配 K8s liveness/readiness 探针 |
| 监控 | `/metrics` 接 Prometheus（见 `22`、`31`） |
| 安全 | 加鉴权网关、别裸暴露（见 `23` 的安全提示） |
| 扩缩容 | 按 GPU 利用率/队列长度做 HPA |
| 多机 | 超大模型用 Ray 跨节点（见 `30`） |

## 镜像构建 CI

`.buildkite/image_build/` 是自动构建镜像的 CI 脚本，为各硬件、各架构（x86/arm64）批量构建并发布镜像。这保证了官方镜像的一致性和及时更新。

## 启动慢怎么办

大模型容器启动可能要几分钟（下载/加载权重 + 编译/捕获 CUDA Graph）。生产上常用：

- 模型权重放共享存储或预拉到节点
- 配置编译缓存（见 `17`）
- 合理设置 K8s 探针的初始延迟，别让没就绪的 Pod 被误杀

## 对新手的意义

- 本地学习直接 `pip install` 或跑官方镜像即可
- 真要上线，按你已有的 Docker/K8s 经验来，重点关注 GPU 暴露、模型缓存、监控、鉴权
- vLLM 的部署模式和普通 Web 服务高度相似，你的后端经验直接复用

## 小结

- `docker/` 提供各硬件的 Dockerfile，多阶段构建控制镜像体积
- 官方镜像 `vllm/vllm-openai`，`docker run --gpus all` 即可起服务
- 生产考量：GPU 调度、模型缓存、健康检查、监控、鉴权、扩缩容、多机
- 部署模式和普通 Web 服务一致，后端经验可直接迁移
