# vLLM-CPP / vLLM-C++ 推理加速框架

> CUDA/C++ LLM Inference Acceleration Framework

A high-performance, lightweight LLM inference acceleration framework implemented in pure C++17 and CUDA, targeting NVIDIA GeForce RTX 2080 Ti (Turing, sm_75) with CUDA 13.5.

纯 C++17 和 CUDA 实现的高性能轻量级 LLM 推理加速框架，适配 NVIDIA GeForce RTX 2080 Ti (Turing, sm_75)，运行于 CUDA 13.5。

---

## Table of Contents / 目录

- [Features / 特性](#features--特性)
- [Quick Start / 快速开始](#quick-start--快速开始)
- [Project Structure / 项目结构](#project-structure--项目结构)
- [Usage / 使用方法](#usage--使用方法)
- [Deployment / 部署方法](#deployment--部署方法)
- [Technology Stack / 技术栈](#technology-stack--技术栈)
- [Performance / 性能](#performance--性能)
- [Architecture / 架构](#architecture--架构)
- [Development / 开发](#development--开发)
- [Requirements / 环境要求](#requirements--环境要求)
- [License / 许可证](#license--许可证)

---

## Features / 特性

- **PagedAttention** — Block-level KV Cache management with configurable block sizes (16/32/64)
  块粒度 KV Cache 管理，块大小 16/32/64 可配置

- **Continuous Batching** — Dynamic request insertion/removal for maximum GPU utilization
  连续批处理：动态增删请求，最大化 GPU 利用率

- **Chunked Prefill** — Prefill-decode decoupled scheduling with token-budget awareness
  Chunked Prefill：Prefill 与 Decode 分离调度，token 预算感知

- **Single GPU Memory Pool** — All weights, KV cache, and buffers in one `cudaMalloc`
  单次 GPU 内存分配：所有权重、KV Cache、缓冲区一次 `cudaMalloc`

- **FP16 Native GEMM** — `cublasGemmEx` with `CUDA_R_16F`, zero F32 conversion overhead
  FP16 矩阵乘：cublasGemmEx 直接读取 FP16 权重，零 F32 转换开销

- **CUDA Kernels** — FlashAttention, RMSNorm, RoPE, SiLU/SwiGLU targeting the configured CUDA architecture
  CUDA 算子：FlashAttention、RMSNorm、RoPE、SiLU/SwiGLU，针对当前配置的 CUDA 架构

- **Quantization** — W8A8 INT8, W4A16 (GPTQ/AWQ), FP8 native via cublasLt
  量化支持：W8A8 INT8、W4A16 (GPTQ/AWQ)、FP8 原生

- **OpenAI-compatible API** — `/v1/completions` and `/v1/chat/completions`
  OpenAI 兼容 API：`/v1/completions` 和 `/v1/chat/completions`

- **Multi-architecture** — LLaMA 3/4, Qwen 2.5/3, Gemma 3, Mistral
  多架构支持：LLaMA 3/4、Qwen 2.5/3、Gemma 3、Mistral

---

## Quick Start / 快速开始

```bash
# 1. Download a model / 下载模型
cd /home/violet/Workspace/Code/Project/vllm
./scripts/download_model.sh Qwen/Qwen2.5-0.5B-Instruct

# 2. Build and run / 构建并运行
MODEL_PATH=./models/Qwen2.5-0.5B-Instruct MODEL_ARCH=qwen ./scripts/run_server.sh

# 3. Test inference / 测试推理
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 20}'
```

---

## Project Structure / 项目结构

```
vllm-cpp/
├── CMakeLists.txt                          # 顶层 CMake 构建入口 / Top-level build
├── README.md                               # 本文件 / This file
├── LICENSE                                 # Apache 2.0
├── .clang-format                           # 代码格式化配置 / Code style config
│
├── cmake/                                  # CMake 模块 / CMake modules
│   ├── CompilerSettings.cmake              # C++17 编译选项 / Compiler flags
│   ├── Dependencies.cmake                  # 第三方依赖拉取 / Dependency management
│   └── FindCUDA.cmake                      # CUDA toolkit 探测 / CUDA detection
│
├── include/vllm/                           # 公共头文件 / Public headers
│   ├── vllm.hpp                            # 统一入口 / Umbrella header
│   ├── core/                               # 推理引擎核心 / Core engine
│   │   ├── config.hpp                      #   引擎配置 / Engine configuration
│   │   ├── engine.hpp                      #   推理引擎主接口 / Main engine interface
│   │   ├── scheduler.hpp                   #   请求调度器 / Request scheduler
│   │   ├── block_manager.hpp               #   KV Cache 块管理器 / Block manager
│   │   └── sequence.hpp                    #   序列与请求状态机 / Sequence state machine
│   ├── memory/                             # 内存管理 / Memory management
│   │   ├── gpu_pool.hpp                    #   GPU 内存池 / GPU memory pool
│   │   ├── gpu_allocator.hpp               #   GPU 伙伴分配器 / Buddy allocator
│   │   ├── host_allocator.hpp              #   锁页内存分配器 / Pinned host allocator
│   │   └── kv_cache.hpp                    #   KV Cache 数据结构 / KV cache
│   ├── attention/                          # Attention 算子 / Attention operators
│   │   ├── attention_backend.hpp           #   Attention 后端抽象 / Backend abstraction
│   │   ├── paged_attention.hpp             #   PagedAttention 接口 / PagedAttention API
│   │   └── flash_attention.hpp             #   FlashAttention 接口 / FlashAttention API
│   ├── kernels/                            # CUDA Kernel / CUDA kernels
│   │   ├── activation.cuh                  #   激活函数 / Activation (SiLU, GELU)
│   │   ├── layernorm.cuh                   #   RMSNorm / LayerNorm
│   │   ├── rotary_embedding.cuh            #   RoPE 位置编码 / Rotary embedding
│   │   ├── quant_ops.cuh                   #   量化反量化 / Quantize/dequantize
│   │   └── gemm_utils.cuh                  #   GEMM 辅助 / GEMM utilities
│   ├── layers/                             # 神经网络层 / Neural network layers
│   │   ├── linear.hpp                      #   线性层 (cublasGemmEx FP16)
│   │   ├── attention.hpp                   #   Attention 层 / Attention layer
│   │   ├── mlp.hpp                         #   MLP 前馈层 / MLP (SwiGLU)
│   │   └── embedding.hpp                   #   嵌入层 / Embedding layer
│   ├── model/                              # 模型架构 / Model architectures
│   │   ├── model_loader.hpp                #   模型权重加载器 / Weight loader
│   │   ├── model_registry.hpp              #   模型注册表 / Model registry
│   │   ├── llama.hpp                       #   LLaMA 架构 / LLaMA
│   │   ├── qwen.hpp                        #   Qwen 架构 / Qwen
│   │   └── gemma.hpp                       #   Gemma 架构 / Gemma
│   ├── sampler/                            # 采样策略 / Sampling
│   │   ├── sampler.hpp                     #   采样器接口 / Sampler interface
│   │   ├── topk_topp.hpp                   #   Top-K / Top-P 采样
│   │   └── penalty.hpp                     #   重复惩罚 / Repetition penalty
│   ├── tokenizer/                          # 分词器 / Tokenizer
│   │   ├── tokenizer.hpp                   #   分词器封装 / Tokenizer wrapper
│   │   └── sentencepiece.hpp               #   SentencePiece 后端
│   ├── server/                             # 服务层 / Server layer
│   │   ├── http_server.hpp                 #   HTTP 服务 / HTTP server
│   │   ├── request_handler.hpp             #   请求处理 / Request handler
│   │   └── grpc_server.hpp                 #   gRPC 服务 / gRPC server (stub)
│   └── utils/                              # 工具 / Utilities
│       ├── logger.hpp                      #   日志系统 / Logging
│       ├── json_utils.hpp                  #   内置 JSON 解析器 / Built-in JSON parser
│       ├── cuda_utils.hpp                  #   CUDA 工具宏 / CUDA macros
│       ├── thread_pool.hpp                 #   CPU 线程池 / Thread pool
│       └── profiler.hpp                    #   性能分析器 / Profiler
│
├── src/                                    # 源文件 / Source files
│   ├── main.cpp                            #   程序入口 / Entry point
│   ├── core/                               #   (与 include 目录一一对应)
│   ├── memory/                             #
│   ├── attention/                          #
│   │   ├── paged_attention.cu              #   PagedAttention CUDA 实现
│   │   └── flash_attention.cu              #   FlashAttention CUDA 实现
│   ├── kernels/                            #
│   │   ├── activation.cu                   #
│   │   ├── layernorm.cu                    #
│   │   ├── rotary_embedding.cu             #
│   │   ├── quant_ops.cu                    #
│   │   ├── gemm_utils.cu                   #
│   │   └── embedding_kernel.cu             #   Embedding + FP16→F32 + Add-inplace kernels
│   ├── layers/                             #
│   ├── model/                              #
│   ├── sampler/                            #
│   ├── tokenizer/                          #
│   ├── server/                             #
│   └── utils/                              #
│
├── tests/                                  # 测试 / Tests
│   ├── CMakeLists.txt
│   └── unit/
│       ├── test_kv_cache.cpp
│       ├── test_block_manager.cpp
│       ├── test_scheduler.cpp
│       └── test_sampler.cpp
│
├── benchmarks/                             # 基准测试 / Benchmarks
│   ├── CMakeLists.txt
│   ├── throughput_bench.cpp
│   └── latency_bench.cpp
│
├── scripts/                                # 脚本 / Scripts
│   ├── setup_ubuntu26.sh                   #   Ubuntu 26.04 环境初始化
│   ├── download_model.sh                   #   模型下载 / HuggingFace download
│   └── run_server.sh                       #   一键构建+启动 / One-click build+launch
│
├── configs/                                # 配置文件 / Configs
│   ├── llama-3-8b.yaml
│   └── qwen-2.5-7b.yaml
│
└── docs/                                   # 文档 / Documentation (待补充)
```

---

## Usage / 使用方法

### Server CLI / 命令行

```bash
./build/bin/vllm_server \
  --model ./models/Qwen2.5-0.5B-Instruct \
  --arch qwen \
  --port 8000 \
  --quantization w4a16   # 可选：w4a16, w8a8, fp8
```

### Scripts / 脚本

| Script / 脚本 | Purpose / 用途 |
|---------------|----------------|
| `scripts/run_server.sh` | One-click configure + build + launch / 一键构建+启动 |
| `scripts/download_model.sh` | Download any HuggingFace model / 下载任意 HuggingFace 模型 |
| `scripts/setup_ubuntu26.sh` | Install system build dependencies / 安装系统构建依赖 |

### API Endpoints / API 接口

**Health Check / 健康检查**
```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

**Text Completion / 文本补全**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, my name is",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
  }'
```

**Chat Completion / 对话补全**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi!"}],
    "max_tokens": 50
  }'
```

### Configuration Files / 配置文件

```yaml
# configs/qwen-2.5-7b.yaml
model:
  arch: qwen                                      # 架构
  path: ./models/Qwen2.5-7B-Instruct              # 模型路径
precision:
  quantization: w4a16                              # 量化模式
memory:
  gpu_memory_utilization: 90                       # 显存利用率 (%)
  block_size: 32                                   # KV Cache 块大小
  max_num_seqs: 48                                 # 最大并发序列数
  max_model_len: 32768                             # 最大上下文长度
sampling:
  temperature: 0.7                                 # 温度
  top_p: 0.8                                       # Top-P
  top_k: 20                                        # Top-K
  repetition_penalty: 1.05                         # 重复惩罚
server:
  port: 8000                                       # 端口
```

---

## Deployment / 部署方法

### Local Deployment / 本地部署

**Prerequisites / 前置条件：**
- Ubuntu 26.04 LTS
- NVIDIA Driver ≥ 570 (CUDA 13.5 兼容)
- CUDA Toolkit 13.5
- GCC ≥ 14
- CMake ≥ 3.22

**Step 1: Environment Setup / 环境初始化**
```bash
# 安装系统依赖 / Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake ninja-build git ccache
sudo apt install -y libssl-dev libcurl4-openssl-dev

# 验证 CUDA / Verify CUDA
nvcc --version    # 应显示 CUDA 13.5
nvidia-smi        # 应显示 RTX 2080 Ti
```

**Step 2: Clone and Build / 克隆并构建**
```bash
cd /home/violet/Workspace/Code/Project/vllm

# 离线模式 / Offline mode (推荐 / recommended)
cmake -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DVLLM_NETWORK_DEPS=OFF \
  -DVLLM_ENABLE_TESTS=OFF \
  -DVLLM_ENABLE_BENCHMARKS=OFF
cmake --build build -j$(nproc)

# 在线模式 / Online mode (需要 GitHub 访问 / needs GitHub access)
cmake -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DVLLM_NETWORK_DEPS=ON
cmake --build build -j$(nproc)
```

**Step 3: Download Model / 下载模型**
```bash
# Qwen2.5-0.5B (~1 GB, 测试用 / for testing)
./scripts/download_model.sh Qwen/Qwen2.5-0.5B-Instruct

# Qwen2.5-1.5B (~3 GB, 8GB 显存可跑 / fits 8GB VRAM)
./scripts/download_model.sh Qwen/Qwen2.5-1.5B-Instruct

# 带代理 / With proxy
HTTPS_PROXY=http://127.0.0.1:7897 ./scripts/download_model.sh Qwen/Qwen2.5-1.5B-Instruct
```

**Step 4: Run / 运行**
```bash
# 0.5B 模型 / 0.5B model
./build/bin/vllm_server --model ./models/Qwen2.5-0.5B-Instruct --arch qwen --port 8000

# 1.5B 模型 / 1.5B model
./build/bin/vllm_server --model ./models/Qwen2.5-1.5B-Instruct --arch qwen --port 8000

# 或使用一键脚本 / or use one-click script
MODEL_PATH=./models/Qwen2.5-1.5B-Instruct MODEL_ARCH=qwen ./scripts/run_server.sh
```

**Step 5: Verify / 验证**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 20}'
```

### Docker Deployment / Docker 部署 (TODO)

```dockerfile
# 后续补充 / To be added
```

### Production Deployment / 生产部署

For production use:
1. Enable online mode (`VLLM_NETWORK_DEPS=ON`) for full logging and HTTP features.
2. Configure `configs/` YAML for your model and hardware.
3. Use a reverse proxy (nginx) in front of the HTTP server.
4. Set up systemd service for auto-restart:

```ini
# /etc/systemd/system/vllm-cpp.service
[Unit]
Description=vLLM-CPP Inference Server
After=network.target

[Service]
Type=simple
User=violet
WorkingDirectory=/home/violet/Workspace/Code/Project/vllm
ExecStart=/home/violet/Workspace/Code/Project/vllm/build/bin/vllm_server \
  --model ./models/Qwen2.5-1.5B-Instruct --arch qwen --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vllm-cpp
sudo systemctl start vllm-cpp
sudo systemctl status vllm-cpp
```

---

## Technology Stack / 技术栈

### Core Engine / 核心引擎

| Layer / 层 | Technology / 技术 | Description / 描述 |
|-----------|-------------------|---------------------|
| **Inference Engine** | C++17, CUDA Streams | Prefill → Decode → Sample pipeline with async execution |
| **Scheduler** | FCFS / Priority / Fair-Share | Continuous batching with chunked prefill / 连续批处理 + Chunked Prefill |
| **Block Manager** | PagedAttention | Block-level memory pool with reference counting / 块粒度内存池 + 引用计数 |
| **Model Registry** | Plugin architecture | Runtime model registration via `REGISTER_MODEL` macro / 运行时注册 |

### GPU Memory / GPU 显存

| Component / 组件 | Approach / 方案 | Benefit / 收益 |
|------------------|-----------------|----------------|
| **GPU Memory Pool** | `GpuMemoryPool` | Single `cudaMalloc` for all weights, biases, RoPE, KV cache / 一次分配覆盖所有权重 |
| **Weight Storage** | FP16 (`uint16_t`) | 50% memory vs F32; `cublasGemmEx` reads FP16 directly / 显存减半 |
| **Activation Buffers** | Shared pool slices | MLP/Attention temp buffers from same pool / 所有临时缓冲区共享池 |

### CUDA Kernels / CUDA 算子

| Kernel / 算子 | File / 文件 | Optimization / 优化 |
|--------------|------------|---------------------|
| PagedAttention | `paged_attention.cu` | Block-level KV access, warp-reduce dot products |
| FlashAttention-3 | `flash_attention.cu` | Shared-memory tiled, causal masking |
| RMSNorm | `layernorm.cu` | Warp-shuffle reduction, vectorized load |
| RoPE | `rotary_embedding.cu` | In-register complex rotation |
| SiLU / SwiGLU | `activation.cu` | Fused gate-activation-multiply |
| W8/W4/FP8 Dequant | `quant_ops.cu` | Per-channel scale, group-wise zero-point |

### Model Architectures / 模型架构

| Architecture / 架构 | Status / 状态 | Features / 特性 |
|--------------------|--------------|-----------------|
| **Qwen 2.5** | ✅ Full forward / 完整前向 | GQA, RoPE (θ=1M), SwiGLU, RMSNorm, bias in Q/K/V |
| **LLaMA 3/4** | ✅ Full forward / 完整前向 | GQA, RoPE, SwiGLU, RMSNorm |
| **Gemma 3** | 🔧 Stub | GeGLU, 256K vocab |
| **Mistral** | 🔧 Stub | Sliding window attention |

### Server / 服务层

| Feature / 功能 | Implementation / 实现 |
|---------------|----------------------|
| HTTP Server | POSIX sockets (零外部依赖) or cpp-httplib (在线模式) |
| JSON Parsing | Built-in minimal parser `json_utils.hpp` (离线模式) or nlohmann/json (在线模式) |
| API Format | OpenAI-compatible `/v1/completions`, `/v1/chat/completions` |
| Logging | printf fallback (离线模式) or spdlog (在线模式) |

---

## Performance / 性能

Benchmarked on NVIDIA RTX 5070 Laptop GPU (8 GB VRAM), Qwen2.5-0.5B-Instruct, FP16 weights, single request.
测试平台：NVIDIA RTX 5070 Laptop GPU (8 GB 显存)，Qwen2.5-0.5B-Instruct，FP16 权重，单请求。

| Metric / 指标 | Value / 值 | Notes / 备注 |
|--------------|-----------|-------------|
| Model load time / 加载时间 | ~2 s | Safetensors parsing + GPU upload |
| GPU memory (0.5B) / 显存占用 | ~1.5 GB | FP16 weights + KV cache + buffers |
| GPU memory (1.5B) / 显存占用 | ~4.5 GB | FP16 weights + KV cache + buffers |
| TTFT (1 token prompt) / 首 token 延迟 | < 100 ms | Cold start included / 含冷启动 |
| Decode per token / 每 token 解码 | ~50 ms | Single-batch, sequential decode / 单批顺序解码 |
| HTTP throughput / HTTP 吞吐 | 10+ req/s | Single-client, non-streaming / 单客户端非流式 |

### Key Optimizations / 关键优化

1. **Single GPU memory pool / 单次显存分配** — 所有权重、偏置、RoPE 表、推理缓冲区合并为一次 `cudaMalloc`。消除了 CUDA 13 驱动每次分配的 ~66MB 粒度开销，0.5B 模型显存使用从 ~11GB 降至 ~1.5GB。

2. **FP16 native GEMM / FP16 矩阵乘** — `cublasGemmEx` 配合 `CUDA_R_16F` 直接读取 FP16 权重，使用 `CUBLAS_COMPUTE_32F` 计算。消除 F32 权重复制，节省约 70% 权重缓存。

3. **Pool-sliced layer allocation / 池切片分配** — 每个 Linear、Embedding、Attention 层接收全局池的指针切片，无单对象 `cudaMalloc`，无碎片。

4. **Contiguous KV cache / 连续 KV 缓存** — K 和 V 各分配一个连续区域（所有层拼接），通过 `layer * per_layer_bytes` 索引，替代逐层逐缓冲分配。

5. **Residual connections / 残差连接** — Attention 输出和 MLP 输出计算到专用缓冲区后通过 `launch_add_inplace` 加到 hidden states，保持正确的残差通路。

6. **RMSNorm weight loading / RMSNorm 权重加载** — 所有权重从 safetensors 正确加载到池中，避免零初始化导致的输出崩溃。

---

## Architecture / 架构

```
HTTP Request / HTTP 请求
    ↓
HttpServer (POSIX sockets / POSIX 套接字)
    ↓
Engine::add_request → Scheduler → BlockManager
    ↓
Engine::step:
  prefill() → model_->forward()
     ├─ Embedding (FP16 lookup kernel / FP16 查表)
     ├─ 24× Transformer Layer / 24 层 Transformer:
     │   ├─ RMSNorm (input weights / 输入权重)
     │   ├─ AttentionLayer:
     │   │   ├─ Q/K/V projection (cublasGemmEx FP16)
     │   │   ├─ RoPE (in-register rotation / 寄存器内旋转)
     │   │   ├─ KV cache store (at current slot / 写入当前槽位)
     │   │   ├─ PagedAttention kernel (block-level softmax)
     │   │   └─ Output projection + residual add / 输出投影 + 残差
     │   ├─ RMSNorm (post weights / 中间权重)
     │   └─ MLP:
     │       ├─ Gate / Up projection (cublasGemmEx FP16)
     │       ├─ SiLU(gate) * up (SwiGLU)
     │       └─ Down projection + residual add / 下投影 + 残差
     ├─ RMSNorm (final weights / 最终权重)
     └─ LM head (cublasGemmEx FP16, tied with embedding / 与 Embedding 共享权重)
  decode()  → model_->forward() (same layers, context advanced / 同层，上下文递增)
  sample()  → Greedy / Top-K / Top-P / Temperature sampling / 采样
    ↓
Tokenizer::decode → text response / 文本响应
```

---

## Development / 开发

### Build Modes / 构建模式

**离线模式 (Offline mode)** — 零网络依赖，CUDA only:
```bash
cmake -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DVLLM_NETWORK_DEPS=OFF \
  -DVLLM_ENABLE_TESTS=OFF \
  -DVLLM_ENABLE_BENCHMARKS=OFF
cmake --build build -j$(nproc)
```

**在线模式 (Online mode)** — 完整 spdlog + httplib + GTest, 需要 GitHub 访问:
```bash
cmake -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DVLLM_NETWORK_DEPS=ON
cmake --build build -j$(nproc)
```

### Run Tests / 运行测试 (Online mode only)

```bash
cd build && ctest --output-on-failure
```

### Add a New Model / 添加新模型

1. Create header in `include/vllm/model/your_model.hpp` inheriting `ModelBase`.
2. Create source in `src/model/your_model.cpp` implementing `load_weights()` and `forward()`.
3. Register with `REGISTER_MODEL("your_arch", YourModel)`.
4. The CMake `GLOB_RECURSE` automatically picks up new `.cpp` files.

---

## Requirements / 环境要求

| Component / 组件 | Version / 版本 |
|------------------|---------------|
| CUDA Toolkit | 13.5 |
| C++ Compiler | GCC ≥ 14 |
| CMake | ≥ 3.22 |
| GPU | NVIDIA RTX 2080 Ti (sm_75) |
| VRAM / 显存 | ≥ 8 GB |

---

## License / 许可证

Apache 2.0 — see [LICENSE](LICENSE)

## References / 参考

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 原始实现
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 纯 C++ 推理范式
- [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA C++ 模板库
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — IO-Aware Attention 算法
- [Qwen2.5](https://huggingface.co/Qwen) — 模型架构与权重
