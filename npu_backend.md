# NPU Backend Design for llama.cpp

- [背景与目标](#背景与目标)
- [范围与非目标](#范围与非目标)
- [现有架构梳理](#现有架构梳理)
  - [ggml 后端生命周期](#ggml-后端生命周期)
  - [llama.cpp 与 ggml 集成](#llamacpp-与-ggml-集成)
  - [构建系统与可选组件](#构建系统与可选组件)
- [NPU 后端总体设计](#npu-后端总体设计)
  - [设备发现与上下文管理](#设备发现与上下文管理)
  - [缓冲区与内存策略](#缓冲区与内存策略)
  - [计算图调度与回退策略](#计算图调度与回退策略)
  - [算子实现优先级](#算子实现优先级)
  - [数据类型与量化支持](#数据类型与量化支持)
  - [多设备与异步执行](#多设备与异步执行)
- [构建与配置集成](#构建与配置集成)
- [llama.cpp 运行时对接](#llamacpp-运行时对接)
- [分阶段里程碑](#分阶段里程碑)
- [测试与验证](#测试与验证)
- [性能分析与调优路径](#性能分析与调优路径)
- [风险与缓解](#风险与缓解)
- [开放问题](#开放问题)
- [知识沉淀与后续工作](#知识沉淀与后续工作)

## 背景与目标

- **现状**：`llama.cpp` 通过 `ggml` 提供可插拔后端机制，已覆盖 CPU、CUDA、Metal、SYCL、CANN 等多种硬件。新硬件生态需要对齐该机制，才能让核心推理路径复用。
- **目标**：面向自研 NPU，提供完整的 `ggml` 后端实现，支持主流大语言模型在该硬件上运行，并与现有 CLI/服务端工具链无缝集成。
- **驱动力**：提升模型推理吞吐、降低功耗成本，同时保持与上游同步的可维护性，便于后续贡献或内部演进。

## 范围与非目标

- **必须交付**：
  - 端到端推理链路（加载权重、KV cache、推理循环）在 NPU 上正确运行；
  - 核心算子（矩阵乘、RMSNorm、Softmax/Rope、量化反量化）在 NPU 上实现或调度；
  - CMake/构建脚本、运行参数、文档齐备，支持开发与 CI。
- **可选增强**：
  - 多设备切分（张量并行 / pipeline）；
  - 自定义监控、Profiling 适配工具。
- **暂不覆盖**：
  - 训练/微调场景；
  - 量化新格式定义（复用现有 GGUF）；
  - 与第三方框架的直接互操作（如 Torch 拓展）。

## 现有架构梳理

### ggml 后端生命周期

`ggml/include/ggml-backend.h` 与 `ggml/src/ggml-backend-impl.h` 定义了后端需实现的接口族：

- **缓冲区类型（buffer type）**：负责张量存储的分配、对齐、最大容量查询，可区分权重存储与计算工作区。
- **缓冲区实例（buffer）**：封装内存生命周期、数据搬运、初始化钩子、清零操作等。
- **后端实例（backend/stream）**：对应一次推理会话的执行上下文，提供图调度、异步同步、事件机制。
- **设备抽象（device）**：描述硬件属性、内存规模、支持的算子/缓冲类型，并提供 `init_backend` 入口。
- **注册表（registry）**：在 `ggml/src/ggml-backend-reg.cpp` 中自动注册静态编译的后端，也支持通过 `ggml_backend_load()` 动态加载。

### llama.cpp 与 ggml 集成

- `src/llama.cpp` 通过 `llama_backend_init()`/`llama_model_load_from_file()` 完成后端初始化与设备选择，默认策略为“优先 GPU/I-GPU，再回退 CPU”。
- 模型加载阶段依据 `ggml_tensor` 的 `buffer type` 决定权重落地位置；推理阶段则在 `llama_context::decode` 内构建 `ggml` 计算图并提交到合适的后端。
- CLI (`tools/main/main.cpp`)、服务端 (`tools/server/server.cpp`) 与工具链统一依赖上述流程，因此新增后端需确保设备枚举、`--device` 过滤、`--list-devices` 输出均可见。

### 构建系统与可选组件

- CMake 顶层通过 `GGML_USE_*` 选项控制后端编译；每个后端在 `ggml/src/ggml-<backend>` 下有独立 `CMakeLists.txt` 描述依赖、编译宏、源文件。
- 现有后端展示了不同集成模式：
  - GPU 专用内核（CUDA、Metal、Vulkan）；
  - 异构接口封装（CANN、SYCL）；
  - CPU 加速库（BLAS、zDNN）。
- 新增 NPU 后端需遵循相同规范，定义 `GGML_USE_NPU`、`ggml_backend_npu_reg()` 等入口，并在根 `CMakeLists.txt`、`cmake/` 辅助模块中声明依赖探测。

## NPU 后端总体设计

### 设备发现与上下文管理

- **SDK 集成**：通过厂商提供的 C/C++ API（假设为 `libnpu_runtime`），实现设备枚举、版本探测、能力描述。构建期通过 `find_package` 或自定义 `FindNPU.cmake` 检测头文件与库路径。
- **注册流程**：
  1. 在 `ggml_backend_npu_reg()` 中创建 `ggml_backend_device` 实例，填充 `get_name`（如 `NPU0`）、`get_description`（芯片型号）、`get_memory` 等回调；
  2. 将设备注册进 `ggml_backend_register()`，保证 `ggml_backend_dev_get()` 可见；
  3. 提供 `init_backend` 回调，建立执行上下文（命令队列、stream、事件池、内核模块句柄）。
- **后端上下文**：维护 per-backend 的
  - 常驻命令队列/stream；
  - kernel module/graph 对象；
  - 内存分配器（设备 + pinned host）；
  - telemetry（错误码、profiling hook）。

#### Registration hook 参考：CANN 后端

复用 CANN 的接入模式可以快速搭建新的后端注册骨架，核心步骤如下（代码位于 `ggml/src/ggml-cann/ggml-cann.cpp` 与 `ggml/src/ggml-backend-reg.cpp`）：

- **全局注册函数**：`ggml_backend_cann_reg()` 使用静态 `ggml_backend_reg`，在首次调用时执行一次 `aclInit(nullptr)`，构造 `ggml_backend_cann_reg_context` 并按设备数量生成 `ggml_backend_device` 对象。
- **设备接口**：每个 `ggml_backend_device` 携带 `ggml_backend_cann_device_interface`，提供 `init_backend`、`get_buffer_type`、`supports_op`、事件管理等回调；设备上下文保存 `device id`、`name`（如 `CANN0`）、`description`。
- **缓冲类型绑定**：通过 `ggml_backend_cann_device_get_buffer_type()` 和 `ggml_backend_cann_supports_buft()`，确保 backend 与 buffer type 的 device id 匹配，避免跨设备误用。
- **宿主缓存支持**：`ggml_backend_cann_device_get_host_buffer_type()` 返回 pinned host buffer，供 CPU 与 NPU 间快速搬运。
- **静态注册点**：在 `ggml-backend-reg.cpp` 中，通过 `#ifdef GGML_USE_CANN` 调用 `register_backend(ggml_backend_cann_reg());`，确保编译启用后自动把 CANN 设备注入全局 registry。
- **动态加载钩子**：文件末尾的 `GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)` 允许在构建为可分发插件时，通过 `ggml_backend_load()` 动态装载。

NPU 后端可沿用该结构：替换底层 SDK 初始化、设备遍历和上下文对象，即可完成注册钩子的最小闭环。

### 缓冲区与内存策略

- **权重缓冲区**：实现 `ggml_backend_buffer_type`，将模型权重直接映射到 NPU 设备内存或统一内存；考虑按张量对齐（依据 SDK 要求，常见为 64/256 byte）。
- **计算缓冲区**：为 KV cache、临时激活分配独立 `buffer type`，支持 `GGML_BACKEND_BUFFER_USAGE_COMPUTE`，启用高带宽内存池 + 子分配器。
- **Host 交互**：
  - 若 SDK 提供零拷贝或 pinned host，暴露 `ggml_backend_dev_host_buffer_type`；
  - 封装 `set_tensor`/`get_tensor`/`cpy_tensor`，内部调度 DMA 或 API 提供的 memcpy。
- **多缓冲拼接**：使用 `ggml_backend_multi_buffer_alloc_buffer()` 合并多个设备块，以支持切片模型或大 tensor 分段加载。

### 计算图调度与回退策略

- `graph_compute` 实现需要解析 `ggml_cgraph`，根据算子类型匹配对应 kernel。推荐流程：
  1. 预处理图：拓扑遍历，收集节点属性（shape、数据类型、是否常量）。
  2. 对支持的算子生成 kernel launch 描述，并按依赖顺序入队。
  3. 对不支持或暂未落地的算子，调用 `ggml_backend_tensor_copy()` 将数据搬回 CPU backend，再借助 `ggml_backend_cpu_reg()` 执行，执行后再搬回（同 CANN、SYCL 的模式）。
- 提供 `supports_op`/`offload_op` 逻辑，确保 `ggml_backend_sched` 可正确将重算子派发到 NPU，同时避免重复搬运。
- 若 SDK 支持图级编译，可在 `graph_plan_create`/`graph_plan_compute` 中缓存编译结果，加速重复推理。

### 算子实现优先级

优先覆盖直接影响推理路径的算子，参考 `docs/ops.md` 与现有模型拓扑：

1. **线性层路径**：`MUL_MAT`、`MUL_MAT_ID`、`ACC`、`ADD`、`ADD1`、`SCALE`；
2. **归一化与激活**：`RMS_NORM`、`RMS_NORM_MUL_ADD`、`SILU`、`GELU` 族；
3. **注意力核心**：`ROPE`/`ROPE_BACK`、`FLASH_ATTN_EXT` 或最小实现的 `SOFT_MAX`、`MUL`、`ACC`；
4. **量化链路**：`DEQUANTIZE`、`QUANTIZE`、`GET_ROWS`（embedding）、`UPMAT` 相关；
5. **KV cache 维护**：`CPY`、`CONCAT`、`IM2COL`（如扩展模型）、`SSM_*`（若支持状态空间模型）。

初始版本可先实现 Float16 + 常见量化（Q4_0/Q5_1/Q8_0）的解量化路径，后续逐步覆盖更多 ops，并在 `docs/ops/` 下生成 `NPU.csv` 以同步文档。

### 数据类型与量化支持

- **计划优先级**：
  - Phase 1：FP16/FP32 推理链路（最易验证）；
  - Phase 2：INT4/INT8 权重量化（解量化到 FP16 执行）；
  - Phase 3：若硬件支持，探索 INT16/FP8 执行路径。
- 与 GGUF 模型格式兼容：复用现有量化元数据，不改动文件格式，仅在后端映射时处理。
- 对于 KV cache，若支持混合精度，可考虑 FP8/INT8 储存，提高容量。

### 多设备与异步执行

- **多卡并行**：利用 `tensor_split`（`llama_model_params.tensor_split`）实现张量并行，需提供 `ggml_backend_split_buffer_type` 实例。
- **流同步**：实现 `event_new`/`event_record`/`event_wait` 以支持跨 stream 或 CPU/NPU 同步；对外暴露 `ggml_backend_tensor_set_async` 与 `ggml_backend_buffer_set_usage` 支持异步搬运。
- **调度策略**：初版可采用单 stream 串行执行 + 显式同步，后续再引入 pipeline、prefetch。

## 构建与配置集成

- CMake 顶层：
  - 新增选项 `GGML_USE_NPU`；
  - 在 `ggml/src/CMakeLists.txt` 注册子目录 `ggml-npu`；
  - 在 `cmake/deps.cmake`（或等效文件）实现 SDK 探测逻辑，暴露 `NPU_INCLUDE_DIRS`、`NPU_LIBRARIES`；
  - 支持交叉编译（如仅在特定 Linux 发行版或 SoC 上可用）时，提供 `NPU_TOOLCHAIN`、`NPU_SYSROOT` 参数。
- 源码组织：
  - `ggml/src/ggml-npu/`：核心实现（设备枚举、内核封装、调度）；
  - `include/`：暴露 `ggml-npu.h`，供外部引用；
  - `docs/backend/NPU.md`：面向用户的安装与调试指引（后续任务）。
- 构建产物：
  - 静态库 `libggml-npu.a` / 动态库 `libggml-npu.so`；
  - 若需要离线编译 kernel，提供 `tools/npu/` 脚本生成 `.bin`/`.json` 并随构建打包。

## llama.cpp 运行时对接

- **设备选择**：更新 `llama_model_default_params()` 或相关初始化逻辑，确保 `GGML_BACKEND_DEVICE_TYPE_ACCEL`/`GPU` 分类合适；支持 `--device npu`、`--device npu0,npu1` 等参数。
- **模型加载**：在 `llama_model_loader` 中识别 NPU buffer type，直接将权重映射到 NPU；若模型过大，提供自动拆分或回退提示。
- **运行时配置**：
  - 新增环境变量（例如 `NPU_VISIBLE_DEVICES`、`NPU_ENABLE_PROFILER`）；
  - 通过 `llama.cpp` CLI 输出 NPU 设备列表、内存用量；
  - 在服务器模式下 (`tools/server`) 支持 per-session 设备绑定与多实例调度。

## 分阶段里程碑

| 阶段 | 里程碑 | 主要交付物 |
|------|--------|------------|
| Phase 0 | SDK 打通 & 构建脚本 | `GGML_USE_NPU` 选项、生效的探测脚本、空壳 backend 注册，可构建并列出设备（即便不可执行算子） |
| Phase 1 | 内存&调度最小闭环 | Buffer type + backend + CPU 回退链路，支持 FP16 基础算子，完成 `llama-bench` 小模型推理 |
| Phase 2 | 量化与核心算子 | Q4_0/Q8_0 解量化、注意力算子、RMSNorm、Softmax；发布 `docs/ops/NPU.csv`，通过主流 7B/13B 模型测试 |
| Phase 3 | 多设备与性能调优 | 张量并行、KV cache 优化、异步流水；完善 Profiling、自动化 Benchmark 报告 |
| Phase 4 | 生态整合 | 用户文档、Docker 镜像、CI 持续测试、向上游提 PR（视策略而定） |

## 测试与验证

- **算子级别**：使用 `tests/test-backend-ops.cpp` & `test-backend-ops` CLI 生成覆盖率报告；与 CPU 后端进行数值校验（绝对/相对误差阈值）。
- **集成级别**：
  - 运行 `tools/llama-bench`、`tools/perplexity`、`examples/embedding` 等场景，比较吞吐与一致性；
  - 若提供服务器模式，压测 `tools/server`。
- **CI 集成**：在专用 NPU runner 上配置 GitHub Actions / Jenkins pipeline，运行最小模型（如 TinyLlama）作为回归。
- **调试辅助**：封装 `--seed`、`--n-predict` 短测试脚本，便于对齐 CPU/NPU 输出。

## 性能分析与调优路径

- **热点定位**：集成硬件自带 profiler（例如 timeline、算子级性能计数器），结合 `llama-bench` 输出的 per-layer latency。
- **优化方向**：
  - Kernel fusion（如将 MatMul + Bias + RMSNorm 组合）；
  - 缓存管理（KV cache page 化 & 复用）；
  - 内存复制并行化（重叠 H2D/D2H 与计算）。
- **指标看板**：维护吞吐（tokens/s）、时延（p50/p99）、功耗等关键指标，建立基准值与回归阈值。

## 风险与缓解

- **SDK 稳定性**：版本兼容性或闭源依赖变动 → 封装统一抽象层，编写版本探测与 Feature flag。
- **算子覆盖不足**：初期回退 CPU 影响性能 → 按优先级表推进，实现自动报警（日志/metrics）提示未加速算子。
- **内存限制**：设备显存不足 → 实现权重切片、KV cache 压缩或自动回退提示。
- **调试复杂度**：异步执行导致定位困难 → 提供 `NPU_DEBUG_SYNC=1` 环境变量，强制同步便于复现。
- **生态差异**：上游接口演进 → 建立与 upstream 的合并节奏，保持 `ggml` API 版本兼容。

## 开放问题

- NPU SDK 是否提供图编译 / kernel 编译工具链？需要离线生成还是运行时动态编译？
- 是否允许将权重直接映射到设备内存（类似 UMA），或必须逐张量拷贝？
- 多进程/多实例共享设备策略如何设计（互斥、MPS、虚拟化）？
- 量化 INT4/INT8 是否有 native kernel 支持，还是需要先解量化到 FP16？
- 是否需要在服务端暴露更细粒度的资源配额（如 stream/SM 分配）？

## 知识沉淀与后续工作

- 在 `docs/backend/NPU.md` 撰写用户指南、FAQ、问题排查手册；
- 在 `docs/development` 目录新增 `HOWTO-npu-backend.md`，记录开发者接口、调试技巧；
- 为内部团队构建 demo 与 benchmark 报告（含对比 CPU/CUDA/Vulkan）；
- 后续可评估将优化贡献 upstream，以减少长期维护成本。
