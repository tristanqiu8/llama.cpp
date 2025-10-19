# CANN 后端源码导读

本文档汇总 `ggml/src/ggml-cann` 目录内所有源码的功能、依赖与设计要点，帮助快速理解 CANN 后端如何接入 GGML。

## 目录概览
- `CMakeLists.txt`：定位 Ascend 工具链、选择支持的 SoC，生成 `ggml-cann` 后端库。
- `Doxyfile`：为该目录生成 doxygen 文档的配置模板。
- `common.h`：公共常量、设备信息、任务队列、内存池与上下文定义。
- `acl_tensor.h / acl_tensor.cpp`：GGML 张量到 ACL 张量的桥接层，含广播工具。
- `aclnn_ops.h / aclnn_ops.cpp`：对 Ascend ACLNN 算子的全面封装，并实现 GGML 所需算子逻辑。
- `ggml-cann.cpp`：CANN 后端主体，实现缓冲区类型、数据搬运、算子分发、图缓存等功能。

## 构建脚本 (`CMakeLists.txt`)
- 自动探测 `ASCEND_TOOLKIT_HOME` 和 SoC 型号（默认通过 `npu-smi`）。
- 根据 SoC 决定编译宏（如 `ASCEND_910B`、`ASCEND_310P`）。在 310P 上明确禁用 `USE_ACL_GRAPH`。
- 设置包含 / 库目录并链接 `ascendcl`, `nnopbase`, `opapi`, `acl_op_compiler`。
- 使用 `ggml_add_backend_library` 收集当前目录下的所有 `.cpp` 生成 `ggml-cann`。

## 文档配置 (`Doxyfile`)
- 输出目录 `docs`，生成英文文档。
- 指定项目名 `ggml` 及说明，启用常规 doxygen 选项；方便为该目录单独生成 API 文档。

## 公共基础 (`common.h`)
### 常量与设备信息
- `MATRIX_ROW_PADDING=512`：量化行对齐的默认填充。
- `GGML_CANN_MAX_STREAMS=8`：每个设备保留的流数量上限。
- `ggml_cann_device_info`：启动时缓存设备数量、是否支持 VMM、共享内存上限、总显存等信息。
- `ggml_cann_info()`：按需初始化并缓存上述信息。

### 错误处理与环境工具
- `ggml_cann_error()`：打印详细错误、设备信息并终止。
- `get_env/parse_bool/parse_integer()`：统一的环境变量解析，常用于后端配置。

### 内存池抽象
- `ggml_cann_pool`：纯虚基类，定义 `alloc/free`。
- `ggml_cann_pool_alloc`：RAII 包装，确保异步算子完成前不会释放内存。
- `ggml_backend_cann_context::new_pool_for_device()`：按环境变量和设备能力选择 `prio`（优先队列池）、`leg`（分段池）或 `vmm`（虚拟内存池）。
- 环境变量：`GGML_CANN_MEM_POOL`（`prio`/`leg`），`GGML_CANN_DISABLE_BUF_POOL_CLEAN` 禁用闲置清理。

### 任务与缓存
- `cann_task`：抽象任务接口。
- `cann_task_queue`：无锁环形缓冲任务队列，可按需启动线程；用于异步算子提交、内存填充等。受 `GGML_CANN_ASYNC_MODE` 控制。
- `ggml_cann_graph_lru_cache`（在 `USE_ACL_GRAPH` 下）：按照 `GGML_CANN_GRAPH_CACHE_CAPACITY` 限制缓存转换后的 ACL 图，并提供 LRU 淘汰。
- `ggml_cann_rope_cache`：缓存 RoPE 所需的 sin/cos / theta_scale，避免重复计算；记录参数以判定缓存是否可复用。
- `ggml_cann_tensor_cache`：缓存常量张量（RMSNorm 的 1 向量、0 向量等）。

### 后端上下文
- `ggml_backend_cann_context`：每个设备一个实例，包含设备 ID、描述、流数组、异步任务队列、rope/常量缓存、内存池句柄等。
- 构造时读取：
  - `GGML_CANN_ASYNC_MODE`：是否异步提交算子并排队。
  - `GGML_CANN_ACL_GRAPH`：启用图模式（有条件编译）。
  - 创建流时确保已设置正确 device。

## ACL 张量封装 (`acl_tensor.h / acl_tensor.cpp`)
### 类型映射与张量创建
- `ggml_cann_type_mapping()`：GGML 类型到 ACL 数据类型。
- `ggml_cann_create_tensor()`：基于 GGML 张量（含广播或偏移信息）构造 ACL 张量；逆序尺寸以匹配 ACL 通用布局。
- 模板版本允许直接从原始缓冲区 + 尺寸/步长创建。

### 广播工具
- `ggml_cann_need_bcast()`：检测是否需要广播。
- `ggml_cann_get_bcast_shape()`：类似 numpy 规则，插入额外维度支持步长广播。
- `ggml_cann_get_mulmat_bcast_shape()`：专用于矩阵乘广播，处理 batch 维扩展。
- 提供宏 `BCAST_SHAPE`、`BCAST_PARAM`、`BCAST_MUL_MAT_SHAPE` 简化调用。

## ACLNN 算子封装 (`aclnn_ops.h / aclnn_ops.cpp`)
### 基础设施
- 统一宏 `GGML_CANN_CALL_ACLNN_OP`：负责申请工作空间、异步提交任务或同步执行。
- `any_acl_resource` 与 `register_acl_resources`：RAII 管理 `aclTensor`、`aclScalar` 等资源，支持在异步模式下延迟回收。
- `aclnn_task`/`release_resource_task`/`async_memcpy_task`/`async_memset_task`：用于异步提交 ACL 执行、复制及资源回收。
- `ggml_cann_async_memcpy/memset()`：根据异步模式决定直接执行或交给任务队列。

### 张量与广播工具
- `bcast_shape()`：判断 src0/src1 是否需要广播，并创建对应 ACL 张量。
- `GGML_CANN_CALL_OP_UNARY` 与 `GGML_CANN_CALL_OP_UNARY_GATED`：模板化封装一元算子及带门控（GLU 类）算子。

### 一元/二元基础算子
- `ggml_cann_repeat/concat/scale/acc` 等函数直接映射到对应 ACLNN 算子（Repeat、Cat、Muls、InplaceAdd 等），内部处理 strides 和 op 参数。
- `ggml_cann_binary_op` 模板组合加减乘除。
- `ggml_cann_arange/argsort/clamp/leaky_relu/softmax/step/elu/mean/...` 等覆盖绝大多数通用算子。

### 归一化与激活
- `ggml_cann_norm`：基于 ACL `LayerNorm`。
- `ggml_cann_group_norm`：额外申请 mean/rstd 缓冲；参数从 `dst->params` 读取。
- `ggml_cann_rms_norm`：复用缓存的常量向量与 rstd 缓冲，减少重复分配。

### 数据排布与复制
- `ggml_cann_dup/cpy`：处理跨 dtype 复制、非连续张量复制、需要转置的情况。
- `aclnn_cast/aclnn_zero/aclnn_fill_scalar`：辅助 dtype 转换、填充统一值。
- `ggml_cann_pad/pad_reflect_1d/upsample_nearest2d/pool2d`：覆盖常见图像操作。

### 广义算子
- `ggml_cann_sum_rows/sum`：统一走 `ReduceSum`。
- `ggml_cann_diag_mask`：生成对角遮罩，组合 `Triu/Tril` 与加法。
- `ggml_cann_get_rows/set_rows`：基于 `IndexSelect/IndexCopy` 完成 embedding gather/scatter；对量化类型先解码再操作。
- `ggml_cann_im2col`：兼容 1D/2D，必要时转 dtype、重排维度，并转换回目标布局。
- `ggml_cann_timestep_embedding`：构建频率表、应用 sin/cos 并拼接。

### RoPE 与缓存
- `ggml_cann_rope`：支持 Neox 模式、YARN 外推；若在 310P 上编译，内含特化路径（多次 roll + 手动乘加）。
- `aclnn_cache_init()`：负责按参数生成/复用 sin/cos 缓存，包含频率扩展、YARN ramp、重复策略（repeat 或 repeat_interleave）。

### Softmax & 掩码
- `ggml_cann_softmax`：
  - 先乘以缩放系数。
  - 若存在掩码，调用 `aclnn_add_alibi` 处理 ALiBi 斜率（自动生成 slope 并广播）。
  - 最后执行 `aclnnSoftmax`。

### 矩阵乘与量化
- `ggml_cann_mul_mat`：
  - 借助广播工具生成输入/权重/输出 ACL 张量。
  - 根据维度切换 `Mm`、`BatchMatMul` 或 `Matmul`。
  - 环境变量 `GGML_CANN_WEIGHT_NZ` 决定是否将权重转为 FRACTAL_NZ 格式。
- `ggml_cann_mul_mat_quant`：
  - 处理 Q4_0/Q8_0 权重 + FP16/FP32 输入。
  - 将输入转换为 FP16，分块调用 `WeightQuantBatchMatmulV2`，并在必要时再转换输出 dtype。
- `ggml_cann_mul_mat_id`：
  - MoE 路径，区分 FP 输入与量化权重，逐 token 按路由索引缩小权重后再做 BatchMatMul。

### Flash Attention
- `ggml_cann_flash_attn_ext`：完成 Q/K/V 转置、可选掩码广播、调用 `FusedInferAttentionScoreV2`，并在 FP32 输出场景下再转换。

## 后端主体 (`ggml-cann.cpp`)
### 设备与环境管理
- `ggml_cann_error/set_device/get_device`：统一错误输出与设备切换。
- `ggml_cann_init/ggml_cann_info`：枚举设备、探测 VMM 支持、记录显存。
- 环境变量：
  - `GGML_CANN_MEM_POOL`：`prio`/`leg`/默认自动（若设备支持 VMM 且未显式设为 `leg`，优先 VMM）。
  - `GGML_CANN_DISABLE_BUF_POOL_CLEAN`：禁用缓冲池闲置回收。
  - `GGML_CANN_WEIGHT_NZ`：控制是否优先将权重转换为 FRACTAL_NZ。

### 内存池实现
- `ggml_cann_pool_buf_prio`：使用 `priority_queue` 按大小重用缓冲，带迟滞清理策略（>1MB 闲置 >100ms 即释放，除非禁用）。
- `ggml_cann_pool_buf`：固定槽位（128 个）管理器，避免动态分配；适合低版本设备。
- `ggml_cann_pool_vmm`：利用 Ascend VMM 能力映射大页，按 32MB 对齐管理。

### 缓冲区类型
- `ggml_backend_cann_buffer_type`：128 字节对齐，按需扩展量化行数以满足 `MATRIX_ROW_PADDING`，或为 FRACTAL_NZ 预留更大空间（通过 `aclnnCalculateMatmulWeightSizeV2`）。
- 提供 `ggml_backend_cann_host_buffer_type`：在主机侧中转，必要时退回 CPU 后端的内存方案。

### 权重量化与格式转换
- `need_transform()`：量化权重（Q4_0/Q8_0）需要特殊转置 / 反转。
- `ggml_backend_cann_transform` / `transform_back`：在 `to`/`from` CANN 时处理量化布局。
- `ggml_cann_nz_workspace`：全局缓存 FRACTAL_NZ 权重缓冲，避免重复 `aclrtMalloc`。

### 计算调度
- `ggml_cann_compute_forward()`：根据 `ggml_op` 枚举分发到 `aclnn_ops` 中的具体实现，是后端算子的中枢。
- `ggml_backend_cann_graph_compute()`：
  - 如果启用图模式，先在 LRU 缓存中查找匹配图；否则执行 eager 路径。
  - 未命中时构建新图节点描述 `ggml_graph_node_properties` 并缓存。
- `ggml_cann_graph_lru_cache`：维持最近使用的 ACL 图，超过容量自动释放。
- `ggml_cann_async_memcpy` 等工具确保异步模式下显存访问正确排队。

### 数据搬运与内存同步
- `ggml_backend_cann_buffer_init_tensor`：在量化张量上补零，防止 padding 带来随机值。
- `ggml_backend_cann_buffer_copy_tensor` / `ggml_backend_cann_buffer_get_tensor`：封装 H2D/D2H 拷贝（支持 views 和偏移），并在必要时使用 CPU 桥接。
- `ggml_backend_cann_host_malloc/free`：优先使用 pin memory（`aclrtMallocHost`），失败时退回普通 malloc。

### 后端注册
- 提供 `ggml_backend_cann()`、`ggml_backend_cann_buffer_type()`、`ggml_backend_cann_host_buffer_type()` 等工厂函数。
- `ggml_backend_cann_reg()`：向 GGML 框架注册后端；`ggml_backend_cann_get_device_count()` 查询可用设备数。

## 环境变量速览
- `GGML_CANN_ASYNC_MODE`：启用异步算子提交与任务队列。
- `GGML_CANN_ACL_GRAPH`：选择 ACL 图模式执行。
- `GGML_CANN_GRAPH_CACHE_CAPACITY`：图缓存 LRU 容量（默认 12）。
- `GGML_CANN_MEM_POOL`：强制使用 `prio` 或 `leg` 内存池；默认自动（优先 VMM）。
- `GGML_CANN_DISABLE_BUF_POOL_CLEAN`：禁用空闲缓冲清理。
- `GGML_CANN_WEIGHT_NZ`：是否自动转换矩阵乘权重到 FRACTAL_NZ。

## 运行流程摘要
1. **初始化**：构建 `ggml_backend_cann_context`，探测设备、创建流与内存池。
2. **加载权重**：通过缓冲区接口在 H2D 过程中按需转为量化格式或 NZ 格式。
3. **执行算子**：`ggml_cann_compute_forward` 按 GGML 图节点调用封装好的 `ggml_cann_*` 函数；异步模式下任务被排入 `cann_task_queue`。
4. **可选图模式**：若开启 `USE_ACL_GRAPH` 并命中缓存，直接重放 ACL 计算图以降低调度开销。
5. **收尾**：上下文析构时停止任务队列、销毁流、释放缓存（rope、常量、图等）。

---
以上即该目录源码的核心逻辑梳理，可作为继续深入阅读或二次开发 CANN 后端时的导航。
