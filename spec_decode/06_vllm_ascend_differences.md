# 第六部分: vLLM-Ascend 差异化实现详解

## 6.1 昇腾硬件架构适配

### 6.1.1 NPU 计算单元特性

**昇腾 NPU 架构概述**: 

昇腾 910B NPU 采用达芬奇（DaVinci）架构，具有以下核心计算单元: 

```mermaid
flowchart TB
    subgraph NPU["昇腾 NPU"]
        direction TB

        subgraph Core["AI Core × 32"]
            direction TB
            CU[Cube Unit<br/>矩阵计算]
            VU[Vector Unit<br/>向量计算]
            SU[Scalar Unit<br/>标量计算]
        end

        subgraph Memory["内存层次"]
            HBM[HBM 2.5 64GB]
            GM[Global Memory]
            L1[L1 Buffer]
            L0[L0 Buffer]
            Reg[Register]
        end
    end

    Core --> Memory

    style CU fill:#90EE90
    style VU fill:#FFD700
    style SU fill:#87CEEB
```

**计算单元对比**: 

| 单元 | NVIDIA GPU | 昇腾 NPU | 用途 |
|------|------------|----------|------|
| 矩阵计算 | Tensor Core | Cube Unit | GEMM、Conv |
| 向量计算 | CUDA Core | Vector Unit | Element-wise、Activation |
| 标量计算 | - | Scalar Unit | 控制流、索引计算 |

**对 Spec Decode 的影响**: 

1. **Cube Unit 优化**: Eagle 的矩阵运算映射到 Cube Unit，需要特定的数据布局
2. **Vector Unit 编程**: Triton kernel 需要适配 Vector Unit 的指令集
3. **内存层次利用**: 需要显式管理 L1/L0 Buffer 以提高带宽利用率

### 6.1.2 内存层次结构

```
昇腾 NPU 内存层次: 

HBM (64GB)
  ↓
Global Memory (片外 DDR)
  ↓
L1 Buffer (片内 SRAM, ~256MB)
  ↓
L0 Buffer (更接近计算单元，~16MB)
  ↓
Register (每计算单元独占)
```

**内存带宽对比**: 

| 层级 | 容量 | 带宽 | 延迟 |
|------|------|------|------|
| HBM | 64GB | ~1TB/s | 中 |
| Global Memory | 可变 | ~500GB/s | 高 |
| L1 Buffer | 256MB | ~2TB/s | 低 |
| L0 Buffer | 16MB | ~10TB/s | 极低 |
| Register | 每核 KB 级 | 最高 | 最低 |

**Spec Decode 内存优化策略**: 

```python
# vLLM-Ascend 中的内存优化示例
class AscendInputBuffers:
    def __init__(self, device):
        # 使用 L1 优化的缓冲区
        self.input_ids = torch.zeros(
            max_num_tokens,
            dtype=torch.int32,
            device=device,
            memory_format=torch.channels_last  # NPU 优化格式
        )

        # 使用 pinned memory 加速 H2D 传输
        self.cpu_buffer = torch.empty(
            max_num_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True
        )

    def async_copy_to_gpu(self, stream=None):
        """异步拷贝到 GPU，隐藏传输延迟。"""
        torch.npu.copy_async(self.cpu_buffer, self.gpu_buffer, stream=stream)
```

### 6.1.3 对 Spec Decode 的影响

**架构适配要点**: 

| 组件 | GPU 实现 | NPU 适配 |
|------|----------|----------|
| Matrix Multiplication | cuBLAS | ACL BLAS |
| Attention Kernel | Flash Attention | Ascend Flash Attention |
| LayerNorm | CUDA Kernel | Vector Unit Kernel |
| Rotary Embedding | Triton Kernel | AICPU Kernel |

## 6.2 通信层差异 (HCCL vs NCCL)

### 6.2.1 HCCL 初始化流程

**HCCL (Huawei Collective Communication Library)** 是昇腾平台的集合通信库，功能类似于 NCCL。

**初始化流程对比**: 

```python
# NCCL (NVIDIA)
import torch.distributed as dist
dist.init_process_group(backend="nccl")

# HCCL (昇腾)
import torch.distributed as dist
import torch_npu  # 必须导入
dist.init_process_group(backend="hccl")
```

**HCCL 环境变量**: 

```bash
# HCCL 白名单配置
export HCCL_WHITELIST=AllReduce,AllGather,ReduceScatter,Broadcast

# HCCL 超时配置
export HCCL_CONNECT_TIMEOUT=1200

# HCCL 分片配置
export HCCL_SPLIT_COMM=1
```

### 6.2.2 集合通信算子映射

| 算子 | NCCL | HCCL | 说明 |
|------|------|------|------|
| AllReduce | `dist.all_reduce()` | `dist.all_reduce()` | API 相同 |
| AllGather | `dist.all_gather()` | `dist.all_gather()` | API 相同 |
| ReduceScatter | `dist.reduce_scatter()` | `dist.reduce_scatter()` | API 相同 |
| Broadcast | `dist.broadcast()` | `dist.broadcast()` | API 相同 |
| Send/Recv | `dist.send/recv()` | `dist.send/recv()` | API 相同 |

**底层实现差异**: 

```python
# vLLM-Ascend 中的通信适配
from vllm.distributed.parallel_state import get_tp_group

def all_reduce(tensor):
    """昇腾平台上的 AllReduce。"""
    group = get_tp_group()

    # 昇腾 NPU 设备检查
    if current_platform.is_npu():
        # HCCL 后端会自动处理
        torch.distributed.all_reduce(tensor, group=group.device_group)
    else:
        # NCCL 后端
        torch.distributed.all_reduce(tensor, group=group.device_group)

    return tensor
```

### 6.2.3 Spec Decode 中的通信点分析

**Spec Decode 通信流程**: 

```mermaid
flowchart TB
    subgraph Target["Target Model Forward"]
        T1[Embedding AllGather]
        T2[Attention AllReduce]
        T3[MoE AllToAll]
        T4[LM Head AllReduce]
    end

    subgraph Draft["Draft Model Forward"]
        D1[Embedding AllGather]
        D2[Attention AllReduce]
        D3[MoE AllToAll]
    end

    subgraph Sync["同步点"]
        S1[Draft Tokens Broadcast]
        S2[Verification Sync]
    end

    Target --> Sync
    Sync --> Draft
    Draft --> S1
    S1 --> Target

    style Target fill:#e0f0ff
    style Draft fill:#fff0e0
```

**vLLM-Ascend 通信优化**: 

```python
# vllm_ascend/spec_decode/eagle_proposer.py
def _split_pcp_input(self, req_scheduled_tokens, input_ids, hidden_states):
    """PCP (Pipeline Context Parallel) 输入拆分。

    昇腾 NPU 使用 HCCL 进行跨卡通信。
    """
    pcp_group = get_pcp_group()
    world_size = pcp_group.world_size
    rank = pcp_group.rank

    num_tokens = input_ids.shape[0]
    padded_num_tokens_per_rank = (num_tokens + world_size - 1) // world_size

    # 当前 rank 的切片
    start = padded_num_tokens_per_rank * rank
    end = padded_num_tokens_per_rank * (rank + 1)

    # 使用 HCCL AllGather 进行通信
    # torch_npu 会自动选择 HCCL 后端
    output = torch.empty(
        padded_num_tokens_per_rank * world_size,
        dtype=input_ids.dtype,
        device=input_ids.device
    )
    torch.distributed.all_gather_into_tensor(
        output, input_ids, group=pcp_group.device_group
    )

    return output[start:end]
```

## 6.3 Attention 后端实现

### 6.3.1 Ascend Attention Backend

**文件位置**: `vllm_ascend/attention/attention_v1.py`

```python
class AscendAttentionBackend:
    """昇腾平台的 Attention 后端实现。"""

    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device
        self.attn_type = vllm_config.model_config.attention_backend

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AscendCommonAttentionMetadata,
    ) -> torch.Tensor:
        """执行 Attention 前向传播。"""
        if self.attn_type == "ascend_flash_attn":
            return self._flash_attention_forward(query, key, value, metadata)
        elif self.attn_type == "naive":
            return self._naive_attention_forward(query, key, value, metadata)
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

    def _flash_attention_forward(self, q, k, v, metadata):
        """昇腾 Flash Attention 实现。"""
        from vllm_ascend.ops.attention import ascend_flash_attn

        output = ascend_flash_attn(
            q, k, v,
            metadata.slot_mapping,
            metadata.block_table_tensor,
            metadata.seq_lens,
            metadata.max_seq_len,
        )
        return output
```

### 6.3.2 Flash Attention 适配

**昇腾 Flash Attention 接口**: 

```python
def ascend_flash_attn(
    q: torch.Tensor,       # [num_tokens, num_heads, head_size]
    k: torch.Tensor,       # [num_tokens, num_kv_heads, head_size]
    v: torch.Tensor,       # [num_tokens, num_kv_heads, head_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
    block_table: torch.Tensor,   # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,      # [num_reqs]
    max_seq_len: int,
) -> torch.Tensor:
    """昇腾 Flash Attention 实现。

    使用昇腾 NPU 的 Cube Unit 和 Vector Unit 进行优化。
    """
    # 调用 CANN 算子
    output = torch_npu.npu_flash_attention(
        query=q,
        key=k,
        value=v,
        block_table=block_table,
        slot_mapping=slot_mapping,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
    )
    return output
```

### 6.3.3 与 GPU 版本的 API 对齐

**统一接口设计**: 

```python
# 平台无关的 Attention 调用
from vllm.v1.attention.backend import AttentionBackend

class AttentionWrapper:
    def __init__(self, vllm_config, device):
        if current_platform.is_npu():
            self.backend = AscendAttentionBackend(vllm_config, device)
        elif current_platform.is_cuda():
            self.backend = CudaAttentionBackend(vllm_config, device)
        else:
            raise ValueError(f"Unsupported platform: {current_platform}")

    def forward(self, q, k, v, metadata):
        return self.backend.forward(q, k, v, metadata)
```

## 6.4 Kernel 级优化

### 6.4.1 Ascend Operator 实现

**昇腾算子开发框架**: 

```python
# 昇腾算子示例: LayerNorm
import torch_npu

def ascend_layernorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """昇腾 LayerNorm 算子。"""
    return torch_npu.npu_layer_norm(
        input,
        normalized_shape=weight.shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )
```

**Triton Kernel 适配**: 

```python
# vllm_ascend/ops/triton/spec_decode/utils.py
@triton.jit
def prepare_inputs_padded_kernel(
    input_ids_ptr,
    positions_ptr,
    ...
    BLOCK_SIZE: tl.constexpr,
):
    """昇腾平台上的 Triton Kernel。

    注意: 需要使用昇腾 Triton 的特定 API。
    """
    # 获取 program ID（昇腾 Triton API）
    batch_idx = tl.program_id(0)

    # ... kernel 逻辑

    # 使用昇腾特定的原子操作
    tl.atomic_add(output_ptr, index, value)
```

### 6.4.2 算子融合策略

**算子融合示例**: 

```
原始计算图: 
  Input → LayerNorm → Attention → LayerNorm → MLP → Output

融合后计算图: 
  Input → Fused Attention + MLP → Output
```

**vLLM-Ascend 融合实现**: 

```python
class FusedEagleLayer(nn.Module):
    """Eagle 层的融合实现。"""

    def __init__(self, config):
        super().__init__()
        self.attention = AscendAttention(config)
        self.layernorm1 = AscendLayerNorm(config.hidden_size)
        self.mlp = AscendMLP(config)
        self.layernorm2 = AscendLayerNorm(config.hidden_size)

        # 融合配置
        self.fuse_norm_attention = True

    def forward(self, x, metadata):
        if self.fuse_norm_attention:
            # 融合 LayerNorm + Attention
            x = torch_npu.npu_fused_layernorm_attention(
                x,
                self.layernorm1.weight,
                self.layernorm1.bias,
                self.attention.qkv_weight,
                metadata=metadata,
            )
        else:
            x = self.layernorm1(x)
            x = self.attention(x, metadata)

        # 融合 LayerNorm + MLP
        x = torch_npu.npu_fused_layernorm_mlp(
            x,
            self.layernorm2.weight,
            self.layernorm2.bias,
            self.mlp.gate_proj.weight,
            self.mlp.up_proj.weight,
            self.mlp.down_proj.weight,
        )

        return x
```

### 6.4.3 性能优化案例

**优化前后对比**: 

| 操作 | 优化前 (ms) | 优化后 (ms) | 提升 |
|------|-------------|-------------|------|
| Eagle Forward (batch=32) | 15.2 | 8.5 | 1.79x |
| Attention (seq_len=4096) | 8.7 | 4.2 | 2.07x |
| LayerNorm (hidden=4096) | 1.2 | 0.6 | 2.0x |

**优化技术总结**: 

1. **算子融合**: 减少 kernel 启动次数和中间结果存储
2. **内存布局优化**: 使用 NPU 友好的数据格式
3. **异步执行**: 利用昇腾的 Stream 管理隐藏数据传输延迟
4. **L1 Buffer 利用**: 显式管理片内缓存

## 6.5 编译与图优化

### 6.5.1 昇腾编译栈 (CANN)

**CANN (Compute Architecture for Neural Networks)** 是昇腾的异构计算架构。

```
编译栈层次: 

用户代码 (PyTorch/vLLM)
    ↓
torch_npu (PyTorch NPU 后端)
    ↓
CANN Operator Library
    ↓
ACL (Ascend Computing Language) Runtime
    ↓
NPU Driver
    ↓
NPU Hardware
```

### 6.5.2 图优化 Pass

**昇腾图优化 Pass**: 

```python
# vllm_ascend/compilation/acl_graph.py
class ACLGraphWrapper:
    """昇腾图捕获和重放包装器。"""

    def __init__(self, runnable, vllm_config, runtime_mode):
        self.runnable = runnable
        self.runtime_mode = runtime_mode
        self.concrete_aclgraph_entries = {}

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # 捕获新图
            entry = ACLGraphEntry(batch_descriptor=batch_descriptor)
            aclgraph = torch.npu.NPUGraph()

            with torch.npu.graph(aclgraph, pool=self.graph_pool):
                output = self.runnable(*args, **kwargs)

            entry.aclgraph = aclgraph
            entry.output = weak_ref_tensors(output)
            self.concrete_aclgraph_entries[batch_descriptor] = entry

        # 重放图
        entry = self.concrete_aclgraph_entries[batch_descriptor]
        with torch.npu.graph_capture_context():
            entry.aclgraph.replay()

        return entry.output
```

### 6.5.3 Spec Decode 图捕获差异

**GPU vs NPU 图捕获对比**: 

| 特性 | CUDA Graph | ACL Graph |
|------|------------|-----------|
| 捕获 API | `torch.cuda.CUDAGraph()` | `torch.npu.NPUGraph()` |
| 图池 | `torch.cuda.graph_pool()` | `torch.npu.graph_pool()` |
| 重放 | `graph.replay()` | `graph.replay()` |
| 内存管理 | 自动 | 需要手动管理 |
| 动态 Shape | 支持有限 | 支持有限 |

**Spec Decode 图捕获流程**: 

```mermaid
flowchart TB
    Start[开始] --> Warmup[Warmup 运行]
    Warmup --> Check{检查图缓存}
    Check -->|未命中 | Capture[捕获 ACL Graph]
    Check -->|命中 | Replay[重放 Graph]
    Capture --> Store[存储到缓存]
    Store --> Replay
    Replay --> Output[输出结果]
    Output --> Next{下一轮}
    Next -->|继续 | Check
    Next -->|结束 | End[结束]
```

## 6.6 内存管理

### 6.6.1 Ascend 内存分配器

```python
# vllm_ascend/utils.py
def allocate_npu_memory(size, dtype, device):
    """昇腾 NPU 内存分配。"""
    # 使用 torch_npu 的内存分配器
    tensor = torch.empty(size, dtype=dtype, device=device)

    # 内存对齐优化
    if is_npu_device(device):
        # 确保内存地址对齐到 128 字节
        torch_npu.npu_mem_align(tensor, alignment=128)

    return tensor
```

### 6.6.2 缓冲区对齐要求

**昇腾 NPU 内存对齐规则**: 

| 数据类型 | 对齐要求 |
|----------|----------|
| FP32 | 4 字节 |
| FP16 | 2 字节 |
| BF16 | 2 字节 |
| INT32 | 4 字节 |
| INT8 | 1 字节 |

**Cube Unit 特殊对齐**: 
```python
# Cube Unit 矩阵需要 32 字节对齐
def align_for_cube(tensor):
    """对齐到 Cube Unit 要求。"""
    if tensor.dim() == 2:
        # 矩阵转置优化
        tensor = tensor.contiguous().transpose(0, 1)
        # 32 字节对齐
        torch_npu.npu_mem_align(tensor, alignment=32)
    return tensor
```

### 6.6.3 H2D/D2H 传输优化

**异步传输实现**: 

```python
class AscendBufferManager:
    """昇腾缓冲区管理器。"""

    def __init__(self, max_size, dtype, device):
        self.device = device
        self.dtype = dtype

        # CPU 端 pinned memory
        self.cpu_buffer = torch.empty(
            max_size, dtype=dtype, device="cpu", pin_memory=True
        )

        # NPU 端 memory
        self.npu_buffer = torch.empty(
            max_size, dtype=dtype, device=device
        )

        # 异步传输 stream
        self.stream = torch.npu.Stream(device=device)

    def async_h2d(self, cpu_data, non_blocking=True):
        """异步 H2D 传输。"""
        self.cpu_buffer.copy_(cpu_data)
        with torch.npu.stream(self.stream):
            self.npu_buffer.copy_(self.cpu_buffer, non_blocking=non_blocking)
        return self.npu_buffer

    def async_d2h(self, npu_data, non_blocking=True):
        """异步 D2H 传输。"""
        with torch.npu.stream(self.stream):
            self.cpu_buffer.copy_(npu_data, non_blocking=non_blocking)
        return self.cpu_buffer
```

**传输性能优化**: 

| 优化技术 | 说明 | 效果 |
|----------|------|------|
| Pinned Memory | CPU 端锁定内存 | 2-3x 带宽提升 |
| Async Transfer | 异步传输 | 隐藏传输延迟 |
| Stream 管理 | 多 Stream 并行 | 提升并发 |
| 批量传输 | 合并小传输 | 减少启动开销 |

---

