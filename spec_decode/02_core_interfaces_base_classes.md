# 第二部分：Spec Decode 核心接口与基类分析

## 2.1 SpecDecodeBaseProposer 基类详解

### 2.1.1 类结构与核心属性

`SpecDecodeBaseProposer` 是 eagle 推测解码方法的基类。

**类层次结构**:
```
vLLM 主仓库:
  SpecDecodeBaseProposer (vllm/v1/spec_decode/eagle.py:57)
  └── EagleProposer (vllm/v1/spec_decode/eagle.py)  # 继承自 SpecDecodeBaseProposer

  NgramProposer (vllm/v1/spec_decode/ngram_proposer.py:12)  # 独立类，无继承
  MedusaProposer (vllm/v1/spec_decode/medusa.py:18)        # 独立类，无继承
  SuffixDecodingProposer (vllm/v1/spec_decode/suffix_decoding.py:9)  # 独立类

vllm-ascend:
  AscendEagleProposer (vllm_ascend/spec_decode/eagle_proposer.py:83)
  └── AscendMtpProposer (vllm_ascend/spec_decode/mtp_proposer.py:22)

  AscendNgramProposer (vllm_ascend/spec_decode/ngram_proposer.py:5)  # 继承自 Vllm NgramProposer
  AscendMedusaProposer (vllm_ascend/spec_decode/medusa_proposer.py:13)  # 继承自 Vllm MedusaProposer
  AscendSuffixDecodingProposer (vllm_ascend/spec_decode/suffix_proposer.py:4)  # 继承自 Vllm SuffixDecodingProposer
```

**关键说明**:
- `SpecDecodeBaseProposer` 仅为 Eagle 系列提出器提供基类
- `NgramProposer`、`MedusaProposer`、`SuffixDecodingProposer` 在 vLLM 中是独立类，**不继承** `SpecDecodeBaseProposer`
- vllm-ascend 通过继承 vLLM 对应类并添加昇腾特性实现

**核心属性**（`vllm/v1/spec_decode/eagle.py:57-250`）:

```python
class SpecDecodeBaseProposer:  # 注意：不是 ABC
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,  # 注意：没有默认值
        runner=None,
    ):
        # ========== 配置相关 ==========
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # ========== Runner 和设备 ==========
        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        # ========== 模型维度相关 ==========
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # ========== 并行配置 ==========
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if self.pass_hidden_states_to_model else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )

        # ========== 并行 drafting 参数 ==========
        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()

        # ========== 缓冲区大小 ==========
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)

        # ========== 多模态支持 ==========
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        # ========== Attention 相关 ==========
        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1

        # ========== Eagle3 特殊配置 ==========
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        # ========== 编译配置 ==========
        self.compilation_config = self.vllm_config.compilation_config

        # ========== CUDA Graph 支持 ==========
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # ========== 持久化缓冲区 (CUDA Graph 兼容) ==========
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )

        # M-RoPE/XDroPE 支持 - 根据 draft model 配置决定
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            self.mrope_positions = torch.zeros(
                (3, self.max_num_tokens + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_num_tokens + 1),
                dtype=torch.int64, device=device,
            )
        else:
            self.positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )

        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # arange 用于设置 query_start_loc
        max_num_slots_for_arange = max(max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        # ========== 用于 parallel drafting 的 mask ==========
        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._raise_if_multimodal()
            self._raise_if_mrope()
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        # ========== 备份和槽映射缓冲区 ==========
        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # ========== Attention 后端类型 (ROCM 平台) ==========
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            # ROCM 平台设置允许的 attention 类型
            ...
```

### 2.1.2 核心方法

#### `propose()` 方法
推测解码的核心方法，生成候选 token。

**方法签名**:
```python
def propose(
    self,
    target_token_ids: torch.Tensor,           # [num_tokens]
    target_positions: torch.Tensor,           # [num_tokens] or [3, num_tokens] when M-RoPE
    target_hidden_states: torch.Tensor,       # [num_tokens, hidden_size]
    next_token_ids: torch.Tensor,             # [batch_size]
    token_indices_to_sample: torch.Tensor | None,
    common_attn_metadata: CommonAttentionMetadata,
    sampling_metadata: SamplingMetadata,
    mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
```

#### `_get_positions()` / `_set_positions()`
位置 ID 的 getter/setter，支持 M-RoPE/XDroPE 的回退。

```python
def _get_positions(self, num_tokens: int):
    if self.uses_mrope:
        return self.mrope_positions[:, :num_tokens]
    if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
        return self.xdrope_positions[:, :num_tokens]
    return self.positions[:num_tokens]

def _set_positions(self, num_tokens: int, positions: torch.Tensor):
    if self.uses_mrope:
        self.mrope_positions[:, :num_tokens] = positions
    elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
        self.xdrope_positions[:, :num_tokens] = positions
    else:
        if self.vllm_config.model_config.uses_mrope:
            positions = positions[0]  # Convert M-RoPE positions
        self.positions[:num_tokens] = positions
```

#### `_greedy_sample()`
贪心采样生成候选 token。

```python
def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.use_local_argmax_reduction:
        return self.model.get_top_tokens(hidden_states)
    return self.model.compute_logits(hidden_states).argmax(dim=-1)
```

#### `initialize_cudagraph_keys()`
初始化 CUDA Graph 调度器的 keys。

```python
def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
    """Eagle 只支持 PIECEWISE cudagraphs (via mixed_mode)."""
    if (not self.speculative_config.enforce_eager
        and cudagraph_mode.mixed_mode() in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]):
        eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
    else:
        eagle_cudagraph_mode = CUDAGraphMode.NONE
    self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)
```

## 2.3 vLLM-Ascend 的 Spec Decode 入口

### 2.3.1 `get_spec_decode_method()` 工厂函数分析

**代码位置**: `vllm_ascend/spec_decode/__init__.py`

```python
def get_spec_decode_method(method, vllm_config, device, runner):
    if method == "ngram":
        return AscendNgramProposer(vllm_config, runner)  # 注意：不传 device 参数
    elif method == "suffix":
        return AscendSuffixDecodingProposer(vllm_config, runner)  # 注意：不传 device 参数
    elif method == "medusa":
        return AscendMedusaProposer(vllm_config, device)  # 注意：不传 runner 参数
    elif method in ("eagle", "eagle3"):
        return AscendEagleProposer(vllm_config, device, runner)
    elif method == "mtp":
        return AscendMtpProposer(vllm_config, device, runner)
    else:
        raise ValueError(f"Unknown speculative decoding method: {method}")
```

**参数差异说明**:
- `AscendNgramProposer` 和 `AscendSuffixDecodingProposer` 不需要 `device` 参数（基于 CPU 的 n-gram/suffix 匹配）
- `AscendMedusaProposer` 不需要 `runner` 参数
- `AscendEagleProposer` 和 `AscendMtpProposer` 需要完整的参数

### 2.3.2 五种推测方法的注册与分发

**支持的方法**:

| 方法 | Proposer 类 | 文件路径 | 继承关系 |
|------|-------------|----------|----------|
| `ngram` | `AscendNgramProposer` | `vllm_ascend/spec_decode/ngram_proposer.py:5` | `NgramProposer` (独立类) |
| `eagle`/`eagle3` | `AscendEagleProposer` | `vllm_ascend/spec_decode/eagle_proposer.py:83` | `EagleProposer` → `SpecDecodeBaseProposer` |
| `mtp` | `AscendMtpProposer` | `vllm_ascend/spec_decode/mtp_proposer.py:22` | `AscendEagleProposer` → `EagleProposer` → `SpecDecodeBaseProposer` |
| `suffix` | `AscendSuffixDecodingProposer` | `vllm_ascend/spec_decode/suffix_proposer.py:4` | `SuffixDecodingProposer` (独立类) |
| `medusa` | `AscendMedusaProposer` | `vllm_ascend/spec_decode/medusa_proposer.py:13` | `MedusaProposer` (独立类) |

### 2.3.3 代码位置

```
vllm-ascend/
└── vllm_ascend/
    └── spec_decode/
        ├── __init__.py              # get_spec_decode_method()
        ├── eagle_proposer.py        # AscendEagleProposer (昇腾实现)
        ├── mtp_proposer.py          # AscendMtpProposer (继承自 AscendEagleProposer)
        ├── medusa_proposer.py       # AscendMedusaProposer
        ├── ngram_proposer.py        # AscendNgramProposer
        └── suffix_proposer.py       # AscendSuffixDecodingProposer
```

---

## 2.4 vLLM-Ascend AscendEagleProposer 实现分析

### 2.4.1 类定义与继承

```python
class AscendEagleProposer(EagleProposer):
    _runnable: ACLGraphWrapper | Callable

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner)

        # ========== 调度相关 ==========
        self.use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling
        self.decode_threshold = 1 + self.num_speculative_tokens

        # ========== 缓冲区 ==========
        self.query_start_loc = self.runner._make_buffer(self.runner.max_num_reqs + 1, dtype=torch.int32)
        self.arange_cpu = torch.arange(self.arange.shape[0], device="cpu", dtype=torch.int32)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        # ========== 昇腾特性 ==========
        self.enable_shared_expert_dp = shared_expert_dp_enabled()
        self.pcp_size = self.runner.pcp_size  # Prefill Context Parallel
        self.dcp_size = self.runner.dcp_size  # Decode Context Parallel
        self.pcp_rank = self.runner.pcp_rank
        self.dcp_rank = self.runner.dcp_rank
        self.full_indices = range(
            self.runner.max_num_tokens * self.pcp_size * self.dcp_size
            + self.pcp_size * self.dcp_size * self.runner.max_num_reqs
        )

        # ========== 稀疏模型支持 ==========
        self.use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")

        # ========== Tensor Parallel 上下文 ==========
        # Draft 模型可能使用与目标模型不同的 TP 大小
        if vllm_config.parallel_config.tensor_parallel_size != self.speculative_config.draft_tensor_parallel_size:
            tp_group = init_model_parallel_group(...)
            self.tp_group_context = patch_tensor_parallel_group(tp_group)
        else:
            self.tp_group_context = nullcontext()

        # ========== CUDA/ACL Graph 配置 ==========
        self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
        if self.method == "mtp":
            self.use_cuda_graph = self.use_cuda_graph and not self.use_async_scheduling

        # ========== 槽映射缓冲区 ==========
        self.last_token_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32, device=device
        )
        slot_mapping_lens = self.runner.max_num_tokens + 2 * self.pcp_size * self.runner.max_num_reqs
        self.slot_mapping_group = [
            torch.zeros(slot_mapping_lens, dtype=torch.int32, device=device, pin_memory=self.runner.pin_memory)
            for _ in range(self.num_speculative_tokens)
        ]

        self._runnable = self._run_merged_draft
```

**关键特性**:
1. **PCP/DCP 并行**: 支持前缀上下文并行和解码上下文并行
2. **ACL Graph**: 使用昇腾的 ACL Graph 替代 CUDA Graph
3. **Tensor Parallel 兼容**: 支持 draft 模型使用不同的 TP 大小
4. **稀疏模型支持**: 通过 `use_sparse` 标志支持稀疏 MoE 模型

### 2.4.2 AscendMtpProposer 特殊实现

`AscendMtpProposer` 继承自 `AscendEagleProposer`，针对 MTP（Multi-Token Prediction）模型进行了特殊处理：

**主要重写方法**:
- `dummy_run()`: 处理 MTP 特殊的图捕获逻辑
- `_propose()`: 处理 MTP 多步推测的特殊流程

**关键差异**:
1. MTP 模型每步推测都需要独立的前向传播
2. 需要处理 `shared_expert_dp` 和 FULL graph 模式的兼容性问题
3. 需要特殊的 attention 元数据更新逻辑

---

## 2.5 NgramProposer 实现分析

### 2.5.1 vLLM 主仓库实现

**代码位置**: `vllm/v1/spec_decode/ngram_proposer.py`

```python
class NgramProposer:
    def __init__(self, vllm_config: VllmConfig):
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # Numba 加速相关
        self.num_tokens_threshold = 8192
        self.num_numba_thread_available = min(1, (cpu_count // 2))
```

**核心算法**:
1. 使用 KMP 算法的变体（LPS 数组）查找最长匹配 n-gram
2. 使用 Numba JIT 编译加速批量处理
3. 基于历史 token 序列预测后续 token

### 2.5.2 vllm-ascend 实现

**代码位置**: `vllm_ascend/spec_decode/ngram_proposer.py`

```python
class AscendNgramProposer(NgramProposer):
    def __init__(self, vllm_config, runner):
        self.runner = runner
        super().__init__(vllm_config)

    def load_model(self, *args, **kwargs):
        pass  # No model to load

    def dummy_run(self, ...):
        pass  # No computation needed

    def propose(self, sampled_token_ids, ...) -> list[list[int]]:
        # 调用 batch_propose 进行 N-gram 匹配
```

---

## 2.6 SuffixDecodingProposer 实现分析

### 2.6.1 vllm-ascend 实现

**代码位置**: `vllm_ascend/spec_decode/suffix_proposer.py`

```python
class AscendSuffixDecodingProposer(SuffixDecodingProposer):
    def __init__(self, vllm_config, runner):
        super().__init__(vllm_config)
        self.runner = runner

    def load_model(self, *args, **kwargs):
        pass  # No model to load

    def dummy_run(self, ...):
        pass  # No computation needed

    def propose(self, valid_sampled_token_ids) -> list[list[int]]:
        return super().propose(self.runner.input_batch, valid_sampled_token_ids)
```

---

## 2.7 MedusaProposer 实现分析

### 2.7.1 vllm-ascend 实现

**代码位置**: `vllm_ascend/spec_decode/medusa_proposer.py`

```python
class AscendMedusaProposer(MedusaProposer):
    """Medusa proposer class for generating token sequences"""

    @torch.inference_mode()
    def dummy_run(self, ...):
        hidden_states = torch.zeros(...)
        with set_ascend_forward_context(...):
            self.model(hidden_states)
            dummy_compute_logits(hidden_states)

    def propose(self, valid_sampled_token_ids, sampling_metadata, ...) -> list[list[int]]:
        # 从 sample_hidden_states 提取正确的 hidden states
        # 调用 super().propose 生成 Medusa 候选 token
```

---

## 2.8 关键差异总结

### 2.8.1 vLLM vs vllm-ascend 架构差异

| 方面 | vLLM | vllm-ascend |
|------|------|-------------|
| 接口定义 | 无统一接口 | 无统一接口 (直接继承) |
| 工厂函数 | 分散在各模块 | 统一 `get_spec_decode_method()` |
| 图执行支持 | CUDA Graph | ACL Graph (昇腾) |
| Attention | 多种后端 | 统一的 AscendAttention |
| Forward 上下文 | `set_forward_context` | `set_ascend_forward_context` |

### 2.8.2 各 Proposer 特性对比

| Proposer | 是否需要模型 | 是否继承 SpecDecodeBaseProposer | 是否支持并行 drafting | 是否支持 M-RoPE | 主要应用场景 |
|----------|-------------|-------------------------------|---------------------|----------------|-------------|
| AscendNgramProposer | ❌ | ❌ (独立类) | ❌ | ❌ | 基于历史 n-gram 匹配 |
| AscendSuffixDecodingProposer | ❌ | ❌ (独立类) | ❌ | ❌ | 基于后缀树匹配 |
| AscendEagleProposer | ✅ | ✅ | ✅ | ✅ | 基于独立 draft 模型 |
| AscendMtpProposer | ✅ | ✅ (间接) | ✅ | ✅ | 基于目标模型 MTP 头 |
| AscendMedusaProposer | ✅ | ❌ (独立类) | ❌ | ❌ | 基于 Medusa 多头 |

---

## 2.9 数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Model Runner (vllm-ascend)                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                  get_spec_decode_method()                      │    │
│  │   入口：根据 method 参数分发到对应的 AscendProposer 实现         │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    vLLM 基类                                    │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │    │
│  │  │    Ngram     │  │    Eagle     │  │   Suffix     │          │    │
│  │  │  Proposer    │  │  Proposer    │  │  Proposer    │          │    │
│  │  │  (独立类)     │  │  (继承自      │  │  (独立类)     │          │    │
│  │  │              │  │  SpecBase)   │  │              │          │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐                             │    │
│  │  │    Medusa    │  │     MTP      │                             │    │
│  │  │  Proposer    │  │  Proposer    │                             │    │
│  │  │  (独立类)     │  │  (继承自      │                             │    │
│  │  │              │  │  Eagle)      │                             │    │
│  │  └──────────────┘  └──────────────┘                             │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                  vllm-ascend Ascend 实现                        │    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │  AscendEagleProposer (EagleProposer)                     │  │    │
│  │  │  └── AscendMtpProposer (AscendEagleProposer)             │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │  AscendNgramProposer (NgramProposer)                     │  │    │
│  │  │  AscendSuffixDecodingProposer (SuffixDecodingProposer)   │  │    │
│  │  │  AscendMedusaProposer (MedusaProposer)                   │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  特性说明:                                                               │
│  - AscendEagleProposer: 添加 ACL Graph、PCP/DCP 并行、昇腾 Attention    │
│  - AscendMtpProposer: 在 AscendEagleProposer 基础上处理 MTP 特殊逻辑    │
│  - 其他 AscendProposer: 主要添加 dummy_run 和昇腾 forward 上下文支持     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.10 Rejection Sampler 实现分析

### 2.10.1 核心概念

Rejection Sampler 是推测解码的关键组件，负责验证 draft token 是否被目标模型接受。其核心算法基于 **典型接受采样 (Typical Acceptance Sampling)**：

```
对于每个 draft token:
    如果 target_prob(draft_token) / draft_prob(draft_token) >= uniform_random:
        接受该 token，继续验证下一个
    否则:
        拒绝该 token，从修正分布中采样一个替代 token
        停止验证后续 token
```

### 2.10.2 vllm-ascend 实现架构

**代码位置**: `vllm_ascend/sample/rejection_sampler.py`

```python
def rejection_sample(
    draft_token_ids: torch.Tensor,      # [num_tokens]
    num_draft_tokens: list[int],        # [batch_size]
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    draft_probs: torch.Tensor | None,   # [num_tokens, vocab_size]
    target_logits: torch.Tensor,        # [num_tokens, vocab_size]
    bonus_token_ids: torch.Tensor,      # [batch_size, 1]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:                      # [batch_size, max_spec_len + 1]
```

**核心流程**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    rejection_sample() 主流程                         │
├─────────────────────────────────────────────────────────────────────┤
│  1. 根据采样类型分支:                                                 │
│     └─ all_greedy? → 贪心采样路径                                    │
│     └─ all_random?  → 随机采样路径                                   │
│     └─ 混合模式     → 分别处理                                       │
│                                                                      │
│  2. 贪心采样路径:                                                     │
│     └─ target_argmax = target_logits.argmax()                       │
│     └─ 比较 draft_token_ids vs target_argmax                        │
│     └─ rejection_greedy_sample_*() 输出结果                          │
│                                                                      │
│  3. 随机采样路径:                                                     │
│     └─ target_probs = target_logits.softmax()                       │
│     └─ uniform_probs = generate_uniform_probs()                     │
│     └─ recovered_tokens = sample_recovered_tokens()                 │
│     └─ 根据是否使用 block_verify 选择内核:                           │
│        ├─ max_spec_len >= 3 → block_verify 内核 (提高接受率)         │
│        └─ max_spec_len < 3  → 标准内核                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.10.3 关键函数详解

#### `apply_sampling_constraints()`
应用采样约束 (temperature, top-k, top-p) 到 logits。

```python
def apply_sampling_constraints(
    logits: torch.Tensor,           # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    if sampling_metadata.all_greedy:
        return logits  # 贪心采样不需要处理

    # Temperature 缩放
    logits.div_(temperature.unsqueeze(-1))

    # Top-k 和 Top-p 过滤
    return apply_top_k_top_p(logits, top_k, top_p)
```

#### `sample_recovered_tokens()`
当 draft token 被拒绝时，从修正分布中采样替代 token。

```python
def sample_recovered_tokens(...) -> torch.Tensor:
    """
    修正分布计算:
    - N-GRAM 模式: prob = target_probs，但将被拒绝的 draft token 概率设为 0
    - 概率模式: prob = max(0, target_probs - draft_probs)

    采样方法: 使用 Gumbel-Max 技巧
    q ~ Exponential(1)
    recovered_token = argmax(prob / q)
    """
    # 生成归一化分布
    q.exponential_()

    # 计算修正概率
    if IS_NGRAM:
        modified_target_probs = target_probs.clone()
        modified_target_probs[token_indices, draft_token_ids] = 0
        prob = modified_target_probs
    else:
        prob = torch.maximum(target_probs - draft_probs, 0)

    # 选择最大概率 token
    return argmax(prob / q)
```

### 2.10.4 Block Verify 优化

当 `max_spec_len >= 3` 时，使用 **Block Verify** 技术（来自 MagicMTP 论文）提高接受率。

**核心思想**: 传统 rejection sampling 是顺序验证，一旦拒绝就停止。Block Verify 允许并行验证所有 token，然后选择最长的接受序列。

```python
def rejection_random_sample_block_verify_pytorch(...):
    # 计算累积接受概率
    pi = target_token_probs / draft_token_probs
    pi = pi.clamp(max=1.0)
    pi = torch.cumprod(pi, dim=-1)  # 累积乘积

    uniform_token_probs = torch.cumprod(uniform_token_probs, dim=-1)

    # 找到所有合法位置
    legal_mask = (draft_token_probs > 0) & (pi >= uniform_token_probs)

    # 找到最后一个接受位置
    last_accept_pos = ...  # 从后往前找第一个合法位置

    # 输出: 接受的 draft tokens + 拒绝位置的 recovered token + bonus token
```

### 2.10.5 Triton vs PyTorch 实现

vllm-ascend 提供两种实现路径：

| 方面 | Triton 实现 | PyTorch 实现 |
|------|-------------|--------------|
| 性能 | 更快 (GPU 内核) | 较慢 (但无 Triton 依赖) |
| 兼容性 | 需要 `HAS_TRITON=True` | 通用 |
| 核心函数 | `rejection_greedy_sample_with_triton` | `rejection_greedy_sample_pytorch` |
|       | `rejection_random_sample_kernel` | `rejection_random_sample_pytorch` |
|       | `rejection_random_sample_block_verify_kernel` | `rejection_random_sample_block_verify_pytorch` |

### 2.10.6 数据流示意

```
输入:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ draft_token_ids │    │ target_logits   │    │ bonus_token_ids │
│ [num_tokens]    │    │ [num_tokens, V] │    │ [batch_size, 1] │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      │
    ┌────────────┐        ┌────────────┐               │
    │ draft_probs│        │target_probs│               │
    │ (可选)      │        │ = softmax  │               │
    └─────┬──────┘        └─────┬──────┘               │
          │                     │                      │
          └──────────┬──────────┘                      │
                     ▼                                 │
            ┌────────────────┐                         │
            │ rejection_sample│                        │
            │ 验证流程        │                        │
            └────────┬───────┘                         │
                     │                                 │
                     ▼                                 │
输出:                │                                 │
┌─────────────────────────────────────────────────────┴───┐
│ output_token_ids [batch_size, max_spec_len + 1]          │
│                                                          │
│ 例如: batch_size=2, max_spec_len=4                       │
│   请求1: [token1, token2, recovered, bonus, PLACEHOLDER] │
│   请求2: [token1, token2, token3, token4, bonus]         │
│                                                          │
│ 说明:                                                    │
│   - token1-N: 被接受的 draft tokens                      │
│   - recovered: 拒绝后从修正分布采样的 token               │
│   - bonus: 全部接受后的额外 token                        │
│   - PLACEHOLDER: 未使用的位置填充                        │
└──────────────────────────────────────────────────────────┘
```

---

## 2.11 总结

本文档详细分析了 vLLM 和 vllm-ascend 中 Speculative Decoding 的核心接口与实现：

1. **基类架构**: `SpecDecodeBaseProposer` 为 Eagle 系列 proposer 提供统一基础，其他 proposer (Ngram, Medusa, Suffix) 为独立实现

2. **vllm-ascend 扩展**: 通过继承 vLLM 基类并添加昇腾特性（ACL Graph, PCP/DCP 并行, Ascend Attention）

3. **五种 Proposer**:
   - `AscendEagleProposer`: 基于独立 draft 模型，功能最完整
   - `AscendMtpProposer`: 继承 Eagle，处理 MTP 多步推测
   - `AscendNgramProposer`: 基于 n-gram 匹配，无需模型
   - `AscendMedusaProposer`: 基于 Medusa 多头，需要轻量级模型
   - `AscendSuffixDecodingProposer`: 基于后缀树，无需模型

4. **Rejection Sampler**: 实现典型接受采样，支持贪心和随机采样，Block Verify 优化提高接受率
