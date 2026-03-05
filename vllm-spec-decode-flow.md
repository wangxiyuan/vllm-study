# vLLM Spec Decode 流程分析

基于 vllm 和 vllm-ascend 代码的 Speculative Decoding 完整流程分析。

## 一、核心组件

### 1.1 主要文件位置

**vllm (upstream):**
```
vllm/v1/spec_decode/
├── eagle.py              # EagleProposer 核心实现
├── draft_model.py        # DraftModelProposer
├── metadata.py           # SpecDecodeMetadata
├── ngram_proposer.py     # NgramProposer
├── medusa.py             # MedusaProposer
└── base.py               # SpecDecodeBaseProposer 基类

vllm/v1/worker/gpu/
├── model_runner.py       # GPUModelRunner - sample_tokens() 调用 propose()
└── spec_decode/
    ├── eagle/
    │   ├── speculator.py # EagleSpeculator (run_model, propose, generate_draft)
    │   ├── utils.py
    │   └── cudagraph.py
    └── rejection_sample.py
```

**vllm-ascend:**
```
vllm_ascend/spec_decode/
└── eagle_proposer.py     # AscendEagleSpeculator (继承 EagleSpeculator)

vllm_ascend/worker/v2/spec_decode/
└── eagle.py              # Ascend 特定的 Eagle 实现
```

### 1.2 核心类关系

```
SpecDecodeBaseProposer (基类)
├── EagleProposer / EagleSpeculator
│   └── AscendEagleSpeculator (vllm-ascend)
├── DraftModelProposer
├── MedusaProposer
├── NgramProposer
└── SuffixDecodingProposer
```

---

## 二、完整执行流程

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLMEngine.generate()                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Scheduler.schedule()                                │
│  - 决定每个 request 调度多少 tokens                              │
│  - 包含 scheduled_spec_decode_tokens (draft tokens)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Worker.execute_model()                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. prepare_inputs()                                     │    │
│  │     - 将 draft tokens 拼接到 input_ids 中                   │    │
│  │     - 构建 attention metadata                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  2. Model Forward                                        │    │
│  │     - TargetModel.forward(input_ids + draft_tokens)      │    │
│  │     - 输出 hidden_states, logits                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  3. Sample & Rejection                                   │    │
│  │     - _sample() 采样                                     │    │
│  │     - rejection_sample() 验证 draft tokens                │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  4. Propose Draft (为下一轮)                              │    │
│  │     - propose_draft_token_ids()                          │    │
│  │     - EagleProposer.propose(hidden_states)               │    │
│  │     - 生成下一轮的 draft tokens                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键代码调用链

```
# 入口
LLMEngine.generate()
  └─> Scheduler.schedule()
        └─> SchedulerOutput.scheduled_spec_decode_tokens (dict: req_id -> [draft_tokens])

# Worker 执行
Worker.execute_model()
  └─> GPUModelRunner.execute_model()
        ├─> prepare_inputs()  # 处理 draft tokens
        │   └─> input_ids = [prompt_tokens, sampled_token, draft_token_1, draft_token_2, ...]
        │
        ├─> _model_forward()  # Target Model Forward
        │   └─> hidden_states, logits = model(input_ids, positions, ...)
        │
        ├─> _sample()  # 采样 + 拒绝采样
        │   ├─> sampler_output = sampler.sample(logits, spec_decode_metadata)
        │   └─> sampled, num_rejected = rejection_sample(...)
        │
        └─> propose_draft_token_ids()  # 生成下一轮 draft tokens
            └─> drafter.propose(hidden_states, ...)
```

---

## 三、分阶段详细流程

### 3.1 第一次迭代（Prefill / 无 draft tokens）

```
SchedulerOutput:
  scheduled_spec_decode_tokens = {}  # 空，没有 draft tokens

GPUModelRunner.execute_model():
  1. prepare_inputs()
     - input_ids = [prompt_tokens]  # 只有 prompt
     - 无 draft tokens 需要拼接

  2. TargetModel.forward()
     - 处理整个 prompt
     - 输出：hidden_states (每个 token 一个), logits (最后一个 token)

  3. _sample()
     - sample(logits) -> sampled_token (第一个生成的 token)

  4. propose_draft_token_ids()
     - EagleProposer.propose(hidden_states=last_hidden_state)
     - 运行 Eagle 模型生成 K 个 draft tokens
     - draft_tokens = [d1, d2, d3, ..., dK]
     - 保存到 req_states.draft_tokens[req_id]

Scheduler 下一轮调度:
  scheduled_spec_decode_tokens[req_id] = [d1, d2, d3, ..., dK]
```

### 3.2 第二轮迭代（有 draft tokens）

```
SchedulerOutput:
  scheduled_spec_decode_tokens[req_id] = [d1, d2, d3]  # 上一轮生成的

GPUModelRunner.execute_model():
  1. prepare_inputs()
     - input_ids = [prompt_tokens, sampled_token, d1, d2, d3]
     - 注意：draft tokens 作为输入的一部分被送入 model

  2. TargetModel.forward()
     - 一次性处理所有 tokens (包括 draft tokens)
     - 输出：hidden_states (所有位置), logits (所有需要采样的位置)
     - spec_decode_metadata 记录哪些位置对应 draft tokens

  3. _sample() + rejection_sample()
     - 获取 target model 对 draft token 位置的 logits
     - 比较：target_sampled_token vs draft_token
     - 接受或拒绝每个 draft token
     - 输出：accepted_tokens, num_rejected

  4. propose_draft_token_ids()
     - 使用本轮 target model 输出的 hidden_states
     - EagleProposer.propose(hidden_states, ...)
     - 生成新一轮的 draft tokens (用于第三轮)
```

### 3.3 第三轮及以后

流程与第二轮相同，形成流水线：
```
每一轮：
  1. TargetModel.forward(上一轮 accepted_token + 上一轮 draft_tokens)
  2. RejectionSample(验证 draft tokens)
  3. 输出 accepted tokens
  4. EagleProposer.propose() -> 生成下一轮 draft tokens
```

---

## 四、关键代码分析

### 4.1 GPUModelRunner.execute_model() 关键片段

位置：`vllm/v1/worker/gpu_model_runner.py`

```python
# 行 3613-3619: Target Model Forward
model_output = self._model_forward(
    input_ids=input_ids,  # 包含 draft tokens
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)

# 行 3622-3649: 处理输出
if self.use_aux_hidden_state_outputs:
    hidden_states, aux_hidden_states = model_output  # EAGLE 3
else:
    hidden_states = model_output
    aux_hidden_states = None

sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)

# 行 3740: 采样 + 拒绝采样
sampler_output = self._sample(logits, spec_decode_metadata)

# 行 3759-3773: 生成下一轮 draft tokens
def propose_draft_token_ids(sampled_token_ids):
    self._draft_token_ids = self.propose_draft_token_ids(
        scheduler_output,
        sampled_token_ids,
        self.input_batch.sampling_metadata,
        hidden_states,  # 使用当前轮的 hidden_states
        sample_hidden_states,
        aux_hidden_states,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        slot_mappings,
    )

# 行 3796: 对于 EAGLE/DraftModel，可以立即生成 draft tokens
if input_fits_in_drafter:
    propose_draft_token_ids(sampled_token_ids)
```

### 4.2 propose_draft_token_ids() 关键片段

位置：`vllm/v1/worker/gpu_model_runner.py` 行 4024-4123

```python
def propose_draft_token_ids(
    self,
    scheduler_output: "SchedulerOutput",
    sampled_token_ids: torch.Tensor | list[list[int]],
    sampling_metadata: SamplingMetadata,
    hidden_states: torch.Tensor,
    sample_hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor] | None,
    spec_decode_metadata: SpecDecodeMetadata | None,
    common_attn_metadata: CommonAttentionMetadata,
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
) -> list[list[int]] | torch.Tensor:

    # ... 处理 ngram, suffix, medusa 等情况 ...

    # EAGLE / DraftModel 情况 (行 4124-4200+)
    elif spec_config.use_eagle() or spec_config.uses_draft_model():
        assert isinstance(self.drafter, EagleProposer | DraftModelProposer)

        # 准备输入：使用上一轮的 sampled_token_ids
        next_token_ids, valid_sampled_tokens_count = (
            self.drafter.prepare_next_token_ids_padded(
                common_attn_metadata,
                sampled_token_ids,
                self.requests,
                self.input_batch,
                self.discard_request_mask.gpu,
            )
        )

        # 调用 EagleProposer.propose()
        # 关键：传入的是当前轮 target model 输出的 hidden_states
        draft_token_ids = self.drafter.propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            hidden_states,  # 当前轮的 hidden_states
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
        )

        return draft_token_ids
```

### 4.3 EagleSpeculator.propose() 关键片段

位置：`vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` 行 190-354

```python
@torch.inference_mode()
def propose(
    self,
    input_batch: InputBatch,
    attn_metadata: dict[str, Any],
    slot_mappings: dict[str, torch.Tensor],
    last_hidden_states: torch.Tensor,  # 来自 target model
    aux_hidden_states: list[torch.Tensor] | None,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
) -> torch.Tensor:

    # 使用 target model 输出的 hidden_states 作为 Eagle 的输入
    if aux_hidden_states:
        hidden_states = self.model.combine_hidden_states(
            torch.cat(aux_hidden_states, dim=-1)
        )
    else:
        hidden_states = last_hidden_states

    # 准备 Eagle 的输入 ids
    last_token_indices = prepare_eagle_inputs(...)

    # 第一步：运行 Eagle 模型 (prefill)
    last_hidden_states, hidden_states = self.run_model(
        num_tokens, attn_metadata, slot_mappings, ...
    )

    # 采样第一个 draft token
    logits = self.model.compute_logits(sample_hidden_states)
    draft_tokens = gumbel_sample(logits, ...)
    self.draft_tokens[:num_reqs, 0] = draft_tokens

    # 准备后续 decode 步骤的输入
    prepare_eagle_decode(...)

    # 后续步骤：生成剩余的 draft tokens (行 346-353)
    self.generate_draft(
        num_reqs, num_tokens_padded, attn_metadata, slot_mappings, ...
    )

    return self.draft_tokens[:num_reqs]
```

### 4.4 generate_draft() 关键片段

位置：`vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` 行 126-171

```python
def generate_draft(
    self,
    num_reqs: int,
    num_tokens_padded: int,
    attn_metadata: dict[str, Any] | None,
    slot_mappings: dict[str, torch.Tensor] | None,
    num_tokens_across_dp: torch.Tensor | None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> None:
    for step in range(1, self.num_speculative_steps):
        # 运行 Eagle 模型 (decode 模式)
        last_hidden_states, hidden_states = self.run_model(
            num_tokens_padded, attn_metadata, slot_mappings, ...
        )

        # 采样 draft token
        logits = self.model.compute_logits(last_hidden_states)
        draft_tokens = gumbel_sample(logits, ...)
        self.draft_tokens[:num_reqs, step] = draft_tokens

        # 更新输入用于下一步
        update_eagle_inputs(
            draft_tokens, hidden_states, self.input_buffers, self.hidden_states, ...
        )
```

### 4.5 Rejection Sampler

位置：`vllm/v1/worker/gpu/spec_decode/rejection_sample.py`

```python
def rejection_sample(
    target_sampled: torch.Tensor,  # target model 采样的 tokens
    input_ids: torch.Tensor,       # 包含 draft tokens 的输入
    cu_num_logits: torch.Tensor,   # cumulative number of logits
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    比较 target_sampled 和 draft_token:
    - 如果相同，接受
    - 如果不同，拒绝，并停止接受后续 tokens
    """
    # Triton kernel 实现高效并行比较
    _rejection_sample_kernel[...]
```

---

## 五、vllm-ascend 的差异

### 5.1 AscendEagleSpeculator 特点

位置：`vllm_ascend/spec_decode/eagle_proposer.py`

```python
class AscendEagleSpeculator(EagleSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # 保留 input_batch 引用，用于 Ascend attention metadata 构建
        self.input_batch: InputBatch | None = None

    def propose(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ):
        # 保存 input_batch 引用
        self.input_batch = input_batch

        # 使用 Ascend attention metadata 构建
        with build_attn_metadata_wrapper():
            return super().propose(...)

    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last_hidden_states, hidden_states = super().run_model(...)

        # Ascend 特定：更新 seq_lens_cpu
        if attn_metadata is not None:
            for attn_meta in attn_metadata.values():
                attn_meta.seq_lens = attn_meta.seq_lens + 1
                attn_meta.seq_len_list = attn_meta.seq_lens.tolist()

        return last_hidden_states, hidden_states
```

### 5.2 主要差异点

| 方面 | vllm (GPU) | vllm-ascend (NPU) |
|------|------------|-------------------|
| Attention Backend | CUDA Attention | Ascend Attention |
| Metadata 构建 | 标准方式 | 需要 seq_lens_cpu 等额外信息 |
| Cudagraph | 支持 | 部分支持，需要特殊处理 |
| 流程 | 相同 | 相同（继承 + 重写特定方法） |

---

## 六、关键数据结构

### 6.1 SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    # 每个 request 调度的 token 数
    num_scheduled_tokens: dict[str, int]

    # 每个 request 的 draft tokens (关键!)
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # 示例: {"req_0": [101, 102, 103], "req_1": [201, 202]}

    # 其他字段...
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
```

### 6.2 SpecDecodeMetadata

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor      # draft tokens 的 id
    num_draft_tokens: list[int]        # 每个 request 的 draft token 数量
    cu_num_draft_tokens: torch.Tensor  # cumulative num draft tokens
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor  # target logits 的位置
    bonus_logits_indices: torch.Tensor   # bonus logits 的位置
    logits_indices: torch.Tensor         # 所有 logits 的位置
```

### 6.3 req_states.draft_tokens

```python
# 在 RequestState 中保存
class RequestState:
    draft_tokens: torch.Tensor  # shape: [max_num_reqs, num_speculative_steps]
    # 示例：[[101, 102, 103, -1, -1], [201, 202, -1, -1, -1]]
    # -1 表示无效的 draft token
```

---

## 七、执行时序图（完整版）

```
时间 →

Round 1 (No draft tokens):
  Scheduler: scheduled_spec_decode_tokens = {}
  │
  ▼
  TargetModel.forward(prompt)
  │  输出：hidden_states[0:len(prompt)]
  ▼
  Sample() -> sampled_token (例如：token_id=100)
  │
  ▼
  EagleProposer.propose(hidden_states[last_pos])
  │  输出：draft_tokens = [101, 102, 103]
  │  保存到：req_states.draft_tokens[req_id] = [101, 102, 103]
  ▼
  返回：sampled_token=100

Round 2 (With draft tokens):
  Scheduler: scheduled_spec_decode_tokens[req_id] = [101, 102, 103]
  │
  ▼
  TargetModel.forward([sampled_token, 101, 102, 103])
  │  输出：hidden_states[所有位置]
  │        logits[所有采样位置]
  ▼
  RejectionSample(
    target_sampled = [100, 101, 102, 103],
    draft_tokens   = [    101, 102, 103]
  )
  │  比较：101==101 ✓, 102==102 ✓, 103!=103 ✗
  │  输出：accepted=[100, 101, 102], num_rejected=1
  ▼
  EagleProposer.propose(hidden_states[accepted 最后位置])
  │  输出：draft_tokens = [104, 105, 106] (用于 Round 3)
  ▼
  返回：accepted_tokens=[101, 102]

Round 3:
  Scheduler: scheduled_spec_decode_tokens[req_id] = [104, 105, 106]
  │
  ▼
  (重复 Round 2 流程...)
```

---

## 八、总结

### 8.1 核心要点

1. **流水线模式**：每一轮使用上一轮生成的 draft tokens，同时为下一轮生成新的 draft tokens

2. **propose() 调用时机**：
   - 在 **Target Model Forward 之后** 调用
   - 使用 **当前轮** 的 `hidden_states` 作为输入
   - 生成 **下一轮** 使用的 draft tokens

3. **draft tokens 传递**：
   - 通过 `req_states.draft_tokens` 保存
   - 通过 `SchedulerOutput.scheduled_spec_decode_tokens` 传递给下一轮

4. **vllm-ascend**：
   - 继承 vllm 的 EagleSpeculator
   - 重写 attention metadata 构建逻辑以适配 Ascend NPU
   - 核心流程与 vllm 相同

### 8.2 关键代码位置索引

| 功能 | vllm 文件路径 | 行号范围 |
|------|--------------|----------|
| ModelRunner 主流程 | vllm/v1/worker/gpu_model_runner.py | 3600-3900 |
| Propose Draft | vllm/v1/worker/gpu_model_runner.py | 4024-4200 |
| EagleSpeculator | vllm/v1/worker/gpu/spec_decode/eagle/speculator.py | 190-354 |
| Rejection Sample | vllm/v1/worker/gpu/spec_decode/rejection_sample.py | 1-71 |
| Ascend 适配 | vllm_ascend/spec_decode/eagle_proposer.py | 32-160 |

---

## 九、调试 Tips

1. **查看 scheduled_spec_decode_tokens**:
   ```python
   print(f"Draft tokens scheduled: {scheduler_output.scheduled_spec_decode_tokens}")
   ```

2. **查看 propose 输入输出**:
   ```python
   # 在 propose_draft_token_ids 中
   print(f"hidden_states shape: {hidden_states.shape}")
   print(f"Draft tokens generated: {draft_token_ids}")
   ```

3. **查看 rejection sample 结果**:
   ```python
   # 在 _sample 后
   print(f"Accepted: {sampler_output.sampled_token_ids}")
   print(f"Num rejected: {num_rejected}")
   ```
