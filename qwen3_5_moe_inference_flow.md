# Qwen3.5-MoE 35B 推理流程详解

## 概述

本文档以 batch=2 为例（包含 `"你好"` 和 `"你是谁"` 两个序列），详细描述 Qwen3.5-MoE 35B 大模型在 vLLM 中的完整推理流程。假设两段文本经 tokenizer 编码后分别为 `[14, 26]` 和 `[14, 37, 52]`，
组成 batch=2（序列长度 2 和 3）。vLLM 将它们展平为 `[14, 26, 14, 37, 52]`，
共 5 个 token，位置编码为 `[0, 1, 0, 1, 2]`。

**Qwen3.5-MoE 35B 的核心创新**：
- **混合注意力**：40 层中 **36 层使用 Gated Delta Net（GDN）线性注意力**，**4 层使用标准 Full Attention**，实现 O(n) 线性复杂度
- **大规模 MoE**：256 个专家，每 token 激活 top-8（稀疏度 **3.125%**），无稠密层
- **解耦注意力维度**：Full Attention 中 `head_dim=256`，Q 投影维度 = `num_heads × head_dim = 16 × 256 = 4096`，KV 投影 = `num_kv_heads × head_dim = 2 × 256 = 512`
- **attn_output_gate**：Full Attention 输出额外乘以 `sigmoid(gate)`，增强门控能力

> 💡 **线性注意力 vs Full Attention 对比**
>
> | 维度 | Full Attention（4 层） | GDN Linear Attention（36 层） |
> |------|----------------------|------------------------------|
> | 复杂度 | O(n²) | O(n) |
> | 状态 | KV Cache（Key-Value 缓存） | 卷积状态 + SSM 状态（固定大小） |
> | 位置编码 | RoPE（旋转位置编码） | 隐式位置（卷积 + 递归状态传递） |
> | 实现 | Qwen3NextAttention (PagedAttention) | QwenGatedDeltaNetAttention |
> | 每层参数 | QKV+Gate + O ≈ 18.9M + 8.4M ≈ 27.3M | in_proj_qkvz + in_proj_ba + conv + out ≈ 25.2M + 0.13M + 0.03M + 8.4M ≈ 33.7M |

**文档基于 Qwen3.5-MoE 35B 配置**：
```json5
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "layer_types": ["linear_attention", ..., "full_attention"],  // 36 L + 4 F
    "full_attention_interval": 4,    // 每 4 层插入一个 Full Attention
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 248320,
    "attn_output_gate": true,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000
    }
  }
}
```

### Layer Types 分布

`full_attention_interval=4` 意味着每 4 层有一个 Full Attention 层，其余为 Linear Attention：

```
Layer  0: linear_attention  (GDN)
Layer  1: linear_attention  (GDN)
Layer  2: linear_attention  (GDN)
Layer  3: full_attention    ← 第 4 层
Layer  4: linear_attention  (GDN)
...
Layer 39: full_attention    ← 第 40 层
总计: 36 层 GDN + 4 层 Full Attention = 40 层
```

> 📐 **配置字段分类说明**

**不影响推理的字段（对应功能未启用）**：
- `moe_intermediate_size=512` — 在 35B 中是实际使用的，所有 40 层均为 MoE
- `shared_expert_intermediate_size=512` — **35B 配置中启用**了共享专家

**训练专用字段**：
- `initializer_range`, `router_aux_loss_coef`, `output_router_logits` — 仅训练时使用

**元数据字段**：
- `architectures`, `model_type`, `transformers_version` — 框架匹配用

> 📏 **关键数学关系**

**Full Attention 维度链**：
```
hidden_size=2048 → Q+Gate 投影 [2048→8192] → 16 heads × (256 + 256) dim  (Q + gate)
                 → K/V 投影 [2048→512]      → 2 KV heads × 256 dim
                 → QKV 合并输出: (16×2 + 2 + 2) × 256 = 9216 (Q + gate + K + V)
```

**GDN Linear Attention 维度链**：
```
hidden_size=2048 → in_proj_qkvz [2048→(2048 + 2048 + 4096 + 4096)] = [2048→12288]  (q + k + v + z)
                 → in_proj_ba   [2048→(32 + 32)] = [2048→64]                     (b + a)
                 → conv1d: 1D 因果卷积 (kernel_size=4) 作用于 qkv 部分 (8192 维)
                 → GDN 递归注意力
                 → out_proj [value_dim→2048] = [4096→2048]
```

**GQA 比率（Full Attention）**：16 Q heads ÷ 2 KV heads = **8**

**MoE 计算量等价**：`top_k × moe_intermediate_size = 8 × 512 = 4096`

**稀疏性**：`num_experts_per_tok ÷ num_experts = 8 ÷ 256` = **3.125%**

**参数量速算**（BF16 内存 = 参数量 × 2 字节）：

| 组件 | 公式 | 参数量 | BF16 内存 |
|------|------|--------|-----------|
| Embedding (×1) | vocab × hidden = 248320 × 2048 | **508M** | 1016 MB |
| LM Head (×1) | hidden × vocab = 2048 × 248320 | **508M** | 1016 MB |
| Full Attn (×4) | (QKV+Gate+O) × hidden = (8192+512+512+4096) × 2048 | **≈27.3M/层** | 54.6 MB/层 |
| GDN Attn (×36) | in_proj_qkvz + in_proj_ba + conv + out = 12288×2048 + 64×2048 + 8192×4 + 4096×2048 | **≈33.7M/层** | 67.4 MB/层 |
| MoE 专家 (×40) | E × 3 × hidden × moe_inter = 256 × 3 × 2048 × 512 | **805M/层** | 1.61 GB/层 |
| MoE 激活 (每 token) | k × 3 × hidden × moe_inter = 8 × 3 × 2048 × 512 | **25.2M** | — |
| 共享专家 (×40) | 3 × hidden × shared_inter = 3 × 2048 × 512 | **3.1M/层** | 6.2 MB/层 |
| Router (×40) | hidden × E = 2048 × 256 | **0.52M/层** | 1 MB/层 |
| RMSNorm (×81) | hidden × 81 = 2048 × 81 | **0.17M 总计** | 0.34 MB |
| **模型总计** | — | **≈ 35B** | **≈ 70 GB** |

> 💡 **与 Qwen3 MoE 30A3B 的关键差异**
>
> | 特性 | Qwen3 MoE 30A3B | Qwen3.5 MoE 35B |
> |------|----------------|-----------------|
> | 注意力 | 全部 Full Attention (GQA 32Q/4KV) | 混合: 36 GDN + 4 Full (16Q/2KV) |
> | 专家数 | 128 | **256** (翻倍) |
> | 激活专家 | 8 | 8 |
> | 稀疏性 | 6.25% | **3.125%** (更低) |
> | 共享专家 | 无 | **有** (shared_inter=512) |
> | head_dim | 128 | **256** (Full) / **128** (GDN) |
> | attn_output_gate | 无 | **有** (Full Attention 输出门控) |
> | 总层数 | 48 | 40 |

---

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Qwen3.5-MoE 35B 推理流程                                      │
│             输入：["你好", "你是谁"] → tokens: [14,26,14,37,52]                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  阶段 1: Token 嵌入 (Token Embedding)                                            │
│  输入：[14, 26, 14, 37, 52] → VocabParallelEmbedding → 输出：[5, 2048]            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  阶段 2: Transformer Decoder Layers (40 层混合)                                    │
│                                                                                   │
│  ┌─ Layer 0 (GDN Linear Attention) ──────────────────────────────────────────┐   │
│  │  RMSNorm → QwenGatedDeltaNetAttention → RMSNorm → MoE(256E, top-8)       │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│  ┌─ Layer 1 (GDN Linear Attention) ──────────────────────────────────────────┐   │
│  │  RMSNorm → QwenGatedDeltaNetAttention → RMSNorm → MoE(256E, top-8)       │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│  ┌─ Layer 2 (GDN Linear Attention) ──────────────────────────────────────────┐   │
│  │  RMSNorm → QwenGatedDeltaNetAttention → RMSNorm → MoE(256E, top-8)       │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│  ┌─ Layer 3 (Full Attention) ────────────────────────────────────────────────┐   │
│  │  RMSNorm → Qwen3NextAttention(GQA, RoPE, gate) → RMSNorm → MoE(256E,t8)  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│  ...                                                                              │
│  ┌─ Layer 39 (Full Attention) ───────────────────────────────────────────────┐   │
│  │  RMSNorm → Qwen3NextAttention(GQA, RoPE, gate) → RMSNorm → MoE(256E,t8)  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  每层输出：[5, 2048]                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  阶段 3~7: 后续处理链                                                             │
│  RMSNorm → LM Head → LogitsProcessor → Sampler → Detokenizer                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 阶段 1: Token 嵌入

### 功能说明

将离散的 token IDs 转换为连续的向量表示。

```
input_ids: [14, 26, 14, 37, 52]   # 展平的 5 个 token
     │
     │  VocabParallelEmbedding(248320, 2048)
     ▼
embedded: [5, 2048]
```

| 属性 | 值 |
|------|-----|
| **算子名称** | `VocabParallelEmbedding` |
| **输入形状** | `[num_tokens]` = `[5]` (int64) |
| **输出形状** | `[num_tokens, hidden_size]` = `[5, 2048]` (bfloat16) |
| **参数量** | `vocab_size × hidden_size` = 248320 × 2048 ≈ 508M |
| **内存占用 (BF16)** | ≈ 1016 MB |

---

## 阶段 2: Transformer Decoder Layers (混合注意力结构)

### 功能说明

Qwen3.5-MoE 35B 的 Decoder Layer 采用 **混合注意力 + 统一 MoE** 结构：
- 所有 40 层均使用 MoE（无稠密层）
- 注意力类型由 `layer_types` 配置决定：
  - `"linear_attention"` → `QwenGatedDeltaNetAttention` (GDN)
  - `"full_attention"` → `Qwen3NextAttention`

```
# Qwen3_5DecoderLayer.__init__ (vllm/model_executor/models/qwen3_5.py:123)
layer_type = config.layer_types[layer_idx]

if layer_type == "linear_attention":
    self.linear_attn = QwenGatedDeltaNetAttention(config, ..., gqa_interleaved_layout=False)
elif layer_type == "full_attention":
    self.self_attn = Qwen3NextAttention(config, ...)

# MLP: 所有层均为 MoE
self.mlp = Qwen3NextSparseMoeBlock(...)  # 256 experts, top-8
```

### 单层计算流程

```
输入：hidden_states [5, 2048], residual [5, 2048] (第 2 层起)
          │
          ▼
    ┌─────────────────────────────┐
    │  2.1 输入层归一化 (RMSNorm)    │
    │  + 残差连接 (Pre-norm)        │
    └─────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  2.2 注意力 (根据 layer_type 选择)    │
    │  ┌─────────────────────────────┐   │
    │  │ linear_attention:            │   │
    │  │ QwenGatedDeltaNetAttention  │   │
    │  │ (卷积 + 递归 SSM)           │   │
    │  └─────────────────────────────┘   │
    │  ┌─────────────────────────────┐   │
    │  │ full_attention:              │   │
    │  │ Qwen3NextAttention          │   │
    │  │ (QKV Proj → QKNorm → RoPE   │   │
    │  │  → PagedAttention →         │   │
    │  │  sigmoid(gate) × Output)    │   │
    │  └─────────────────────────────┘   │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────┐
    │  2.3 注意力后归一化           │
    │  + 残差连接                  │
    └─────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────┐
    │  2.4 MoE (Qwen3NextSparseMoeBlock)│
    │  256 专家, top-8              │
    │  + 共享专家 Shared Expert     │
    └─────────────────────────────┘
          │
          ▼
输出：hidden_states [5, 2048], residual [5, 2048]
```

---

### 2.1 输入层归一化 (Input RMSNorm)

与 Qwen3 MoE 完全相同，使用 `GemmaRMSNorm`（`Qwen3_5RMSNorm`）。Qwen3.5 使用 `GemmaRMSNorm`，它与普通 RMSNorm 的**关键区别**是：权重初始化为全 0，计算时通过 `1.0 + weight` 获得有效缩放因子（相当于初始有效权重为 1.0，即不做缩放）。

```python
# vllm/model_executor/layers/layernorm.py (GemmaRMSNorm)
# 与普通 RMSNorm 的唯一区别：output = x * (1.0 + weight) / RMS，weight 从 0 开始学习
```

| 属性 | 值 |
|------|-----|
| **输入** | `hidden_states` [5, 2048] |
| **输出** | `normalized` [5, 2048] |
| **参数量** | `hidden_size` = 2048 |

---

### 2.2.a GDN Linear Attention (QwenGatedDeltaNetAttention)

Gated Delta Net (GDN) 是一种 **线性复杂度注意力替代方案**，通过带门控的递归（recurrent）注意力机制，实现 O(n) 的推理复杂度。

#### 总体流程

GDN 有三大子步骤：

```
hidden_states [5, 2048]
    │
    ├─ [Part 1] 输入投影 Input Projection
    │   ├─ in_proj_qkvz:  [2048 → (key_dim × 2 + value_dim × 2)] = [2048 → 12288]
    │   │   └─ 拆分为 q, k, v, z
    │   └─ in_proj_ba:    [2048 → (num_v_heads + num_v_heads)] = [2048 → 64]
    │       └─ 拆分为 b, a
    │
    ├─ [Part 2] Core Attention (Custom Op: qwen_gdn_attention_core)
    │   │  （内部包含 conv1d → fused_post_conv_prep → 递归 SSM）
    │   ├─ 1D 因果卷积 (causal_conv1d, kernel=4)
    │   ├─ 后处理: fused_post_conv_prep → q, k, v, g, beta
    │   └─ 递归 SSM: chunk_gated_delta_rule (prefill)
    │                / fused_sigmoid_gating_delta_rule_update (decode)
    │
    └─ [Part 3] 输出投影 Output Projection
        ├─ RMSNormGated(core_attn_out, z)   ← z 作为门控
        └─ out_proj [value_dim → 2048]
```

#### 维度详解

```
hidden_size = 2048
linear_num_key_heads   = 16   (16 K heads)
linear_num_value_heads = 32   (32 V heads)
linear_key_head_dim    = 128  (每 K head 维度)
linear_value_head_dim  = 128  (每 V head 维度)

key_dim  = num_key_heads × key_head_dim   = 16 × 128 = 2048
value_dim = num_value_heads × value_head_dim = 32 × 128 = 4096

QKV dim  = key_dim + key_dim + value_dim = 2048 + 2048 + 4096 = 8192
Z dim    = value_dim = 4096

in_proj_qkvz: 共 8192 + 4096 = 12288 → [5, 2048] → [5, 12288]
in_proj_ba:   共 32 + 32 = 64       → [5, 2048] → [5, 64]

TP 分片后（tp_size=1）：
  mixed_qkv = qkvz[:8192]    → [5, 8192]
  z         = qkvz[8192:]    → [5, 4096] → reshape → [5, 32, 128]
  b, a = ba.chunk(2)          → [5, 32], [5, 32]
```

> 💡 **Qwen3.5 vs Qwen3-Next 的 in_proj 布局差异**
>
> Qwen3.5 和 Qwen3-Next 的 GDN 层 checkpoint 权重布局不同：
> - **Qwen3-Next** (`gqa_interleaved_layout=True`)：q, k, v, z 在 GQA group 维度上交错排列，权重融合成一个输出 shard
> - **Qwen3.5** (`gqa_interleaved_layout=False`)：q, k, v, z 各自独立，in_proj_qkvz 用 4 个独立的 output_sizes `[key_dim, key_dim, value_dim, value_dim]` 创建 `MergedColumnParallelLinear`
>
> 代码路径：`QwenGatedDeltaNetAttention.create_qkvz_proj()` (qwen_gdn_linear_attn.py:566)

#### Part 2: Core Attention 详解

Core Attention 是 GDN 的核心计算部分，由自定义 CUDA kernel `qwen_gdn_attention_core` 驱动。conv1d 在该 Custom Op 内部执行（`forward_cuda` 直接传入未卷积的 `mixed_qkv`）。核心流程包含：

```
mixed_qkv [5, 8192], b [5, 32], a [5, 32]
    │
    ├─ 1. Conv1D 变换 (kernel=4, 在 custom op 内部)
    │   conv1d.weight: [8192, 1, 4] (1D 深度可分离卷积)
    │   conv1d.bias:   [8192]
    │   输入 [5, 8192] → 因果卷积 → [5, 8192]
    │
    ├─ 2. fused_post_conv_prep（在 custom op 内部）
    │   从卷积输出中提取 q [5, 16, 128], k [5, 16, 128], v [5, 32, 128]
    │   生成 g (门控): f(a + dt_bias) → g = -exp(A_log) × softplus(a + dt_bias)
    │   生成 beta: sigmoid(b)
    │   L2 Norm 应用于 q, k
    │
    └─ 3. GDN 递归注意力 (选择 prefill 或 decode 路径)

    [Prefill 路径] chunk_gated_delta_rule:
      将序列分块, 块内并行, 块间递归：
      - 每块内: 用 chunked 方式计算 Q @ K^T 和加权 V
      - 块间: 用递归状态传递 (SSM state)
      支持 3 种后端: Triton/FLA (默认), FlashInfer (SM90), CuteDSL (Blackwell)

    [Decode 路径] fused_sigmoid_gating_delta_rule_update:
      逐 token 递归更新：
      state_new = g × state + beta × k^T @ v    (token-by-token)
      output = q @ state
      对单 token decode 场景触发快速路径
```

#### GDN 递归更新公式

GDN 的核心是带门控的 Delta Rule：

```
# 输入: q, k, v (均经过 L2 Norm), g (门控), beta (遗忘因子)
# 状态: S (SSM state, shape [num_v_heads, head_v_dim, head_k_dim])

# 状态更新（逐 token）:
S_t = exp(g_t) × S_{t-1} + sigmoid(b_t) × k_t^T @ v_t
                              ↑ beta          ↑ delta rule

# 输出:
o_t = q_t @ S_t
```

与标准 Mamba 的区别：
- **Delta Rule**：用 `k_t^T @ v_t` 替代了 Mamba 的 `diag(Δ) @ C^T @ v`
- **双门控**：`g`（指数门控）控制状态衰减速率，`beta = sigmoid(b)` 控制新信息注入比例
- **L2 Norm**：q 和 k 在计算前经过 L2 归一化，稳定训练

#### Part 3: 输出投影

```
core_attn_out [5, 32, 128]    ← GDN 递归注意力输出
z [5, 32, 128]                ← 从 in_proj_qkvz 分出

1. RMSNormGated:
    按 head 维度做 RMSNorm，再用 silu(z) 逐元素门控
    output = RMSNorm(core_attn_out) * silu(z)  # z 作为输出 gate

2. out_proj:
   [5, 4096] (flatten) → RowParallelLinear(4096→2048) → [5, 2048]
```

> 💡 **RMSNormGated 与普通 RMSNorm 的区别**
>
> RMSNormGated 额外接受一个 gate 输入 z，在归一化后对输出进行 `silu(z)` 逐元素门控：
> `output = RMSNorm(x) * silu(z)`（`norm_before_gate=True`）。
> 相当于给每个 head 的每个维度一个可学习的非线性门控权重。

#### Conv1D 状态与 SSM 状态

GDN Linear Attention 维护两种状态（存于 KV Cache 中）：

| 状态 | 形状 | 作用 | 更新方式 |
|------|------|------|---------|
| conv_state | `[conv_dim, kernel-1+num_spec]`（DS 布局）或 `[kernel-1+num_spec, conv_dim]`（SD 布局后 transpose） | 1D 因果卷积的滑动窗口 | 每次 decode 覆盖旧值，per-sequence 通过 state_indices 索引 |
| ssm_state | `[num_blocks, num_v_heads, head_v_dim, head_k_dim]`（float32） | GDN 递归状态，按 block 索引 | 递归更新（in-place，通过 state_indices 索引） |

对于 batch=2, [2,3] 的例子，prefill 阶段并行计算所有 5 个 token 的 conv + SSM，得到最终状态；decode 阶段每步只处理 2 个新 token，用 conv1d_update + ssm_update 逐 token 更新状态。

| 属性 | 值 |
|------|-----|
| **输入** | `hidden_states` [5, 2048] |
| **输出** | `attn_output` [5, 2048] |
| **in_proj_qkvz 参数量** | 2048 × (2048+2048+4096+4096) = 2048 × 12288 ≈ **25.2M** |
| **in_proj_ba 参数量** | 2048 × 64 ≈ **0.13M** |
| **conv1d 参数量** | 8192 × 4 ≈ **32.8K** |
| **out_proj 参数量** | 4096 × 2048 ≈ **8.4M** |
| **总计 (BF16)** | ≈ 33.7M 参数, ≈ 67.4 MB |

---

### 2.2.b Full Attention (Qwen3NextAttention)

每 4 层插入一层的标准 Transformer Attention，使用 GQA + RoPE + 输出门控。

#### QKV 投影（含 attn_output_gate）

```
hidden_size = 2048
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256

Q 带 gate 投影维度 = num_heads × (1 + attn_output_gate) × head_dim = 16 × 2 × 256 = 8192
K 投影维度 = num_kv_heads × head_dim = 2 × 256 = 512
V 投影维度 = num_kv_heads × head_dim = 2 × 256 = 512

QKV 总维度 = 8192 + 512 + 512 = 9216
```

QKVParallelLinear 的 `num_heads` 参数接收 `total_num_heads * (1 + attn_output_gate)` = 32，控制 Q 维度翻倍。

```python
# Qwen3NextAttention.__init__ (qwen3_next.py:244)
self.qkv_proj = QKVParallelLinear(
    config.hidden_size,                                # 2048
    self.head_dim,                                     # 256
    self.total_num_heads * (1 + self.attn_output_gate), # 16 × 2 = 32 → Q 输出 32×256=8192
    self.total_num_kv_heads,                           # 2 → K/V 各 2×256=512
    bias=False,
)
```

#### QK 归一化 + RoPE

```
qkv [5, 9216] → split → q_gate [5, 8192], k [5, 512], v [5, 512]

q_gate → reshape [5, 16, 512] → chunk → q [5, 16, 256], gate [5, 16, 256]
                                                           ↓
k [5, 512] → reshape [5, 2, 256]        → QK RMSNorm (per-head, dim=256)
q [5, 16, 256] → reshape [5, 16, 256]   → QK RMSNorm (per-head, dim=256)
                                           ↓
                                          RoPE (partial_rotary_factor=0.25)
                                          仅旋转每 head 的前 64 维 (256 × 0.25)
```

> 💡 **Fused QK Norm + RoPE + Gate**
>
> 当条件满足（CUDA, Neox-style RoPE, attn_output_gate=True, 纯文本）时，
> vLLM 使用 fused Triton kernel `fused_qk_rmsnorm_rope_gate` 一次完成
> QK 拆分 → RMSNorm → RoPE → gate 提取，减少 kernel launch 开销。
>
> 核函数内部：
> ```
> 输入 q_gate [5, 8192], k [5, 512]
> → 拆分为 q [5, 16, 256], gate [5, 16, 256]
> → q ÷ RMS(q) × (1 + q_norm_weight), k ÷ RMS(k) × (1 + k_norm_weight)
> → 前 64 维应用 RoPE 旋转
> → 输出 q, k (flat), gate (flat)
> ```

#### 注意力计算 + 输出门控

```
q [5, 4096], k [5, 512], v [5, 512]

# GQA: 2 KV heads → repeat_interleave(8) → 16 Q heads
k [5, 2, 256] → k_exp [5, 16, 256]
v [5, 2, 256] → v_exp [5, 16, 256]

# PagedAttention: Q @ K^T / sqrt(256) → softmax → × V
attn_output [5, 4096]

# 输出门控: attn_output × sigmoid(gate)
# gate [5, 4096] 经过 sigmoid 变成 (0,1) 的门控值
# 逐元素相乘: 大值保留，小值抑制
attn_output = attn_output × torch.sigmoid(gate)

# O 投影: [5, 4096] → RowParallelLinear(4096→2048) → [5, 2048]
```

#### RoPE 参数特点

```
rope_theta = 10,000,000  (比 Qwen3 MoE 的 1,000,000 大 10 倍)
partial_rotary_factor = 0.25  (只旋转每 head 前 64 维)

mrope_section = [11, 11, 10]  (MRoPE: 3D 旋转，用于视觉)
mrope_interleaved = True
```

| 属性 | 值 |
|------|-----|
| **Q+Gate 权重** | `[8192, 2048]` = 8192 × 2048 ≈ **16.8M** |
| **K 权重** | `[512, 2048]` = 512 × 2048 ≈ **1.0M** |
| **V 权重** | `[512, 2048]` = 512 × 2048 ≈ **1.0M** |
| **O 权重** | `[2048, 4096]` ≈ **8.4M** |
| **QKV 总计** | 16.8M + 1.0M + 1.0M = **18.8M** |
| **全层总计** | 18.8M + 8.4M = **27.2M** |

---

### 2.3 Post-Attention RMSNorm

与输入层归一化相同的 `GemmaRMSNorm`，负责残差融合 + 归一化。

```python
# Qwen3_5DecoderLayer.forward (继承自 Qwen3NextDecoderLayer.forward)
# === Attention 子层 ===
if residual is None:
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
else:
    hidden_states, residual = input_layernorm(hidden_states, residual)
# → self_attn or linear_attn ...

# === MLP 子层 ===
hidden_states, residual = post_attention_layernorm(hidden_states, residual)
hidden_states = mlp(hidden_states)

return hidden_states, residual
```

| 属性 | 值 |
|------|-----|
| **输入** | `hidden_states` [5, 2048], `residual` [5, 2048] |
| **输出** | `normalized` [5, 2048], `new_residual` [5, 2048] |
| **参数量** | `hidden_size` = 2048 |

---

### 2.4 Qwen3NextSparseMoeBlock (MoE)

这是 Qwen3.5-MoE 的 FFN 组件，所有 40 层均为 MoE 结构。
相比 Qwen3 MoE (128 专家)，Qwen3.5-MoE 将专家数翻倍到 **256**，同时新增了**共享专家**机制。

```
self.gate = ReplicatedLinear(2048, 256, bias=False)    # 路由器

self.shared_expert = Qwen3NextMLP(                      # 共享专家
    hidden_size=2048, intermediate_size=512,
    hidden_act="silu",
    expert_gate=self.shared_expert_gate,                # 共享专家门控
)

self.experts = FusedMoE(                              # 工厂函数，返回 MoERunner
    gate=self.gate,
    shared_experts=self.shared_expert,                  # 传共享专家给 MoERunner 管理
    num_experts=256,
    top_k=8,
    hidden_size=2048,
    intermediate_size=512,
    renormalize=True,
    ...
)
```

#### 2.4.a 路由器 (Gate)

```python
router_logits = gate(hidden_states)  # [5, 2048] → [5, 256]
```

| 属性 | 值 |
|------|-----|
| **权重** | `[256, 2048]` = 2048 × 256 ≈ **0.52M 参数** |
| **输入** | `hidden_states [5, 2048]` |
| **输出** | `router_logits [5, 256]` |

#### 2.4.b Top-K 选择

```python
# FusedMoE 内部: ops.topk_softmax(gating_output)
# → softmax([5, 256]) → topk(k=8) → renormalize
topk_weights [5, 8], topk_ids [5, 8]
```

#### 2.4.c 专家 MLP

每个专家是 SwiGLU MLP，gate_up_proj 合并为一个矩阵：

```
专家 MLP:
  gate_up_proj: [2048 → 1024] (gate: 512, up: 512)
    → SiLU(gate) × up → [512]
  down_proj: [512 → 2048]
    → output [2048]
```

融合 kernel (`fused_moe_kernel`) 并行计算所有被激活的专家，输出加权求和结果。

#### 2.4.d 共享专家

共享专家在 Qwen3NextMLP 内部完成门控：

```python
# Qwen2MoeMLP.forward (qwen2_moe.py:114)
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)    # [2048→1024], gate+up 各 512
    out = self.act_fn(gate_up)           # SiLU(gate) × up → [512]
    out, _ = self.down_proj(out)         # [512→2048]
    if self.expert_gate is not None:
        out = F.sigmoid(self.expert_gate(x)[0]) * out   # 标量门控 × 输出
    return out

# self.expert_gate: ReplicatedLinear(2048, 1) → [batch, 1]
```

对应维度：`gate_up_proj [2048→1024]`, `down_proj [512→2048]`, `shared_expert_gate [2048→1]`。

MoE Runner 最终将路由专家和共享专家输出直接相加：

```python
# moe_runner.py:707-708
result = shared_output + fused_output
```

共享专家在每层都参与计算（所有 token 都经过），可以理解为一个"公共知识"通道。

| 属性 | 值 |
|------|-----|
| **总专家数** | 256（路由）+ 1（共享） |
| **每 token 激活路由专家** | 8 (top_k) |
| **稀疏性** | 8/256 = **3.125%** |
| **路由器参数** | 2048 × 256 ≈ **0.52M** |
| **路由专家参数（单层）** | 256 × 3 × 2048 × 512 ≈ **805M** |
| **激活参数（每 token）** | 8 × 3 × 2048 × 512 ≈ **25.2M** |
| **共享专家参数（单层）** | 3 × 2048 × 512 ≈ **3.1M** |
| **单层总参数** | 805M + 3.1M + 0.52M ≈ **809M** |

> 💡 **为什么使用共享专家？**
>
> 共享专家是 MoE 架构的常见优化：让所有 token 都经过一个小型 MLP（容量小但通用），
> 路由器只需要负责将 token 分配到"差异化的路由专家"。
> 这样可以降低路由器的负载（不需要每个 token 都从 256 个专家中找到完全正确的一个），
> 同时确保所有 token 都获得基本的 FFN 处理。Qwen3 MoE 30A3B 没有共享专家，而 35B 版本加入了。

---

### 阶段 2 总结：40 层混合结构的完整迭代

```
配置: layer_types=[L,L,L,F, L,L,L,F, ..., L,L,L,F], 共 40 层
      所有层均使用 MoE (256 专家, top-8)

Layer 0 (GDN):  RMSNorm → GDN_Attn → RMSNorm → MoE
Layer 1 (GDN):  RMSNorm → GDN_Attn → RMSNorm → MoE
Layer 2 (GDN):  RMSNorm → GDN_Attn → RMSNorm → MoE
Layer 3 (Full): RMSNorm → Full_Attn(GQA,RoPE,gate) → RMSNorm → MoE  ← 每 4 层
Layer 4 (GDN):  ...
...
Layer 39 (Full): RMSNorm → Full_Attn → RMSNorm → MoE

最终输出：hidden_states [5, 2048]

# 统计:
# - GDN Linear Attention 层: 36 层 (90%)
# - Full Attention 层: 4 层 (10%)
# - 全部 40 层均为 MoE (无稠密层)
# - 总路由专家: 256
# - 每 token 激活路由专家: 8
# - 稀疏性: 3.125%
```

---

## 阶段 3~7: 后续处理链

| 阶段 | 输入 → 输出 | 参数量 | 功能 |
|------|-----------|--------|------|
| **3. 最终 RMSNorm** | `[5, 2048]` → `[5, 2048]` | 2048 | 最后一层归一化 |
| **4. LM Head** | `[5, 2048]` → `[5, 248320]` | 508M | 线性投影到词表维度 |
| **5. Logits 处理** | `[5, 248320]` → `[2, 248320]` | 0 | 取每序列末位 logits |
| **6. 采样** | `[2, 248320]` → `token_ids` | 0 | 每序列采样一个 token |
| **7. Token 解码** | `token_ids` → `text` | 0 | token ID → 文本 |

---

## 关键算子详解

### 算子汇总表

| 算子名称 | 功能 | 输入形状 | 输出形状 | 参数量 | 内存 (BF16) |
|---------|------|---------|---------|--------|------------|
| **VocabParallelEmbedding** | 词嵌入查找 | `[N]` | `[N, H]` | 508M | 1016 MB |
| **RMSNorm** (Gemma) | 均方根归一化 | `[N, H]` | `[N, H]` | H | 4 KB |
| **QKVParallelLinear** (Full) | QKV+Gate 投影 | `[N, H]` | `[N, 9216]` (tp=1) | 18.8M | 37.6 MB |
| **GDN in_proj** | QKVZ+BA 投影 | `[N, H]` | `[N, 12288]`, `[N, 64]` | 25.3M | 50.6 MB |
| **RowParallelLinear** | O 投影 (Full/GDN) | `[N, N_q×D]` / `[N, value_dim]` | `[N, H]` | 8.4M | 16.8 MB |
| **ReplicatedLinear** | 路由器 | `[N, H]` | `[N, E]` | 0.52M | 1 MB |
| **FusedMoE** | 融合 MoE | `[N, H]` | `[N, H]` | 805M | 1.61 GB |
| **ParallelLMHead** | LM 头 | `[N, H]` | `[N, V]` | 508M | 1016 MB |

**符号说明：**
- N = num_tokens
- H = hidden_size (2048)
- E = num_experts (256)
- V = vocab_size (248320)
- N_q = query heads (16)
- N_kv = key-value heads (2)
- D = head_dim (256, Full Attention)

---

## 代码调用栈

```python
# 用户调用
output = llm.generate("你好", sampling_params)

# 内部调用栈
└── TokenExecutor.execute_model()
    └── GPUModelRunner.execute_model()
        ├── _model_forward(input_ids, positions)
        │   └── Qwen3_5MoeForConditionalGeneration.forward()
        │       └── Qwen3_5Model.forward()
        │           ├── embed_input_ids(input_ids)
        │           │   └── VocabParallelEmbedding
        │           └── [Qwen3_5DecoderLayer × 40]
        │               ├── input_layernorm (GemmaRMSNorm)   # Pre-norm
        │               ├── self_attn (根据 layer_type 选择)
        │               │   ├── [GDN] QwenGatedDeltaNetAttention
        │               │   │   ├── in_proj_qkvz (MergedColumnParallelLinear)
        │               │   │   ├── in_proj_ba (MergedColumnParallelLinear)
        │               │   │   ├── conv1d (1D 因果卷积)
        │               │   │   ├── chunk_gated_delta_rule
        │               │   │   │   └── (Triton / FlashInfer / CuteDSL)
        │               │   │   └── out_proj (RowParallelLinear)
        │               │   └── [Full] Qwen3NextAttention
        │               │       ├── qkv_proj (QKVParallelLinear)
        │               │       ├── q_norm, k_norm (GemmaRMSNorm)
        │               │       ├── rotary_emb (MRoPE)
        │               │       ├── paged_attn (PagedAttention)
        │               │       ├── sigmoid gate
        │               │       └── o_proj (RowParallelLinear)
        │               ├── post_attention_layernorm (GemmaRMSNorm)
        │               └── mlp (Qwen3NextSparseMoeBlock)
        │                   ├── gate (ReplicatedLinear) ← 路由器
        │                   ├── shared_expert_gate (ReplicatedLinear) ← 共享专家门控
        │                   ├── shared_expert (Qwen3NextMLP)
        │                   └── experts (FusedMoE)
        │                       ├── Top-K 专家选择 (top-8/256)
        │                       └── 融合专家计算
        │                   # shared_expert 输出与 routed 输出在 MoERunner 内相加
        │           └── norm (GemmaRMSNorm)              # 最终输出归一化
        └── sample_tokens(logits)
            └── Sampler()
                ├── temperature scaling
                ├── TopKTopPSampler
                ├── apply_penalties
                └── multinomial / argmax
```

---

## Qwen3.5-MoE 35B 核心创新总结

1. **混合注意力架构**：36 层 GDN 线性注意力 (O(n)) + 4 层 Full Attention (O(n²))
   - 长序列场景大幅降低 KV Cache 显存（GDN 状态固定，不随序列长度增长）
   - Full Attention 保留精确的位置建模能力（每 4 层穿插一层）

2. **大规模 MoE (256 专家)**：来自 Qwen3 MoE 128 专家的翻倍，稀疏度降至 **3.125%**，激活参数量不变（top-8），但总容量翻倍

3. **共享专家**：新增共享专家机制，确保所有 token 获得基础 FFN 处理，降低路由器压力

4. **attn_output_gate**：Full Attention 输出经过 `sigmoid(gate)` 门控，增强注意力输出的调节能力

5. **解耦注意力维度**：
   - Full Attention: `head_dim=256`, Q+Gate 投影 = 8192 (16 heads × 512)，纯 Q = 4096 (16 heads × 256)
   - GDN Attention: `head_k_dim=128`, 16 K heads, 32 V heads
   - 两种注意力使用不同的 head 配置，各取所长

### 与 Qwen3 MoE 30A3B 的全面对比

| 维度 | Qwen3 MoE 30A3B | Qwen3.5 MoE 35B |
|------|----------------|-----------------|
| **总层数** | 48 | 40 |
| **注意力** | 全 Full (32Q/4KV, GQA=8) | 36 GDN + 4 Full (16Q/2KV, GQA=8) |
| **head_dim** | 128 | 256 (Full) / 128 (GDN) |
| **RoPE theta** | 1,000,000 | 10,000,000 |
| **KV Cache (per token)** | 4×128×2B×48=49KB (Full Attn) | Full: 2×256×2B×4≈2KB + GDN状态≈2.1MB×36≈74MB (固定) |
| **专家数** | 128 | **256** |
| **激活专家** | 8 (6.25%) | 8 (**3.125%**) |
| **moe_intermediate** | 768 | 512 |
| **共享专家** | 无 | 有 (intermediate=512) |
| **激活参数量/层** | 37.7M | 25.2M (routed) + 3.1M (shared) |
| **模型总参数** | ≈ 30.5B | ≈ 35B |
| **词表大小** | 151936 | 248320 |
| **显存需求 (BF16)** | ≈ 61 GB | ≈ 70 GB |
| **序列复杂度** | O(n²) | O(n) (90% 层) |

> 💡 **Qwen3.5-MoE 35B 的设计哲学**
>
> 1. **注意力革命**：从传统二次复杂度注意力转向线性复杂度 GDN。36/40 (90%) 的层使用 GDN，
>    仅保留 4 层 Full Attention 确保位置敏感性。这使长上下文推理的显存占用从 O(n) 状态大小
>    （GDN 只占 KV Cache 极小部分）变为实际可用的 O(1) 每层。
>
> 2. **MoE 升级**：专家数从 128→256，但中间维度从 768→512，保持每 token 激活参数量不变。
>    相当于用更"窄"但更"多"的专家覆盖更细粒度的知识分布，配合共享专家确保基线质量。
>
> 3. **大词表**：词表从 151K→248K，增加了约 60% 的词汇量，改善多语言和编码能力。
>    代价是 Embedding + LM Head 的参数量增长到约 1B（2GB 显存）。
