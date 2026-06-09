# Qwen3MoeForCausalLM 推理流程详解

## 概述

本文档以 batch=2 为例（包含 `"你好"` 和 `"你是谁"` 两个序列），详细描述 Qwen3MoeForCausalLM 大模型在 vLLM 中的完整推理流程。假设两段文本经 tokenizer 编码后分别为 `[14, 26]` 和 `[14, 37, 52]`，
组成 batch=2（序列长度 2 和 3）。vLLM 将它们展平为 `[14, 26, 14, 37, 52]`，
共 5 个 token，位置编码为 `[0, 1, 0, 1, 2]`。

> 💡 **如果 batch=1, seq_len=5 呢？**
>
> 同样是 5 个 token，展平后的 tensor 形状 `[5, 2048]` 完全一样，
> 但**序列边界不同**，导致以下差异：
>
> | 维度 | batch=2, [2,3] | batch=1, [5] |
> |------|---------------|--------------|
> | **position IDs** | `[0,1,0,1,2]`（各序列独立编号） | `[0,1,2,3,4]`（统一编号） |
> | **注意力掩码** | 序列间**不可见**（token2 看不到 token0） | 标准因果掩码（token2 可以看到 token0） |
> | **末位 logits** | 取 2 个位置（每序列最后一个）→ `[2, V]` | 取 1 个位置 → `[1, V]` |
> | **采样输出** | 2 个 token | 1 个 token |
>
> 除此以外的矩阵乘法（Q @ K^T、FFN 等）在 tensor 层面**完全一致**，因为形状都是 `[5, 2048]`。

Qwen3MoE 是 Qwen3 的 **Mixture of Experts (MoE)** 变体，核心区别在于部分（或全部）Decoder Layer 使用稀疏 MoE 结构替代稠密 MLP，实现更高效的条件计算。

> 💡 **本文档基于 Qwen3MoE 30B 配置**，该配置 `decoder_sparse_step=1`、
> `mlp_only_layers=[]`，**全部 48 层均为 MoE**，无稠密层。
> 对于 `decoder_sparse_step > 1` 的其他变体，稠密层和稀疏层才会交替出现。

**文档基于 Qwen3MoE 30B 配置**：
```json
{
  "architectures": [
    "Qwen3MoeForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "decoder_sparse_step": 1,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "max_position_embeddings": 40960,
  "max_window_layers": 48,
  "mlp_only_layers": [],
  "model_type": "qwen3_moe",
  "moe_intermediate_size": 768,
  "norm_topk_prob": true,
  "num_attention_heads": 32,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 48,
  "num_key_value_heads": 4,
  "output_router_logits": false,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "router_aux_loss_coef": 0.001,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

### 架构设计特点：解耦的注意力维度

Qwen3MoE 30B 采用**解耦设计**，`hidden_size` 与 `num_attention_heads × head_dim` 不相等：

```
hidden_size = 2048（残差流维度）
num_attention_heads × head_dim = 32 × 128 = 4096（Q 投影维度）
num_kv_heads × head_dim = 4 × 128 = 512（KV 投影维度）
```

**注意力层的维度变换**：
```
输入：[batch, seq, 2048]
  │
  ├─→ Q 投影：[2048 → 4096] → [batch, seq, 32, 128]
  ├─→ K 投影：[2048 → 512]  → [batch, seq, 4, 128]
  └─→ V 投影：[2048 → 512]  → [batch, seq, 4, 128]
  │
  └─→ 注意力计算 (GQA: 32 Q heads vs 4 KV heads)
  │
  └─→ O 投影：[4096 → 2048] → 输出回残差流
```

**设计优势**：
- 提升注意力表达能力（更大的 Q 投影维度）
- 控制 KV Cache 显存（GQA 减少 KV 头数）
- 平衡 MoE 计算负载（FFN 是主要瓶颈）

---

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Qwen3MoeForCausalLM 推理流程                               │
│             输入：["你好", "你是谁"] → tokens: [14,26,14,37,52]                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  阶段 1: Token 嵌入 (Token Embedding)                                            │
│  输入：[14, 26, 14, 37, 52] → 查表 → 输出：[5, 2048] (5 tokens x 2048 维)          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  阶段 2: Transformer Decoder Layers                                               │
│  48× (RMSNorm → Attention → RMSNorm → MoE)         [30B: 全部 MoE]              │
│  输入：[5, 2048] → 输出：[5, 2048]                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  阶段 3~7: 后续处理链                                                             │
│  RMSNorm → LM Head → LogitsProcessor → Sampler → Detokenizer                   │
│  输入：[5, 2048] → 输出：["好", "谁"] (每序列一个 token)                                                  │
│  最终 RMSNorm → LM Head → 取各序列末位 logits → 每序列采样 → 解码               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 阶段 1: Token 嵌入

### 功能说明

将离散的 token IDs 转换为连续的向量表示，即查表（embedding lookup）。

```
input_ids: [14, 26, 14, 37, 52]   # 展平的 5 个 token
     │
     │  VocabParallelEmbedding(151936, 2048)
     ▼
embedded: [5, 2048]     ← 5 = token0~token4, 2048 = 每个 token 的向量维度

# 实际数据示例（每个 token 查表得到 2048 维向量）：
# token0  ("你"):  [0.12, -0.34, 0.56, -0.78, ..., 0.89]    ← 2048 个浮点数
# token1  ("好"):  [-0.78, 0.91, -0.23, 0.45, ..., 0.12]
# token2  ("你"):  [0.12, -0.34, 0.56, -0.78, ..., 0.89]    ← token0 和 token2 相同（都是"你"）
# token3  ("是"):  [0.67, -0.12, 0.34, -0.56, ..., 0.45]
# token4  ("谁"):  [0.89, -0.45, 0.23, 0.12, ..., -0.67]
```

### 简要说明

| 属性 | 值 |
|------|-----|
| **算子名称** | `VocabParallelEmbedding` |
| **输入形状** | `[num_tokens]` = `[5]` (int64, 展平后的 5 个 token) |
| **输出形状** | `[num_tokens, hidden_size]` = `[5, 2048]` (float16/bfloat16, 展平后) |
| **参数量** | `vocab_size × hidden_size` (151936 × 2048 ≈ 311M) |
| **内存占用 (BF16)** | `311M × 2 bytes` ≈ 622 MB |

---

## 阶段 2: Transformer Decoder Layers (混合结构)

### 功能说明

Qwen3MoE 的 Decoder Layer 采用 **混合结构**：
- **稠密层**：使用 `Qwen3MoeMLP`（标准 SwiGLU MLP）
- **稀疏层**：使用 `Qwen3MoeSparseMoeBlock`（MoE 结构）

层类型由配置决定：
```python
# vllm/model_executor/models/qwen3_moe.py
layer_idx = extract_layer_index(prefix)
mlp_only_layers = config.mlp_only_layers if hasattr(config, "mlp_only_layers") else []

if (layer_idx not in mlp_only_layers) and (
    config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
):
    # 稀疏层：使用 MoE
    self.mlp = Qwen3MoeSparseMoeBlock(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
else:
    # 稠密层：使用标准 MLP
    self.mlp = Qwen3MoeMLP(...)
```

**Qwen3MoE 30B 配置**（`decoder_sparse_step=1`, `mlp_only_layers=[]`）：
- Layer 0: 稀疏层 (MoE) ← 不在 mlp_only_layers 且 (0+1) % 1 == 0
- Layer 1: 稀疏层 (MoE) ← 不在 mlp_only_layers 且 (1+1) % 1 == 0
- Layer 2: 稀疏层 (MoE) ← 不在 mlp_only_layers 且 (2+1) % 1 == 0
- ...
- Layer 47: 稀疏层 (MoE) ← 所有 48 层都是 MoE

### 单层计算流程（通用）

```
输入：hidden_states [5, 2048]
          │
          ▼
    ┌─────────────────────────────┐
    │  2.1 输入层归一化 (RMSNorm)   │
    │  对输入做 RMS 归一化：        │
    │  x / sqrt(mean(x²) + ε) * γ │
    └─────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────┐
    │  2.2 自注意力                │
    │  (Qwen3MoeAttention)        │
    │  QKV投影 → QKNorm → RoPE →  │
    │  GQA注意力 → O投影            │
    └─────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────┐
    │  2.3 注意力后归一化           │
    │  (RMSNorm)                  │
    │  与输入层相同，对注意力输出归一化 │
    └─────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────┐
    │  2.4 MLP 块 (根据层类型选择)       │
    │  ┌─────────────────────────┐    │
    │  │ 稠密层：Qwen3MoeMLP      │    │
    │  └─────────────────────────┘    │
    │  ┌─────────────────────────┐    │
    │  │ 稀疏层：MoE (核心区别)     │    │
    │  └─────────────────────────┘    │
    └─────────────────────────────────┘
          │
          ▼
          │ 残差连接已嵌入 2.1 和 2.3 的 RMSNorm 中（非独立步骤）
          │
          ▼
输出：hidden_states [5, 2048]
```

---

### 2.1 输入层归一化 (Input RMSNorm)

对输入进行归一化，让数据的尺度保持稳定，防止深层网络中数值爆炸或消失。

#### 数学公式

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \cdot \gamma$$

#### 逐项解释

| 符号 | 含义 | 在本模型中的值 |
|------|------|-------------|
| $x$ | 输入的 2048 维向量（每个 token 一个） | token0 的 hidden = `[0.12, -0.34, 0.56, ..., 0.89]` |
| $d$ | 向量维度 | `hidden_size = 2048` |
| $\frac{1}{d}\sum x_i^2$ | 所有元素的**均方值**（RMS 的平方） | 比如 `mean([0.12², -0.34², ...]) ≈ 0.15` |
| $\epsilon$ | 防除零的极小常数 | `1 × 10⁻⁶`（约等于 0.000001） |
| $\gamma$ (gamma) | 可学习的**缩放向量**，与 $x$ 逐元素相乘 | shape `[2048]`，每个维度一个缩放因子 |

#### 计算过程（以 token0 为例）

```
原始 x (2048 维):  [0.12, -0.34, 0.56, -0.78, ..., 0.89]
                             ↓
第 1 步：计算 RMS
    mean(x²) = (0.12² + (-0.34)² + 0.56² + ...) / 2048
             ≈ 0.15
    RMS = sqrt(0.15 + 1e-6) ≈ 0.387
                             ↓
第 2 步：归一化
    x_normalized = x / 0.387
                 = [0.31, -0.88, 1.45, -2.02, ..., 2.30]
                             ↓
第 3 步：乘以可学习权重 γ
    γ = [0.98, 1.02, 0.95, 1.05, ..., 0.99]    ← 通过训练学出来的
    output = x_normalized ⊙ γ                   ← 逐元素相乘
           = [0.30, -0.90, 1.38, -2.12, ..., 2.28]
```

#### 为什么需要 $\epsilon$？

如果 $x$ 恰好全是 0（比如刚初始化的 token），均方值就是 0，除零会导致 NaN。$\epsilon$ 保证分母永远 `> 0`。

#### 为什么需要 $\gamma$？

归一化把数据拉到了均值为 0、RMS 为 1 的固定范围，但模型可能需要某些维度更重要。$\gamma$ 让网络**在归一化的基础上重新学习每个维度的合适缩放比例**，恢复表达能力。

#### $\gamma$ 从哪来？

$\gamma$ 是 **模型权重文件（checkpoint）的一部分**，随整个模型一起训练得到。每个 RMSNorm 层都有自己的 $\gamma$，存储为 `weight` 参数。

```
# 加载权重时（简化）
state_dict = torch.load("qwen3_moe_30b.pt")
layer_0_rmsnorm_weight = state_dict["model.layers.0.input_layernorm.weight"]
# shape: [2048]，就是这个 γ
```

Qwen3MoE 30B 共有 $48\ \text{(层)} \times 2\ \text{(每层两个 RMSNorm)} + 1\ \text{(最终 norm)} = 97$ 个 RMSNorm，每个存一份 $\gamma$（2048 个 float），
总计 $97 \times 2048 \times 2\text{ bytes} \approx 0.4\ \text{MB}$。

| 属性 | 值 |
|------|-----|
| **输入** | `hidden_states` [5, 2048] |
| **输出** | `normalized` [5, 2048] |
| **参数量** | `hidden_size` = 2048 |
| **内存占用 (BF16)** | `2048 × 2 bytes` = 4 KB |

---

### 2.2 自注意力 (Qwen3MoeAttention)

自注意力的核心思想：**每个 token 用"我的 query"去问所有 token 的 key，算出和谁关系大，再按关系权重取 value 做加权和**。

Qwen3MoE 采用 **GQA（Grouped Query Attention）**：32 个 Q head 共享 4 组 KV head。

#### 为什么要做投影？

hidden state（2048 维）是"浓缩"的语义表示，但注意力的计算需要三种不同角色。**每个 token 同时产生自己的 Q、K、V**：
- **Query（查询）**：当前 token 作为"提问者"的身份向量 → 用它去 match 所有人的 K
- **Key（键）**：当前 token 作为"被关注者"的标签向量 → 供所有人的 Q 来匹配
- **Value（值）**：当前 token 作为"信息源"的实际内容 → 按匹配权重汇总

用一个共享的 2048 维向量同时表达这三种角色不够灵活，所以用三个不同的**投影矩阵**（Q、K、V 权重）把同一条 hidden state 投射到不同的语义空间。

#### 为什么要解耦（Q 投影维度 > hidden_size）？

常规设计是 `hidden_size = num_heads × head_dim`，投影后维度不变。但 Qwen3MoE 用了**解耦设计**：

```
hidden_size(2048)    ← 压缩后的残差流
Q 投影维度(4096)      ← 展开，给注意力更多"视野"
KV 投影维度(512)      ← GQA 压缩，少存 K/V cache
```

Q 维度更大意味着每个 head 有更多参数来学习"问什么"，提升注意力表达能力。K/V 维度更小且用 GQA（4 个 KV head 服务 32 个 Q head）则大幅节省 KV cache 显存。

#### 为什么要 reshape 成多头形式？

Attention 的并行计算依赖于**多头独立计算**。投影出来的 Q 是 `[5, 4096]`，如果不 reshape，它就是 4096 维的平铺向量。reshape 成 `[5, 32, 128]` 后：
- 第 0 维：5 个 token
- 第 1 维：32 个独立的 head
- 第 2 维：每个 head 的 128 维向量

这样 32 个 head 可以**同时、独立**地做 dot product attention，互不干扰。

包含子步骤：
1. **QKV 投影** (`QKVParallelLinear`) — hidden_size=2048 → Q: 4096, K/V: 512
2. **QK 归一化** (`RMSNorm`, per-head) — 先 reshape 成 per-head 形状，逐 head 归一化，再 reshape 回
3. **旋转位置编码** (`RotaryEmbedding`) — 在 QK 归一化之后应用
4. **注意力计算** (`PagedAttention`) — GQA + 因果掩码
5. **输出投影** (`RowParallelLinear`) — 4096 → 2048 回到残差流

| 属性 | 值 |
|------|-----|
| **输入** | `hidden_states` [5, 2048], `positions` [5] |
| **输出** | `attn_output` [5, 2048] |
| **参数量** | ≈ 18.9M (QKV + O 投影) |
| **内存占用 (BF16)** | ≈ 37.8 MB |

> 💡 **Q 是 4096 维，为什么还要区分 32 heads × 128 dim？**
>
> 4096 不是随便的平铺——它是 **32 组 128 维的向量首尾拼起来的**：
> ```
> Q = [head0_0..127, head1_0..127, head2_0..127, ..., head31_0..127]
>      ├──── 128 ────┤├──── 128 ────┤├──── 128 ────┤     ├──── 128 ────┤
> ```
>
> 在注意力计算时，**每组 128 维独立做 dot product**，互不干扰。
> 32 个 head 并行算出 32 个注意力权重。
>
> 为什么不用一整个 4096 维做一次 dot product？
> - 一个 4096 维的 Q @ K^T 只产生 1 个注意力分数 → 模型只能关注**一种关系**
> - 32 个 128 维的 head 产生 32 个注意力分数 → 模型可以同时关注**32 种不同的关系**
>
> **不同 head 关注点不同是怎么来的？**
> 每个 head 有自己的独立权重矩阵。Q 权重 `[4096, 2048]` 被切成 32 份 `[128, 2048]`，
> 每份 head i 独享。32 份权重随机初始化 → 反向传播接收不同梯度 → 自然分化出不同关注模式。
>
> **计算时 head 是分开算还是合在一起算？**
> 逻辑上分开（head0 只看自己的 128 维），物理上合并——reshape 成 `[5, 32, 128]` 后，
> GPU 把 `32`（heads）当作 batch 维，用一次 batched GEMM 同时算完所有 head，
> 而不是 for 循环 32 次。

#### 2.2.a QKV 投影（解耦维度设计）

一句话：**每个 token 同时算出自己的 Q、K、V，分别扮演提问者、被关注者、信息源三种角色**。

```
输入: [5, 2048]      ← 5 个 token 的"浓缩语义"

Q 投影 [2048→4096] → [5, 4096] → reshape→ [5, 32, 128]
   每个 token 算出 32 组"提问向量" → 去匹配所有人的 K

K 投影 [2048→512]  → [5, 512]  → reshape→ [5, 4, 128]
   每个 token 算出 4 组"被关注标签" → 供所有人的 Q 来匹配

V 投影 [2048→512]  → [5, 512]  → reshape→ [5, 4, 128]
   每个 token 算出 4 组"信息内容" → 按匹配权重汇总
```

之后做 `Q @ K^T`，即**每个 token 用自己的 Q 去匹配所有 token（包括自己）的 K**，
匹配度高的 token 对最终的 value 加权和贡献更大。

**Qwen3MoE 30B 采用解耦设计**：
- `hidden_size = 2048`（残差流维度）
- `num_attention_heads × head_dim = 32 × 128 = 4096`（Q 投影维度）
- `num_kv_heads × head_dim = 4 × 128 = 512`（KV 投影维度）

**权重形状**：
```python
# QKV 投影权重 [output_dim, input_dim]
Q 权重：[4096, 2048]   # 2048 → 4096
K 权重：[512, 2048]    # 2048 → 512
V 权重：[512, 2048]    # 2048 → 512

# 合并的 QKV 权重（物理上三个矩阵拼成一个，减少 kernel launch）
QKV 权重：[5120, 2048]  # [4096+512+512, 2048]
```

**计算过程**：
```python
# 输入
hidden_states: [5, 2048]

# QKV 投影（一次矩阵乘法算出 Q、K、V）
qkv = F.linear(hidden_states, weight)  # [5, 2048] @ [2048, 5120] = [5, 5120]

# 拆分
q = qkv[..., :4096]    # [5, 4096] -> reshape -> [5, 32, 128]
k = qkv[..., 4096:4608]  # [5, 512] -> reshape -> [5, 4, 128]
v = qkv[..., 4608:5120]  # [5, 512] -> reshape -> [5, 4, 128]

# 数据示例（仅展示 token0 的 head0 前 4 个值）：
# q[0, 0, :4] = [0.32, -0.15, 0.78, -0.42, ...]   ← token0, Q head0, 128 维
# k[0, 0, :4] = [0.11, 0.54, -0.33, 0.27, ...]    ← token0, KV head0, 128 维
# v[0, 0, :4] = [-0.28, 0.63, 0.19, -0.51, ...]   ← token0, KV head0, 128 维
```

| 属性 | 值 |
|------|-----|
| **Q 权重** | `[4096, 2048]` = 4096 × 2048 ≈ 8.4M 参数 |
| **K 权重** | `[512, 2048]` = 512 × 2048 ≈ 1.0M 参数 |
| **V 权重** | `[512, 2048]` = 512 × 2048 ≈ 1.0M 参数 |
| **QKV 总计** | 8.4M + 1.0M + 1.0M = **10.5M** 参数 |

> 💾 **BF16 内存**：Q=16.8 MB, K=2.0 MB, V=2.0 MB, QKV 合计 = 10.5M × 2B = **21 MB**
>
> 📐 **参数量的计数规则**：本文档统一使用 ML 领域的惯例，`1M = 1,000,000`（10⁶）。
> 如果按计算机存储的 1024×1024 算，`4096 × 2048 = 8,388,608 ≈ 8.0 MiB`（相差约 5%）。

---

#### 2.2.b QK 归一化 (QKNorm)

对 Q 和 K 的 **每个 head 独立做 RMSNorm**，稳定注意力计算的数值范围。

```
Q [5, 32, 128] → QKNorm(per-head, dim=128) → Q_norm [5, 32, 128]
K [5, 4, 128]  → QKNorm(per-head, dim=128) → K_norm [5, 4, 128]
```

> 💡 **为什么要先 reshape 成 per-head 再归一化？**
>
> 不能直接在 4096 维上统一做，因为 **不同 head 的数值分布可能完全不同**：
> ```
> 假设 4096 维 Q 中仅看前两个 head：
>   head0: [10.0, -10.0, 10.0, -10.0, ...]   ← 大数值，RMS ≈ 10
>   head1: [0.01, -0.01, 0.01, -0.01, ...]   ← 小数值，RMS ≈ 0.01
> ```
>
> - ❌ **统一归一化**（4096 维一起算 RMS）：RMS ≈ 7（被 head0 主导），
>   head1 被过度缩小，信息丢失
> - ✅ **per-head 归一化**（每组 128 维独立算 RMS）：head0 `÷10`、head1 `÷0.01`，
>   各自回到 RMS≈1，互不干扰
>
> 每个 head 学到的关注模式不同，激活值的尺度就不同。per-head 归一化保证了**每个 head 在
> 自己的尺度上独立做归一化**，不被其他 head 干扰。

与 2.1 节的 RMSNorm 完全相同的公式，但作用范围是每个 head 的 128 维，不是整个 2048 维。
参数量：`2 × head_dim = 2 × 128 = 256`（q_norm 和 k_norm 各 128）。

> 🔍 **Q 在 QK Norm 中的 shape 变化（临时 reshape，用完恢复）**：
> ```
> Q (flat): [5, 4096]  ← QKV 投影后的原始形状
>    ↓ 临时 reshape（仅在 QK Norm 内部）
> Q (head): [5, 32, 128] → QKNorm(per-head, dim=128) → [5, 32, 128]
>    ↓ reshape 回来
> Q (flat): [5, 4096]  ← 传给 RoPE 和 Attention
> ```
> 源码实际路径：`q.view([5, 32, 128]) → q_norm → q.view([5, 4096]) → rotary_emb(positions, q, k)`
> 因此 **RoPE 和 Attention 的入参都是 flat 的 `[5, 4096]`**，内部自行处理 reshape。

---

#### 2.2.c 旋转位置编码 (RoPE)

##### 为什么需要位置编码？

Attention 本身是**排列不变**的——如果把"你好"和"好你"的 token 顺序打乱，
`Q @ K^T` 计算出的注意力分数矩阵完全相同，因为 Q 和 K 的内容相同，只是位置换了。
但自然语言中语序至关重要（"我爱你"≠"你爱我"），所以必须显式注入位置信息。

##### RoPE 的核心思想

对每个 token 的 Q 和 K 向量做**旋转**，旋转角度由 token 的位置决定：

```
位置 0: Q[0] → 旋转 0°      → Q_rope[0] = Q[0]
位置 1: Q[1] → 旋转 θ₁      → Q_rope[1] = 旋转(Q[1], θ₁)
位置 2: Q[2] → 旋转 θ₂      → Q_rope[2] = 旋转(Q[2], θ₂)
...
```

θ₁, θ₂, θ₃ 各不相同且**随位置单调递增**——位置越远，旋转角度越大。

##### 如何在 128 维向量上做旋转？

RoPE 把 128 维向量看成 **64 对 (2 维平面)**，每对独立旋转：

```
head_dim=128  →  拆成 (dim0, dim1), (dim2, dim3), ..., (dim126, dim127)  共 64 对

第 k 对在位置 pos 的旋转：
  [x₂ₖ,   x₂ₖ₊₁]   →   [x₂ₖ·cos(pos·ωₖ) - x₂ₖ₊₁·sin(pos·ωₖ),
                         x₂ₖ·sin(pos·ωₖ) + x₂ₖ₊₁·cos(pos·ωₖ)]
```

其中 ωₖ 为**预设的频率**（不同 k 不同频率，类似三角函数），由 `rope_theta` 参数控制：

```
ω_k = 1 / (rope_theta ^ (2k / head_dim))    rope_theta = 1,000,000

k=0:   ω₀ = 1/1,000,000⁰          = 1              ← 最快（相邻位置差异大）
k=1:   ω₁ = 1/1,000,000^(2/128)   ≈ 0.81
k=32:  ω₃₂ = 1/1,000,000^(64/128) ≈ 0.001          ← 较慢
k=63:  ω₆₃ = 1/1,000,000^(126/128)≈ 0.000001       ← 最慢（长距离）
```

频率从快到慢递减：低维度负责捕捉**短距离**的相对位置（相邻词），高维度负责**长距离**（跨段落）。

##### 预计算表生成

```python
# Python 伪代码：预计算 cos/sin 表
max_pos, head_dim, theta = 40960, 128, 1_000_000.0
cos_table = torch.zeros(max_pos, head_dim // 2)
sin_table = torch.zeros(max_pos, head_dim // 2)

for pos in range(max_pos):
    for k in range(head_dim // 2):
        omega = 1.0 / (theta ** (2.0 * k / head_dim))
        angle = pos * omega
        cos_table[pos, k] = cos(angle)
        sin_table[pos, k] = sin(angle)

# 推理时查表（省去重复计算）
# Q  [5, 4096] = 5 token × (32 heads × 128 dim),  cos_table [40960, 64],  sin_table [40960, 64]
# Q[0] [4096] 旋转：内部逻辑 = 先 reshape 成 [32, 128] → 64 对分别旋转 → reshape 回 [4096]
# cos_table[0] [64] → 广播为 [1, 64] → 每对重复 2 次 → [1, 128] → 对齐 [32, 128] → reshape [4096]
Q_rope[0] = Q[0] * cos_table[0] + rotate_half(Q[0]) * sin_table[0]  # 位置 0 → [4096]
Q_rope[5] = Q[5] * cos_table[5] + rotate_half(Q[5]) * sin_table[5]  # 位置 5 → [4096]
# 最终 Q_rope: [5, 4096]，形状不变（flat），数值变了
```

> 💡 **reshape 在哪里做的？**
>
> `rotary_emb` 传入的是 flat `[5, 4096]`，但 RoPE 需要按 128 维的 head 做旋转。
> 这个 reshape **在 `RotaryEmbedding.forward()` 内部完成**：
> ```
> # PyTorch 路径（vllm/model_executor/layers/rotary_embedding/base.py:177）
> query = query.view(num_tokens, -1, head_size)  # [5, 4096] → [5, 32, 128]
> # ... 旋转计算 ...
> query = query.reshape(original_shape)           # [5, 32, 128] → [5, 4096]
> ```
> CUDA kernel 路径类似，通过 `num_heads = hidden_size / head_size` 动态解析头部结构。
> 所以**调用方无需手动 reshape**，传入 flat 即可。
```

> 💡 **`rotate_half` 做了什么？**
>
> 把向量**前后半互换，前半取负**。对 128 维的 head：
> ```
> rotate_half([a₀, a₁, ..., a₆₃, b₀, b₁, ..., b₆₃])
>     = [-b₀, -b₁, ..., -b₆₃, a₀, a₁, ..., a₆₃]
> ```
>
> **为什么要这么做？** 2D 旋转公式 `(x, y) → (x·cosθ - y·sinθ, x·sinθ + y·cosθ)` 
> 可以重写为 `[(x, y) · cosθ + (-y, x) · sinθ]`。把 128 维看成 64 个 (x,y) 对，
> `rotate_half` 就是同时对所有 (x,y) 算出 `(-y, x)`，然后一次性做完旋转。
>
> 实际调用时 Q 是 flat 的 `[5, 4096]`，rotate_half 在每 128 维的 head 上独立操作，
> 等价于 `Q.view([5, 32, 128])` → 对最后一维 rotate_half → `view([5, 4096])`。
```

> 💡 **表里到底存了什么？**
>
> `cos_table[pos, k]` 存的是 **`cos(pos × ωₖ)`** 的值（不是角度本身）：
> ```
> cos_table = [40960 rows × 64 columns]
>   row 0    = [cos(0×ω₀), cos(0×ω₁), ..., cos(0×ω₆₃)]     ← 位置 0 的 64 个余弦值
>   row 1    = [cos(1×ω₀), cos(1×ω₁), ..., cos(1×ω₆₃)]     ← 位置 1
>   ...
>   row 40959= [cos(40959×ω₀), ..., cos(40959×ω₆₃)]        ← 位置 40959
> ```
> `sin_table` 同理，存 `sin(pos × ωₖ)`。推理时 token 位置 = 查表行号，
> 直接用 cos/sin 值旋转 Q/K，**不用现场算三角函数**。

##### 关键性质：相对位置

RoPE 最巧妙的地方：**两个 token 的 Q 和 K 做点积时，结果只取决于它们的相对位置差**：

```
Q_rope(pos=m) @ K_rope(pos=n)^T = f(Q_m, K_n, m-n)
```

即位置 m 和 n 的向量点积，等价于原始 Q、K 内容加上 (m-n) 的函数。
这意味着模型学到的是**相对位置关系**（距离 3 个 token），而不是绝对位置（第 5 个 token）。

##### 参数与内存

| 属性 | 值 |
|------|-----|
| **参数量** | 0（无训练参数） |
| **预计算表** | cos/sin 各 `[40960, 64]`，BF16 合计 ≈ 10.4 MB（所有层共享同一张表） |
| **max_position_embeddings** | = 40960，即模型训练时支持的最大序列长度 |

> 💡 **`max_position_embeddings = 40960` 是什么意思？**
>
> 它决定了 RoPE 预计算表的大小——`cos_table` 需要为 **0 到 40959** 每个位置都存一份
> 旋转角度。如果推理时来了一个位置 50000 的 token，表中查不到，就需要**外推**
> （用公式现场计算 ωₖ × 50000 的 cos/sin），或者通过位置插值等方法处理。
>
> Qwen3MoE 30B 支持 40960 的上下文长度，约 3 万汉字，比大多数模型的 8192/32768 长得多。

> 💡 **RoPE 后的 Q 和 K 仍保持原有形状**：`Q_rope [5, 4096]`, `K_rope [5, 512]`（flat），
> 只是每个向量的数值发生了变化——相同 token 在不同位置得到不同的 RoPE 值。

---

#### 2.2.d 注意力计算

RoPE 之后，用 Q 和 K 计算注意力分数，再用分数加权聚合 V：

```
注意力分数 = softmax(Q @ K^T / √128)   ← 每个 head 独立算，32 组
注意力输出 = 分数 @ V                   ← 加权和
```

> 💡 **dot product attention（点积注意力）是什么？**
>
> 核心公式：$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
>
> 其中：
> - $Q \in \mathbb{R}^{n \times d_k}$ — 所有 token 的 Query 向量堆成的矩阵（$n=5$ 个 token, $d_k=128$ 维）
> - $K \in \mathbb{R}^{n \times d_k}$ — 所有 token 的 Key 向量
> - $V \in \mathbb{R}^{n \times d_v}$ — 所有 token 的 Value 向量
> - $Q K^T$ — 矩阵乘法，结果是 $n \times n$ 的**注意力分数矩阵**，`[i, j]` = 第 i 个 Q 与第 j 个 K 的点积
>
> **点积衡量相似度**：$$\text{dot}(a, b) = a \cdot b = \sum_{k=1}^{d_k} a_k \times b_k$$
> - 方向一致 → 点积大 → token i 应该关注 token j
> - 方向相反 → 点积小 → 忽略
>
> **除以 $\sqrt{d_k}$** 是缩放——$d_k=128$ 维的点积求和后数值可能很大（$\propto \sqrt{d_k}$），
> 除以 $\sqrt{128}$ 把数值拉回合理范围，防止 softmax 梯度消失。
>
> 每个 head 独立做一次上述计算，32 个 head 得到 32 组不同的注意力分布。

**GQA 处理**：4 个 KV head 通过 repeat_interleave 扩展到 32 组，与 32 个 Q head 一一对应：
```
K [5, 4, 128] → repeat_interleave(8) → K_expanded [5, 32, 128]
V [5, 4, 128] → repeat_interleave(8) → V_expanded [5, 32, 128]
```

**因果掩码**：自回归生成时，每个 token 只能看到自己和前面的 token，不能看到后面的。
具体在 vLLM 中通过 PagedAttention 的 block table 实现。

> 💡 **batch=2 与 batch=1 的注意力掩码区别**
>
> 同样 5 个 token，掩码完全不同：
> ```
> batch=2, seq_lens=[2,3] → 5×5 掩码：
>          t0 t1 t2 t3 t4
>   seq0 t0  ✓  ✗  ✗  ✗  ✗    ← token0 只 attend 自己
>        t1  ✓  ✓  ✗  ✗  ✗    ← token1 可以 attend token0
>   seq1 t2  ✗  ✗  ✓  ✗  ✗    ← token2 跨序列不可见 token0/1
>        t3  ✗  ✗  ✓  ✓  ✗
>        t4  ✗  ✗  ✓  ✓  ✓
>
> batch=1, seq_len=5 → 标准因果掩码：
>          t0 t1 t2 t3 t4
>   t0     ✓  ✗  ✗  ✗  ✗
>   t1     ✓  ✓  ✗  ✗  ✗
>   t2     ✓  ✓  ✓  ✗  ✗    ← 这里不同！batch=1 时 token2 可以看到 token0/1
>   t3     ✓  ✓  ✓  ✓  ✗
>   t4     ✓  ✓  ✓  ✓  ✓
> ```
> vLLM 通过 PagedAttention 的 block table 来区分序列边界，kernel 内部根据
> block table 的 slot 映射自动施加跨序列掩码，不需要显式的 mask 矩阵。

---

#### 2.2.e 输出投影 (O Projection)

32 个 head 的输出拼回 `[5, 4096]` 后，通过 O 投影压缩回 hidden_size=2048：

```
注意力输出 [5, 32, 128] → reshape → [5, 4096]
O 投影 [4096 → 2048]   →  attn_output [5, 2048]
```

| 属性 | 值 |
|------|-----|
| **O 权重** | `[2048, 4096]` = 8.4M 参数 |
| **BF16 内存** | 8.4M × 2B = **16.8 MB** |
| **输入** | `[5, 4096]`（32 head × 128 dim 拼回） |
| **输出** | `[5, 2048]`（回到残差流维度） |

> **数据示例**：注意力输出（以 token0 为例，仅展示前 4 维）：
> ```
> attn_output[0, :4] = [-0.15, 0.63, -0.28, 0.42, ...]  ← [5, 2048]
> ```
> 经过 O 投影后已从 4096 维回到 2048 维，与 hidden_size 对齐。

---

### 2.3 注意力后归一化 (Post-Attention RMSNorm)

与输入层归一化相同的 RMSNorm，对注意力输出结果做归一化后，送入 MLP/MoE 块。

> 💡 **残差融合在哪里完成？**
>
> 两个 RMSNorm（`input_layernorm` 和 `post_attention_layernorm`）都承担残差融合：
> ```
> # input_layernorm（第1层以后）
> hidden_states, residual = input_layernorm(hidden_states, residual)
>   # 内部：new_residual = hidden_states + residual  ← 残差连接
>   #       normalized = rms_norm(new_residual)
>   #       返回 (normalized, new_residual)
>
> # post_attention_layernorm（所有层）
> hidden_states, residual = post_attention_layernorm(hidden_states, residual)
>   # 同上：残差融合 + 归一化
> ```
> 所以残差连接不是单独的 `Add()` 步骤，而是**嵌入在 RMSNorm 内部**作为一种 fused 操作。
>
> 完整的 Pre-norm 残差结构（`Qwen3MoeDecoderLayer.forward`）：
> ```python
> def forward(self, positions, hidden_states, residual):
>     # === Attention 子层 ===
>     if residual is None:
>         residual = hidden_states
>         hidden_states = input_layernorm(hidden_states)
>     else:
>         hidden_states, residual = input_layernorm(hidden_states, residual)
>     hidden_states = self_attn(positions, hidden_states)
> 
>     # === MLP 子层 ===
>     hidden_states, residual = post_attention_layernorm(hidden_states, residual)
>     hidden_states = mlp(hidden_states)
> 
>     return hidden_states, residual
> ```

| 属性 | 值 |
|------|-----|
| **输入** | `hidden_states` [5, 2048] |
| **输出** | `normalized` [5, 2048] |
| **参数量** | `hidden_size` = 2048 |

---

### 2.4 核心组件：Qwen3MoeSparseMoeBlock (MoE)

> 💡 **`moe_intermediate_size` 是什么？**
>
> 单个专家的 MLP 中间维度。SwiGLU 计算流程：
> ```
> input [2048] → gate_proj [2048→768] → SiLU
>             → up_proj   [2048→768] → multiply → down_proj [768→2048] → output [2048]
>                                ↑
>                    moe_intermediate_size = 768
> ```
> 注意：**单个专家是降维的**（2048→768），因为每个专家只需要处理被路由分配给它的那部分 token，
> 不需要"自我膨胀"。MoE 的"容量"来自于 **128 个不同专家的多样性**，而不是单个专家的宽度。
>
> 当 8 个专家同时激活时，总计算量相当于一个 `intermediate_size = 8 × 768 = 6144` 的稠密 MLP，
> 所以稠密版本（如果有的话）用 `intermediate_size=6144` 时计算量相同。

---


这是 Qwen3MoE 的 **关键组件**，实现稀疏混合专家计算。代码入口很简洁，因为实际逻辑委托给了 `FusedMoE` 工厂创建的 `MoERunner`。

##### 代码位置

```python
# vllm/model_executor/models/qwen3_moe.py
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        ...
        # 创建路由器权重 [128, 2048]
        self.gate = ReplicatedLinear(2048, 128, bias=False, ...)

        # FusedMoE 工厂 → 内部创建 MoERunner（含 router、experts）
        # 把 self.gate 传进去，让 runner 内部调用
        self.experts = FusedMoE(
            gate=self.gate,
            shared_experts=self.shared_expert,
            num_experts=128, top_k=8,
            hidden_size=2048, intermediate_size=768,
            renormalize=True, ...
        )

    def forward(self, hidden_states):
        # 注意：router_logits=hidden_states 是占位！
        # self.gate 已在 FusedMoE 内部注册，由 MoERunner 内部调用
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
        return final_hidden_states
```

##### 调用链详解

`FusedMoE` 工厂创建的对象结构（`vllm/model_executor/layers/fused_moe/`）：

```
FusedMoE( gate=self.gate, ... )   ← 工厂函数
  └→ MoERunner                    ← 返回的实例
        ├── gate = self.gate      ← Qwen3 传来的 ReplicatedLinear
        ├── router = FusedTopKRouter  ← 封装 softmax + topk 逻辑
        └── routed_experts = RoutedExperts  ← 持有 w13, w2 权重
```

运行时调用链：

```python
# Qwen3MoeSparseMoeBlock.forward()
self.experts(hidden_states, router_logits=hidden_states)
  │ router_logits=hidden_states 是占位（因为 gate 在内部）
  │
  └→ MoERunner.forward()
      └→ MoERunner._forward_impl()     ← 实际计算入口
           │
           ├─ [1] 调用 gate
           │     router_logits = self.gate(hidden_states)   # [5, 128]
           │
           ├─ [2] 调用 router.select_experts(router_logits)
           │     └→ FusedTopKRouter
           │           └→ ops.topk_softmax(gating_output)   # CUDA 融合 kernel
           │              返回 topk_weights [5,8], topk_ids [5,8]
           │
           ├─ [3] 调用 routed_experts.forward_modular(...)
           │     └→ quant_method.apply()
           │           └→ fused_moe_kernel (Triton JIT)
           │              输入: hidden_states, topk_weights, topk_ids, w13, w2
           │              输出: [5, 2048]  (已加权求和)
           │
           └─ [4] 返回 final_hidden_states [5, 2048]
```

> 🔑 **关键设计**：`router_logits=hidden_states` 这个占位是最容易困惑的地方。
> 因为 `self.gate` 已传给 `FusedMoE`，MoERunner 内部会再次调用 `self.gate(hidden_states)`，
> 所以 Qwen3 传什么作为 `router_logits` 都不影响结果。这是 FusedMoE 统一接口设计的代价。
> 理解这点后，运行流程就很清晰了——**Qwen3MoE 只是把 gate 的"调用权"委托给了 MoERunner**。

---

### 2.4.a 路由器 (Gate) — 给每个 token 打 128 个专家的"分诊分"

路由器是一个 `ReplicatedLinear(2048, 128)`，把 `hidden_states [5, 2048]` 映射为每个 token 对各专家的偏好分数：

```
router_logits = hidden_states @ W_gate^T   # [5, 2048] @ [2048, 128] = [5, 128]
```

| 属性 | 值 |
|------|-----|
| **权重** | `W_gate`: `[128, 2048]` = 2048 × 128 ≈ **0.26M 参数** (0.5 MB BF16) |
| **输入** | `hidden_states [5, 2048]` | 
| **输出** | `router_logits [5, 128]` — 每个 token 对 128 个专家的 raw score |

**数据示例**：
```
token0 "你": [2.3, 1.1, 0.5, -0.8, 3.2, ..., -1.1]  ← 128 个专家的 raw logits
token1 "好": [1.8, 0.9, 2.1, 0.3, -1.5, ..., 0.7]
token2 "你": [2.3, 1.1, 0.5, -0.8, 3.2, ..., -1.1]  ← token0 和 token2 同为"你"，logits 相同
token3 "是": [-0.5, 2.7, 0.8, 1.2, -0.3, ..., 0.4]
token4 "谁": [1.2, -0.7, 0.3, 2.5, 0.1, ..., -0.9]
```

> ⚠️ **raw logits ≠ 概率**：ReplicatedLinear 输出任意实数，后面要经过 softmax → topk。

---

### 2.4.b Top-K 选择与权重归一化

在 `FusedMoE` 内部，对 `router_logits` 依次做三步：

```
router_logits [5, 128]
  → softmax → 转成 [0,1] 概率     每个 token 的 128 个权重和为 1
  → topk(k=8) → 只保留最高的 8 个  选出 8 个专家的 id 和 weight
  → renormalize(可选) → 8 个权重重新归一化到和为 1
```

**数据示例**：
```
token0: softmax 后 top-8 → 专家1(0.35), 7(0.12), 15(0.08), ..., 89(0.01)
                            ↑ 权重最高的 8 个，和为 1.0
token1: softmax 后 top-8 → 专家2(0.28), 0(0.19), 5(0.14), ..., 78(0.02)
```

> 💡 **`norm_topk_prob=True` 做了什么？**
>
> softmax 后的 8 个权重不直接使用——再做一次 `权值 ÷ sum(权值)` 重新归一化到和为 1。
> 这样可以消除未被选中的 120 个专家对概率分布的影响。

---

### 2.4.c Token 分发与专家 MLP 计算

选出 top-8 专家后，需要把 token 分发给对应的专家做 MLP，再按权重合并。

#### 每个专家在做什么 MLP？

每个专家就是一个 **SwiGLU 的双层 MLP**，和标准 Transformer FFN 一模一样：

```
输入 hidden [2048]
  │
  ├─ gate_proj [2048→768] → SiLU 激活  → gate [768]   ← 控制"信息是否通过"
  ├─ up_proj   [2048→768]              → up   [768]   ← 实际的信息内容
  │
  ├─ SiLU(gate) × up  (逐元素相乘)      → activated [768]  ← gated 机制
  │
  └─ down_proj [768→2048]              → output [2048] ← 压回残差流
```

关键操作是 `SiLU(gate) × up`（SwiGLU）：
- `gate` 经过 `SiLU` 激活后变成 [0,1] 区间的**门控信号**
- `up` 是"该专家想表达的内容"
- 相乘后，门控信号决定哪些信息通过、哪些被抑制
- 最后 `down_proj` 压回 2048 维，和残差流对齐

每个专家独立做上述计算。8 个专家各算出 `[2048]`，按 topk_weight 加权求和，得到最终的 `[2048]`。

#### 为什么需要融合 kernel？

朴素实现的问题是**负载不均**：

```
专家 1：收到 token0(token0→专家1)、token3  → 2 个 token → 很快算完
专家 7：收到 token0、token1、token2、token4  → 4 个 token → 很慢
...
```
按专家逐个串行计算 → GPU 利用率极低（一些专家在等，一些专家闲着）。

#### FusedMoE 融合 kernel

vLLM 的 `FusedMoE` 用一个融合 CUDA kernel 一次完成所有被激活专家的计算：

```
输入: hidden_states [5, 2048], topk_weights [5, 8], topk_ids [5, 8]
  ↓
融合 kernel 内部并行：
  ├─ 按 expert_id 排序 → 把属于同一专家的 token 聚在一起
  ├─ 对每个有 token 到达的专家，独立执行 gate_up_proj → SiLU → down_proj
  ├─ 对每个 token 的输出乘以对应的 topk_weight
  └─ 同一个 token 的 8 个专家结果求和 → [5, 2048]
```

伪代码逻辑：

```python
def fused_moe_forward(hidden_states, topk_weights, topk_ids, w13, w2):
    # w13: [num_experts, 2×moe_intermediate, hidden_size]  gate+up 合并
    # w2:  [num_experts, hidden_size, moe_intermediate]      down 投影
    return fused_moe_cuda_kernel(
        hidden_states, topk_weights, topk_ids, w13, w2,
        top_k=8, renormalize=True,
    )  # → [5, 2048]
```

> 🔍 **`w13` 和 `w2` 是什么？**
>
> 每个专家有 3 个权重矩阵：`gate_proj`、`up_proj`、`down_proj`。
> 为减少 kernel launch，`gate_proj` 和 `up_proj` 合并为 `w13`（拼在一起），
> `down_proj` 单独为 `w2`。所以 `w13` 的最后一维是 `2 × moe_intermediate_size = 1536`。

---

### 2.4.d 结果归约

融合 kernel 输出已经是加权求和后的 `[5, 2048]`，但如果有分布式并行，还需要通信：

| 并行方式 | 操作 | 说明 |
|---------|------|------|
| **Tensor Parallel (TP)** | `all_reduce` | 每张卡持有部分 expert 的输出，求和得到完整结果 |
| **Expert Parallel (EP)** | 已在 kernel 内完成 | 每个 expert 只在一张卡上，通过 EPLB 均衡跨卡负载 |
| **Sequence Parallel (SP)** | `all_gather` | 每张卡持有部分 token，收集后拼回完整序列 |

对于单卡推理（tp_size=1, ep_size=1），不需要任何通信操作。

**数据示例**（MoE 最终输出）：
```
token0 "你": [-0.12, 0.45, 0.78, -0.33, ...]  ← 8 个专家加权求和
token1 "好": [0.33, -0.21, 0.56, 0.89, ...]
token2 "你": [-0.12, 0.45, 0.78, -0.33, ...]  ← 与 token0 接近但不完全相同
                                                 （47 层前馈后上下文已分化）

> 💡 **batch=1 对比**：如果 5 个 token 同属一个序列（"你好你是谁"），
> token0"你"和 token2"你" 的 hidden state **差异会小很多**。
> 因为 token2 可以通过注意力直接看到 token0/1 的上下文，词嵌入一致。
> batch=2 时两序列隔离，token2 完全看不到 token0/1，分化更大。
```

---

### 2.4.e 参数与稀疏性汇总

| 属性 | 值 |
|------|-----|
| **总专家数** | 128 |
| **每 token 激活专家数 (top_k)** | 8 |
| **稀疏性** | 8/128 = **6.25%** |
| **路由器参数** | 2048 × 128 ≈ 0.26M (**0.5 MB**) |
| **专家参数（单层）** | 128 × 3 × 2048 × 768 ≈ **604M** (**1.21 GB**) |
| **激活参数（每 token）** | 8 × 3 × 2048 × 768 ≈ **37.7M** |
| **共享专家** | 无（30B 未配置） |

> 🧠 **为什么用 MoE？——直观理解**
>
> MoE 的思想是：**不用让所有参数处理每个 token**。
> 
> 类比一个医院：**稠密模型** = 每个科室都有全科医生（所有参数都参与）；
> **MoE 模型** = 有 128 个专科医生（专家），每个 token 根据"病情"（路由权重）
> 只找最相关的 8 个专科医生。路由器（gate）就是分诊台护士。
>
> 效果：总医生团队 128 人（总参数大），但每次只看 8 个专家（激活参数少）。
> 这就是 **"大容量、低延迟"** 的秘密。
>
> **注意**：Qwen3MoE 30B 没有共享专家（`shared_expert_intermediate_size=0`）。

---

### 阶段 2 总结：48 层 MoE 的完整迭代

```
配置：decoder_sparse_step=1, mlp_only_layers=[], 共 48 层

Layer 0 (稀疏):  input=[5,2048], residual=None
                 → residual = input  (保存残差)
                 → hidden_states = input_layernorm(input)      # Pre-norm
                 → hidden_states = self_attn(positions, hidden_states)
                 → hidden_states, residual = post_attention_layernorm(hidden_states, residual)  # Pre-norm
                 → hidden_states = mlp_MoE(hidden_states)      # Qwen3MoeSparseMoeBlock
                 → output=[5,2048], residual=[5,2048]

Layer 1 (稀疏):  input=[5,2048], residual_from_prev=[5,2048]
                 → hidden_states = input_layernorm(input + residual)
                 → hidden_states = self_attn(positions, hidden_states)
                 → hidden_states, residual = post_attention_layernorm(hidden_states, residual)
                 → hidden_states = mlp_MoE(hidden_states)
                   - router_logits = gate(hidden_states)       # [5, 128]
                   - topk_weights, topk_ids = topk(router_logits, k=8)
                   - fused_out = FusedMoE(hidden_states, topk_weights, topk_ids)
                 → output=[5,2048], residual=[5,2048]

Layer 2 (稀疏):  同上
...
Layer 47 (稀疏): 同上

最终输出：hidden_states [5, 2048]

# 统计:
# - 稠密层：0 层 (全部 48 层均为 MoE)
# - 总专家数：128
# - 每 token 激活专家数：8
# - 稀疏性：8/128 = 6.25% —— 每层仅使用 6.25% 的 MoE 参数
```

> 📌 **Pre-norm 残差模式**：源码确认采用 Pre-LayerNorm 结构。
> 每个子层（Attention / MLP）之前先做 RMSNorm，
> 残差流通过 `residual` 变量显式传递，与输入相加后再进入下一子层。

---

## 阶段 3~7: 后续处理链（与标准语言模型相同）

这些阶段的原理和实现与标准 LLaMA/Qwen3 等模型完全一致，此处仅做概要。

| 阶段 | 输入 → 输出 | 参数量 | 功能 |
|------|-----------|--------|------|
| **3. 最终 RMSNorm** | `[5, 2048]` → `[5, 2048]` | 2048 | 最后一层归一化 |
| **4. LM Head** | `[5, 2048]` → `[5, 151936]` | 311M | 线性投影到词表维度 |
| **5. Logits 处理** | `[5, 151936]` → `[2, 151936]` | 0 | 取每序列末位 logits |
| **6. 采样** | `[2, 151936]` → `token_ids` | 0 | 每序列采样一个 token |
| **7. Token 解码** | `token_ids` → `text` | 0 | token ID → 文本 |

### 实物数据示例

以 5 个 token `[14("你"), 26("好"), 14("你"), 37("是"), 52("谁")]` 为例：

```
# 阶段 4 — LM Head 输出（取 token4 "谁" 的 logits）
token_id   "的"(31)  "了"(45)  "谁"(52)  "是"(37)  "我"(20)  ...
logit       5.2       4.8      12.3      3.1      -1.2       ...

# 阶段 5 — Logits 处理：取每序列末位
序列0 "你好" → logits[1] = ["的":3.2, "了":5.8, "好":10.1, "是":2.4, ...]  ← "好"的 logit 最高
序列1 "你是谁" → logits[4] = ["的":5.2, "了":4.8, "谁":12.3, "是":3.1, ...]  ← "谁"的 logit 最高

# 阶段 6 — 采样（greedy, temperature=0）
序列0 → argmax = 31 ("好"),   序列1 → argmax = 52 ("谁")

# 阶段 7 — 解码
[31, 52] → ["好", "谁"]    → 续写结果："你好"+"好" = "你好 好",  "你是谁"+"谁" = "你是谁 谁"

# batch=1 对比（如果 5 个 token 属于同一个序列 "你好你是谁"）：
# 末位 logits 只有 1 个位置 → 采样 1 个 token → 解码 1 个 token
# 续写结果：可能是 "你好你是谁"+"的"
```

> 📌 **关于 `tie_word_embeddings`**：
> Qwen3MoE 30B 配置 `tie_word_embeddings=false`，
> Embedding（311M 参数）和 LM Head（311M 参数）**各自独立**，
> 不共享权重。总参数量已包含两者。

---

## 关键算子详解

### 算子汇总表

| 算子名称 | 功能 | 输入形状 | 输出形状 | 参数量 | 内存占用 (BF16) | 计算复杂度 |
|---------|------|---------|---------|--------|----------------|-----------|
| **VocabParallelEmbedding** | 词嵌入查找 | `[B, S]` | `[B, S, H]` | 311M | 622 MB | O(B×S) |
| **RMSNorm** | 均方根归一化 | `[B, S, H]` | `[B, S, H]` | H | 4 KB | O(B×S×H) |
| **QKVParallelLinear** | QKV 投影 | `[B, S, H]` | `[B, S, (N_q+N_kv)×D]` | 10.5M | 21 MB | O(B×S×H²) |
| **RowParallelLinear** | 行并行线性 (O 投影) | `[B, S, N_q×D]` | `[B, S, H]` | 8.4M | 16.8 MB | O(B×S×H²) |
| **SiluAndMul** | SiLU 激活 | `gate, up` | `[B, S, H_inter]` | 0 | 0 | O(B×S×H_inter) |
| **ReplicatedLinear** | 复制线性 (Router) | `[N, H]` | `[N, E]` | H×E | H×E×2B ≈ 0.5 MB | O(N×H×E) |
| **FusedMoE** | 融合 MoE 计算 | `[N, H]` | `[N, H]` | E×(3×H×H_inter) | E×(3×H×H_inter)×2 B | O(N×K×H×H_inter) |
| **ParallelLMHead** | LM 头投影 | `[B, S, H]` | `[B, S, V]` | 311M | 622 MB | O(B×S×H×V) |

**符号说明：**
- B = batch_size
- S = sequence_length
- H = hidden_size (2048)
- N = num_tokens (B×S)
- E = num_experts (128)
- K = top_k (8)
- H_inter = moe_intermediate_size (768)
- N_q = query heads (32)
- N_kv = kv heads (4)
- D = head_dim (128)
- V = vocab_size (151936)

**注意**：QKV 投影采用解耦设计：
- Q 输出维度：`N_q × D = 32 × 128 = 4096`
- KV 输出维度：`N_kv × D = 4 × 128 = 512`
- QKV 总输出：`4096 + 512 + 512 = 5120`

### MoE 特有算子详解

#### ReplicatedLinear (路由器)

| 属性 | 值 |
|------|-----|
| **功能** | 为每个 token 计算专家路由权重 |
| **输入** | `[num_tokens, hidden_size]` |
| **输出** | `[num_tokens, num_experts]` |
| **参数量** | `hidden_size × num_experts` = `2048 × 128` ≈ 262K |
| **内存占用** | `262K × 2 bytes` ≈ 524 KB |

#### FusedMoE

| 属性 | 值 |
|------|-----|
| **功能** | 融合 Top-K 路由、专家分发、专家计算、结果收集 |
| **输入** | `hidden_states: [N, H]`, `router_logits: [N, E]` |
| **输出** | `fused_out: [N, H]` |
| **参数量** | `E × (2 × H × H_inter + H_inter × H)` = `128 × 3 × 2048 × 768` ≈ 604M |
| **激活参数** | `K × (3 × H × H_inter)` = `8 × 3 × 2048 × 768` ≈ 37.7M (每 token) |
| **稀疏性** | `K/E` = `8/128` = 6.25% |

---

## 代码调用栈

### 完整调用链路

```python
# 用户调用
output = llm.generate("你好", sampling_params)

# 内部调用栈
└── TokenExecutor.execute_model()
    └── GPUModelRunner.execute_model()
        ├── _model_forward(input_ids, positions)
        │   └── Qwen3MoeForCausalLM.forward()
        │       └── Qwen3MoeModel.forward()
        │           ├── embed_input_ids(input_ids)
        │           │   └── VocabParallelEmbedding
        │           └── [Qwen3MoeDecoderLayer × 48]
        │               ├── input_layernorm (RMSNorm)    # Pre-norm: 先归一化再进子层
        │               ├── self_attn (Qwen3MoeAttention)
        │               │   ├── qkv_proj (QKVParallelLinear)
        │               │   ├── q_norm, k_norm (RMSNorm) # Per-head QK 归一化
        │               │   ├── rotary_emb (RotaryEmbedding)
        │               │   ├── attn (PagedAttention)    # GQA: 32Q / 4KV
        │               │   └── o_proj (RowParallelLinear)
        │               ├── post_attention_layernorm (RMSNorm) # Pre-norm + 残差融合
        │               └── mlp (混合类型)
        │                   ├── 稠密层：Qwen3MoeMLP       # 仅 decoder_sparse_step>1 时存在
        │                   │   ├── gate_up_proj (MergedColumnParallelLinear)
        │                   │   ├── SiluAndMul
        │                   │   └── down_proj (RowParallelLinear)
        │                   └── 稀疏层：Qwen3MoeSparseMoeBlock
        │                       ├── gate (ReplicatedLinear) ← 路由器
        │                       ├── shared_expert (可选)  # 30B 无共享专家
        │                       └── experts (FusedMoE) ← 路由专家
        │                           ├── Top-K 专家选择
        │                           ├── 专家并行分发 (EPLB)
        │                           ├── 融合专家计算
        │                           └── Tensor Parallel 归约
        │           └── norm (RMSNorm)                  # 最终输出归一化
        └── sample_tokens(logits)
            └── Sampler()
                ├── temperature scaling      # temperature=0 → greedy argmax
                ├── TopKTopPSampler           # Top-K / Top-P 概率过滤
                ├── apply_penalties           # repetition / frequency 惩罚
                └── multinomial / argmax      # 随机采样或贪心解码
        └── TokenizerDetokenizer.decode()     # token ID → 文本
```

---

## Qwen3MoE vs Qwen3 对比

### 架构差异

| 特性 | Qwen3 (稠密) | Qwen3MoE (稀疏) |
|------|-------------|----------------|
| **MLP 类型** | 所有层使用 `Qwen3MLP` | 混合使用 `Qwen3MoeMLP` 和 `Qwen3MoeSparseMoeBlock` |
| **参数效率** | 所有参数始终激活 | 仅激活 `top_k/num_experts` 比例参数 |
| **模型容量** | `N_layers × H_inter` | `N_layers × (H_inter + num_experts × H_inter_moe)` |
| **推理速度** | 稳定 | 取决于 MoE 层比例和 top_k |
| **显存占用** | 较低 | 较高 (需存储所有专家权重) |
| **计算效率** | 高 | 依赖 EPLB 负载均衡 |

### 参数量与内存

假设配置 (Qwen3MoE 30B)：`hidden_size=2048`, `intermediate_size=6144`, `num_experts=128`, `moe_intermediate_size=768`, `top_k=8`

| 组件 | Qwen3 (稠密) | Qwen3MoE 30B | BF16 内存(MoE) |
|------|-------------|------------|----------------|
| **Embedding** | 311M | 311M | 622 MB |
| **Attention (×48)** | 18.9M × 48 = 907M | 18.9M × 48 = 907M | 1.81 GB |
| **MLP / MoE (×48)** | 48 × 37.7M ≈ 1.81B | 48 × 604M ≈ 29B | 58 GB |
| **Router (×48)** | 0 | 48 × 262K ≈ 12.5M | 25 MB |
| **LM Head** | 311M | 311M | 622 MB |
| **RMSNorm (×97)** | 97 × 2048 ≈ 0.2M | 97 × 2048 ≈ 0.2M | 0.4 MB |
| **总计** | ≈ 3.34B | ≈ 30.5B | **≈ 61 GB** |
| **激活参数/层** | 37.7M (MLP) | 37.7M (MoE, top_k=8) | — |

> 💾 **权重总内存**：Qwen3MoE 30B 的全部权重在 BF16 下约占用 **61 GB** 显存。
> 如果使用 4-bit 量化（GPTQ / AWQ），内存可压缩至 61 ÷ 4 ≈ **15 GB**，单张 24G 显卡即可加载。

> 💡 **GQA（Grouped Query Attention）快速理解**：
> Qwen3MoE 30B 有 32 个 Q head 但只有 4 个 KV head（每组 8 个 Q head 共享一组 KV）。
> 这相当于 8 个学生（Q head）共用一套笔记（K/V），大幅减少 KV cache 显存，
> 同时通过更多 Q head 保持注意力表达能力。
>
> **结论**：Qwen3MoE 总参数更多（30.5B vs 3.34B，约 9×），但**每层激活参数量相同**（37.7M）。
> MoE 不节省计算量——它在**相同计算成本下提供 9× 的模型容量**。
> 稠密 MLP 的 `intermediate_size=6144` 恰好等于 `top_k × moe_intermediate_size = 8 × 768 = 6144`，
> 因此二者的矩阵乘法 FLOPs 相同，MoE 用"更多专家知识"换"相同推理速度"。

---

## 性能优化要点

### 1. Expert Parallel (EPLB)

```python
# 专家并行配置
eplb_config = parallel_config.eplb_config
num_redundant_experts = eplb_config.num_redundant_experts  # 冗余专家数
enable_eplb = parallel_config.enable_eplb  # 启用 EPLB

# 专家分配
n_physical_experts = n_logical_experts + n_redundant_experts
n_local_physical_experts = n_physical_experts // ep_size
```

### 2. Sequence Parallel

```python
# 序列并行配置
is_sequence_parallel = parallel_config.use_sequence_parallel_moe

if is_sequence_parallel:
    hidden_states = sequence_parallel_chunk(hidden_states)
    # ... MoE 计算 ...
    final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
```

### 3. Token 负载均衡

MoE 推理性能关键：确保 token 均匀分布到各专家，避免负载不均。

---

## 总结

Qwen3MoE 的核心创新在于：

1. **稀疏 MoE**：每 token 仅激活 top_k（=8）个专家，实现条件计算——"大容量、低延迟"
2. **解耦注意力维度**：hidden_size(2048) 与 Q 投影维度 (4096) 解耦，增强注意力表达能力
3. **GQA + QKNorm**：32 Q head / 4 KV head 减少 KV cache，per-head RMSNorm 稳定训练
4. **专家并行**：EPLB 实现多卡高效分布式推理，FusedMoE 融合 kernel 提升 GPU 利用率

> 📌 **本文档基于 Qwen3MoE 30B 配置**：
> - `decoder_sparse_step=1`：全部 48 层均为 MoE，无稠密层
> - `shared_expert_intermediate_size=0`：无共享专家
> - `tie_word_embeddings=false`：Embedding 与 LM Head 独立

与 Qwen3 相比，Qwen3MoE 总参数更多（30.5B vs 3.34B，约 9×），但每层激活参数量相同（均为 37.7M），因此推理计算量不变。MoE 的核心优势是：**在相同计算成本下，用更大的总参数量提供更强的模型表达能力**。
