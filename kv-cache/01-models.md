# 四个模型的 KV / KV Cache 结构

> 所有层号从 0 开始。
> 数字是“按 config 推导的量级”，不是某台机器的实测；实际以 `KVCacheConfig` 和日志为准。

## 0. 一张总表

| | DeepSeek V4.1 Flash | Kimi K3 | Qwen3.8 Flash Next | GLM-5.3 Flash |
|---|---|---|---|---|
| 总参数 / 激活 | **552B** / 8B 输入、16B 输出 | **2.8T** / 104B | **125B 主干 + 51B n-gram + 4B MTP ≈ 180B** / 6B | **320B** / 18B |
| 总层数 | 40（+3 MTP） | 93 | 48 | 45（+1 MTP） |
| full/sparse attention 层 | 40 层 sparse MLA | 24 层 MLA | 12 层 GQA + QSA | 11 层 sparse MLA |
| 线性注意力层 | 无 | 69 层 KDA | 36 层 GDN | 34 层 KDA |
| attention 的 KV 形态 | latent 512 + rope 64（缓存 512/层） | latent 512 + rope 64（缓存 576/层） | K/V 各 2 头 × 256 | latent 512（无 RoPE） |
| 是否跨层共享 KV | **是**（4 个 source 层） | 否 | 否 | 否 |
| 是否有滑窗 cache | **是**（128，所有层） | 否 | 否 | 否 |
| 额外 cache | indexer KV、compressor ring | 无 | QSA 压缩 key、QSA ring | indexer kpool、tail ring |
| 主要后端 | FlashMLA / FlashInfer sparse / MegaAttention | FlashMLA / FlashInfer / Cutlass MLA | FlashAttention + QSA | FlashMLA / FlashInfer sparse |

> 参数量来自 ModelScope 模型卡/官方发布：DeepSeek 552B、Kimi 2.8T、Qwen 125B+51B+4B、GLM 320B。
> “激活参数”是 MoE 每个 token 实际参与计算的参数，不等于权重总量；KV cache 只与层结构/头数/维度有关，与总参数量没有直接公式关系。

### 0.1 “多少 B”到底指什么

- **B = billion（十亿）**，**T = trillion（万亿）**；2.8T = 2800B。
- **总参数**：权重文件里一共有多少参数（MoE 的所有专家都算）。
- **激活参数**：每个 token 实际参与计算的参数（MoE 只激活 top-k 专家）。
- DeepSeek V4.1 是**非对称**结构：输入激活 8B、输出激活 16B，总参数 552B。
- Qwen3.8 的 125B 是主干；51B n-gram embedding 和 4B MTP 是额外参数，合计约 180B，但每个 token 只激活 6B 主干。
- **KV cache 大小不由总参数决定**：GLM 320B 的 KV 比 Kimi 2.8T 小，因为 GLM 只有 11 层 sparse MLA、Kimi 有 24 层 MLA；Qwen 180B 的 GQA KV 又比 GLM 大，因为每层 2 个 KV 头 × 256 维。

**最重要的一句话：**

- **DeepSeek V4.1** 的 KV cache 是“每层 SWA + 少数 source 层 compressed KV + indexer/compressor”，消费者层**不拥有** compressed cache；
- **Kimi / Qwen / GLM** 是“少量 attention 层 paged KV + 大量线性注意力层固定 state”。

---

## 1. DeepSeek-V4.1-Flash-1017

### 1.1 结构

```
40 层 backbone，每层：
  hidden 5120
  → q_a_proj (q_lora 1280) → q_b_proj (64 heads × 512)
  → kv_a_proj (512 + 64) → compressor (ratio 0/1/2)
  → sparse MLA (64 heads, head_dim 512, rope 64)
  → o_lora 1024 / o_groups 8
  → MoE (384 routed + 1 shared, top-6)
```

关键 config：

```json
"num_hidden_layers": 40,
"num_attention_heads": 64,
"num_key_value_heads": 1,
"head_dim": 512,
"qk_rope_head_dim": 64,
"q_lora_rank": 1280,
"o_lora_rank": 1024,
"sliding_window": 128,
"compress_ratios": [0,0,2,2,...,2,1,1,...,1,0,0,0],
"kv_source_layer_ids": [2,8,14,20],
"index_source_layer_ids": [2,8,14,20,24,28,32,36]
```

`compress_ratios` 的长度是 43：40 个 backbone + 3 个 MTP。backbone 的分布：

| 层号 | ratio | 含义 | 是否有 compressed KV |
|---|---|---|---|
| 0–1 | 0 | 纯 SWA | 否 |
| 2–19 | 2 | 2 个 token 压成 1 个 state | 仅 source 层 2/8/14 |
| 20–39 | 1 | 1 个 token 1 个 state | 仅 source 层 20 |
| 40–42 | 0 | MTP | 否 |

### 1.2 每层到底有哪些 cache

这是 V4.1 最容易被误读的地方。`get_kv_cache_spec()` 在 `vllm/models/deepseek_v41/attention.py:1195`：

```python
if not self.is_kv_source:
    return None
...
return MLAAttentionSpec(
    block_size=...,
    num_kv_heads=1,
    head_size=self.head_dim,          # 512
    tokens_per_state=self.compress_ratio,  # 1 or 2
    cache_dtype_str=self.kv_cache_dtype,
    model_version="deepseek_v4",
    state_content_bytes=...           # fp8_ds_mla: 584
)
```

但每层**另有一个独立注册的 SWA cache 模块**，在 `attention.py:566`：

```python
self.swa_cache_layer = DeepseekV4SWACache(
    head_dim=self.head_dim,           # 512
    window_size=self.window_size,     # 128
    dtype=...,
    block_size=32,
    packed_bytes_per_token=self.swa_bytes_per_token,  # 584
    ...
)
```

所以对一层来说：

| 层类型 | 拥有的 cache | spec 类 | 关键字段 |
|---|---|---|---|
| 所有层 | SWA cache（窗口 128） | `SlidingWindowMLASpec` | `window=128`, `block=32`, `head=512`, `tokens_per_state=1` |
| source 层 2/8/14/20 | compressed KV | `MLAAttentionSpec` | `head=512`, `tokens_per_state=2`（2/8/14）或 `1`（20） |
| source 层 2/8/14/20 | compressor state | `CircularBufferSpec` | `head=512?`, 1 block/req |
| index source 层 2/8/14/20/24/28/32/36 | indexer KV | `MLAAttentionSpec` | `head=128`, `tokens_per_state=2/1` |
| 其他层 | **无 compressed KV** | 返回 `None` | 通过 `compressed_cache_prefix` 读 source |

> `get_kv_cache_spec()` 返回 `None` 不代表这层没有 KV cache，而是“这层不拥有 compressed cache”；它的 SWA cache 是另一个模块。

### 1.3 跨层共享怎么工作

- `is_kv_source = layer_id in kv_source_layer_ids`。
- 对每个 `compress_ratio>0` 的层，`kv_source_layer_id = max(s for s in sources if s <= layer_id)`。
- 非 source 层把自己的 `compressed_cache_prefix` 指向 source 层的前缀（`attention.py:605-619`）。
- 运行时 attention 通过 `forward_context` 拿到 source 层的 cache，而不是自己写一份。

效果：

```
layer 0,1:   SWA only
layer 2:     SWA + compressed(ratio2) ← source
layer 3-7:   SWA + read layer2's compressed
layer 8:     SWA + compressed(ratio2) ← source
...
layer 20:    SWA + compressed(ratio1) ← source
layer 21-39: SWA + read layer20's compressed
```

**这就是 DeepSeek V4.1 能把 40 层长上下文的 KV 压到“4 层 compressed”的根本原因。**

### 1.4 字节手算

fp8_ds_mla packed record：`swa_bytes_per_token = 584`（512 维 fp8 + UE8M0 scale/padding）。
compressed record 也是 584 B/state；ratio=2 时每 token 292 B。

| 项 | 公式 | 128k 上下文 |
|---|---|---|
| SWA（40 层） | `40 × 128 × 584` | ≈ 2.99 MB / 请求（固定） |
| compressed ratio2（3 层） | `3 × (S/2) × 584` | ≈ 114.8 MB |
| compressed ratio1（1 层） | `1 × S × 584` | ≈ 76.5 MB |
| indexer（8 层，128 维，ratio 2/1） | 约 `8 × (S/2) × 128` | ≈ 67 MB（粗算） |
| compressor state（4 层） | 4 × ring | 几 MB 量级 |
| **合计** | | **约 260 MB / 128k 请求** |

对比传统 61 层 MLA（DeepSeek-V3）128k 约 8.6 GiB：V4.1 靠“4 层共享 + 压缩 + SWA”把量级降了一个数量级。  
**注意**：这是每请求的 KV；生产环境靠前缀缓存/分页共享进一步摊薄。

### 1.5 易错点

1. `head_dim=512` 不是 576；`qk_rope_head_dim=64` 是其中一部分（`nope=448`）。cache 行是 512，不是 576。
2. `compress_ratios` 包含 MTP 层，长度 43 不是 40；`layer_id < len` 才取，MTP 默认 0。
3. 非 source 层的 `get_kv_cache_spec()` 返回 `None`，但 SWA cache 仍然存在。
4. `kv_source_layer_ids` 与 `index_source_layer_ids` 不是一个集合；indexer 有 8 个 source。
5. `is_kv_source` 只在 backbone（`layer_id < num_hidden_layers`）成立；MTP 不算。

---

## 2. Kimi K3

### 2.1 结构

```
93 层：
  24 层 full MLA（config 用 1-based：4,8,...,92,93 → 0-based 3,7,...,91,92）
  69 层 KDA（Kimi Delta Attention，线性注意力 + short conv）
每层 MoE：896 experts，top-16，2 shared
hidden 7168，96 heads
```

关键 config：

```json
"num_hidden_layers": 93,
"hidden_size": 7168,
"num_attention_heads": 96,
"kv_lora_rank": 512,
"qk_nope_head_dim": 128,
"qk_rope_head_dim": 64,
"v_head_dim": 128,
"mla_use_nope": true,
"linear_attn_config": {
  "full_attn_layers": [4,8,...,92,93],
  "kda_layers": [1,2,3,5,...],
  "num_heads": 96,
  "head_dim": 128,
  "short_conv_kernel_size": 4
}
```

### 2.2 每层 cache

| 层类型 | 数量 | spec | 关键字段 | 增长 |
|---|---|---|---|---|
| MLA | 24 | `MLAAttentionSpec` | `num_kv_heads=1`, `head_size=576`, `max_tp_shards=1` | O(S) |
| KDA | 69 | `MambaSpec` | conv state + recurrent state | **O(1)** |

MLA 的 `head_size` 在 `vllm/models/kimi_k3/nvidia/mla.py:156`：

```python
self.head_size = kv_lora_rank + qk_rope_head_dim  # 512 + 64 = 576
```

即使 `mla_use_nope=true`（不应用 RoPE），代码仍然按 576 分配 cache（`qk_rope_head_dim` 的槽位保留）。这是读 Kimi 时最容易算错的地方。

KDA 的 spec 在 `vllm/models/kimi_k3/nvidia/kda.py:830`，最终落到 `MambaSpec`（`vllm/model_executor/layers/mamba/abstract.py:72`）：

```python
MambaSpec(
    shapes=tuple(self.get_state_shape()),
    dtypes=...,
    block_size=mamba_block_size,
    mamba_type=...,
    num_speculative_blocks=...,
)
```

KDA state 形状（`mamba_utils.py:305`）：

```
conv_state:      [conv_dim / TP, conv_kernel - 1 + num_spec]
recurrent_state: [num_heads / TP, head_dim, head_dim]
conv_dim = num_heads*head_dim + 2*num_k_heads*head_k_dim
```

### 2.3 字节手算（TP=8，FP32 recurrent）

| 项 | 公式 | 量级 |
|---|---|---|
| MLA KV / token | `24 × 576 × 2`（bf16） | 27,648 B ≈ 27 KiB/token |
| MLA KV / 128k | `131072 × 27648` | ≈ 3.4 GiB |
| KDA / layer / rank | conv `36864/8×3×2` + recurrent `96/8×128×128×4` | ≈ 0.8 MB |
| KDA / 69 层 / rank | `69 × 0.8 MB` | ≈ 55 MB（固定） |

**结论**：Kimi 的长上下文成本几乎全在 24 层 MLA；69 层 KDA 的 state 与序列长度无关。  
若把 MLA 换成 `fp8_ds_mla`（656 B/token/layer），24 层约 15.4 KiB/token，128k 约 1.9 GiB。

### 2.4 易错点

1. **Kimi 的 `kda_layers` / `full_attn_layers` 是 1-based**：代码用 `(layer_idx + 1) in ...` 判断；0-based 层号要减 1。
2. `num_key_value_heads=96` 是 config 的干扰项；MLA 实际 `num_kv_heads=1`（latent）。
3. `mla_use_nope=true` 不等于 `head_size=512`；代码仍是 576。
4. KDA 不是 attention，没有 block table；`MambaSpec` 的 `block_size` 是 state block，不是 token block。
5. KDA state 是 **TP 分片**的；估算总显存要乘 TP（或按每 rank 算）。

---

## 3. Qwen3.8-Flash-Next

### 3.1 结构

```
48 层，模式：3 × linear_attention + 1 × full_attention
  36 层 GDN（Gated DeltaNet，linear attention）
  12 层 full attention（3,7,11,...,47）
full attention 不是 MLA，是 GQA：
  24 Q heads, 2 KV heads, head_dim 256
  + QSA（Qwen Sparse Attention）indexer：
      indexer 4 heads, head_dim 128, compress_ratio 4, budget 2048
```

关键 config：

```json
"num_hidden_layers": 48,
"num_attention_heads": 24,
"num_key_value_heads": 2,
"head_dim": 256,
"full_attention_interval": 4,
"layer_types": ["linear_attention", ..., "full_attention", ...],
"indexer_kv_heads": 1,
"indexer_n_heads": 4,
"indexer_head_dim": 128,
"indexer_compress_ratio": 4,
"indexer_budget": 2048,
"linear_num_key_heads": 16,
"linear_num_value_heads": 48,
"linear_key_head_dim": 128,
"linear_value_head_dim": 128
```

### 3.2 每层 cache

| 层类型 | 数量 | spec | 关键字段 | 增长 |
|---|---|---|---|---|
| full attention | 12 | `FullAttentionSpec` | `num_kv_heads=2`, `head_size=256`, `head_size_v=256` | O(S) |
| QSA compressed key | 12 | `MLAAttentionSpec` | `num_kv_heads=1`, `head=128`, `tokens_per_state=4` | O(S/4) |
| QSA ring | 12 | `CircularBufferSpec` | `capacity=4+num_spec`, 1 block/req | O(1) |
| GDN | 36 | `MambaSpec` | conv + recurrent state | O(1) |
| PLE short conv | 1（layer 2） | `MambaSpec` | `tp_replicated=True` | O(1) |

代码：

- full attention spec：`vllm/models/qwen4_exp/nvidia/qsa.py:552`。
- QSA compressed：`vllm/models/qwen4_exp/common/qsa_cache.py:868`。
- QSA ring：`:844`。
- GDN/PLE spec：`vllm/models/qwen4_exp/nvidia/model.py:882`。

### 3.3 字节手算（BF16）

| 项 | 公式 | 量级 |
|---|---|---|
| 主 KV / token | `12 × 2 heads × (256+256) × 2` | 24,576 B ≈ 24 KiB/token |
| 主 KV / 262k | `262144 × 24576` | ≈ 6.0 GiB |
| QSA compressed / token | `12 × 128 × 2 / 4` | 768 B/token |
| QSA compressed / 262k | `262144 × 768` | ≈ 192 MiB |
| GDN / rank（TP=8） | recurrent `48/8×128×128×4` ≈ 393 KB/layer + conv | 36 层约 15 MB |

Qwen 的 full attention 每层 KV 比 MLA 大（GQA 2 头 × 256 维），但只有 12 层；总成本仍远低于 48 层全 attention。

### 3.4 易错点

1. Qwen3.8 的 full attention 是 **GQA**，不是 MLA；不要套 576 的公式。
2. QSA 的 compressed key cache 是 `MLAAttentionSpec`，但它是 **indexer 的 side cache**，不是主 KV；不要把它当 V 用。
3. QSA ring 是 `CircularBufferSpec`，`uses_slot_mapping=False`，不参与前缀缓存。
4. PLE 层的 `MambaSpec` 是 `tp_replicated=True`，每卡都存全量，估算显存时不要乘 TP 分摊。

---

## 4. GLM-5.3-Flash

### 4.1 结构

```
45 层，模式：3 × linear_attention + 1 × deepseek_sparse_attention
  34 层 KDA（linear attention）
  11 层 sparse MLA（3,7,11,...,43）
sparse MLA：
  q_lora 1536, kv_lora 512, qk_nope 256, qk_rope 0, v 256
  mla_use_nope = true（无 RoPE）
  + indexer：32 heads, head_dim 128, kpool=4, topk=2048
  + tail ring：保存未满 pool 的 raw K + gate
```

关键 config：

```json
"num_hidden_layers": 45,
"num_attention_heads": 64,
"kv_lora_rank": 512,
"qk_nope_head_dim": 256,
"qk_rope_head_dim": 0,
"v_head_dim": 256,
"mla_use_nope": true,
"index_head_dim": 128,
"index_n_heads": 32,
"index_topk": 2048,
"index_kpool": 4,
"layer_types": ["linear_attention", ..., "deepseek_sparse_attention", ...]
```

### 4.2 每层 cache

| 层类型 | 数量 | spec | 关键字段 | 增长 |
|---|---|---|---|---|
| sparse MLA | 11 | `MLAAttentionSpec` | `num_kv_heads=1`, `head=512` | O(S) |
| indexer kpool | 11 | `MLAAttentionSpec` | `head=128`, `tokens_per_state=4` | O(S/4) |
| indexer tail | 11 | `CircularBufferSpec` | `num_kv_heads=2`, `head=128`, bf16, ring | O(1) |
| KDA | 34 | `MambaSpec` | conv + recurrent | O(1) |

代码：

- indexer spec：`vllm/models/glm5next/common/attention.py:112`。
- tail spec：`:156`。
- KDA spec：`vllm/models/glm5next/common/kda.py:168`。

GLM 的 sparse MLA 是“无 RoPE 的 MLA”：`qk_rope_head_dim=0`，所以 cache 行只有 512 维（latent），没有 64 维 rope 尾巴。  
但它的 `head_dim=0` 是 config 的占位，实际 head size 由 `kv_lora_rank + qk_rope_head_dim = 512` 决定（`common/attention.py` 的 `self.qk_head_dim = qk_nope + qk_rope`）。

### 4.3 字节手算（BF16）

| 项 | 公式 | 量级 |
|---|---|---|
| MLA / token | `11 × 512 × 2` | 11,264 B ≈ 11 KiB/token |
| MLA / 128k | `131072 × 11264` | ≈ 1.38 GiB |
| indexer kpool / token | `11 × 128 × 2 / 4` | 704 B/token |
| indexer / 128k | `131072 × 704` | ≈ 88 MiB |
| tail / 请求 | `11 × 2 heads × 128 × 2 × ring` | 与 ring 有关，KB 量级 |
| KDA / rank（TP=8） | recurrent `64/8×128×128×4` ≈ 524 KB/layer | 34 层约 18 MB |

### 4.4 易错点

1. `head_dim=0` 不是“没有 head”；看 `kv_lora_rank` 和 `qk_*_head_dim`。
2. `qk_rope_head_dim=0`，所以没有 RoPE cache；不要把 576 套过来。
3. indexer 的 `tokens_per_state=index_kpool=4`：一个 state 覆盖 4 个 token，block 内 state 数 = `block_size/4`。
4. tail ring 的 `block_size` 必须整除 attention block size；不是任意值。
5. KDA 与 Kimi/Qwen 的 linear 层一样，是 `MambaSpec`，不是 paged KV。

---

## 5. 四种 spec 的字段速查

| 字段 | `MLAAttentionSpec` | `SlidingWindowMLASpec` | `MambaSpec` | `CircularBufferSpec` |
|---|---|---|---|---|
| 随 token 增长 | 是（按 `tokens_per_state` 压缩） | 是（窗口内） | 否 | 否 |
| `block_size` | token block | token block（DSV4 SWA=32） | state block | ring 容量 |
| `num_kv_heads` | 1 | 1 | `num_heads`（通常 1） | 1 或 2（GLM tail） |
| `head_size` | latent+rope（512/576） | 512 | 无（看 shapes） | 128/512 |
| `tokens_per_state` | ratio（1/2/4） | 1 | -1 | 1 |
| `page_size_bytes` | `heads × (block/ratio) × content` | 同左 | conv+recurrent state | ring×heads×head |
| 前缀缓存 | 是 | 是（bounded replay 除外） | 是（checkpoint） | **否** |
| slot mapping | 是 | 是 | 是 | **否** |

---

## 6. 手算练习

1. DSV4.1：`block_size=32`，compressed ratio=2，fp8 584 B/state，求一个 compressed page 的字节数。
2. DSV4.1：128k 上下文，3 个 ratio2 source + 1 个 ratio1 source，求 compressed KV 总量（GB）。
3. Kimi：24 层 MLA，576 维，bf16，128k，求 KV 总量（GiB）。
4. GLM：11 层 MLA，512 维，fp8（1 B），128k，求 KV 总量（GiB）。
5. Qwen：12 层 GQA，2 KV heads × 256，bf16，262k，求主 KV 总量（GiB）。
6. 为什么 KDA/GDN 的 `MambaSpec` 不随序列长度增长？它的 `max_memory_usage_bytes` 与 `max_model_len` 有关吗？
7. DSV4.1 的非 source 层 `get_kv_cache_spec()` 返回 `None`，它的 SWA cache 是谁分配的？
8. GLM 的 indexer 为什么用 `tokens_per_state=4`，而不是单独写一个压缩 cache 类？

### 参考答案

1. `page = 1 head × (32/2) states × 584 = 9,344 B`。
2. `3 × 65536 × 584 + 1 × 131072 × 584 = 114,819,072 + 76,546,048 ≈ 191 MB`（约 0.18 GiB）。
3. `24 × 576 × 2 × 131072 ≈ 3.62 GB ≈ 3.37 GiB`。
4. `11 × 512 × 1 × 131072 ≈ 738 MB ≈ 0.69 GiB`。
5. `12 × 2 × 512 × 2 × 262144 ≈ 6.44 GB ≈ 6.0 GiB`。
6. 因为它是 recurrent state（固定形状的 conv + SSM state），不是按 token 存的 K/V；`max_memory_usage_bytes` 只看 state 形状和 speculative blocks，不看 `max_model_len`。
7. 每层都有一个独立注册的 `DeepseekV4SWACache` 模块，worker 会把它当作一个 `AttentionLayerBase` 收集；它的 `get_kv_cache_spec()` 返回 `SlidingWindowMLASpec`。
8. 因为 indexer 的语义就是“每 `kpool` 个 token 一个 entry”，直接复用 `MLAAttentionSpec.tokens_per_state` 就能让 vLLM 的 slot mapping/seq_lens/page table 自动按 pool 粒度生成，不需要新 spec 类。
