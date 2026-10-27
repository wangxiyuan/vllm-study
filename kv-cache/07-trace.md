# 一次请求的 KV Cache 全流程：以“你好，你是谁”为例

> 目标：把“临时 tensor、cache tensor、每层是否复用、为什么 group 跨层”一次讲透。
> 为了讲形状，假设 tokenizer 把“你好，你是谁”切成 **T=6** 个 token；实际可能是 4–8 个，形状只跟 T 有关。

## 0. 先纠正三个误解

1. **KV cache 不是“一份所有层共用”的 K/V。**  
   每一层都有自己的 K/V（或 MLA latent），内容不同；共享的是**block id / block table / slot_mapping** 这些元数据。
2. **KV cache 里存的不是 token id，也不是 hidden state。**  
   存的是每层、每个 token 的 K 和 V 向量（MLA 存 latent+rope）。token id 只在 `input_ids` 里，是一次性的。
3. **block id 不是物理地址。**  
   `block_table[req][block_idx] = 42` 表示“第 2 个逻辑页映射到物理页 42”；层 `l` 的页 42 还有自己的 `layer_stride` 偏移。

## 1. 一句话答案

> **Prefill 时，每层为这 6 个 token 各算一份 K/V，按同一个 `slot_mapping` 写进各自的 page；decode 时每层只追加 1 个 slot。所有层共享“页号/槽位”的编号，但不共享 K/V 内容。**

## 2. 时间线：从请求到第一个输出 token

### 2.1 请求到达（step 0）

输入：“你好，你是谁” → tokenizer → 6 个 token。

| 张量 | shape | 内容 | 生命周期 |
|---|---|---|---|
| `input_ids` | `[6]` | 6 个 token id | 本步临时 |
| `positions` | `[6]` | `[0,1,2,3,4,5]` | 本步临时 |
| `block_table` | `[1, max_blocks]` | 空 | 每请求持久（行宽按 max_model_len 预分配） |
| KV cache | 启动时已分配 | 全 0 | 持久 |

此时**还没有分配任何 block**；`slot_mapping` 也还不存在。

### 2.2 Prefill（step 1）

调度器给请求分配 1 个物理 block，假设 `block_id=42`；`block_size=16`，所以这个 block 覆盖 16 个 slot：

```
slot(p) = block_table[p // 16] * 16 + (p % 16)
        = 42 * 16 + p
        = 672 + p
```

| 逻辑位置 p | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| slot | 672 | 673 | 674 | 675 | 676 | 677 |

**临时张量（每层都会重新算）**：

以 GQA（Qwen 风格，`Hq=24, Hkv=2, D=256`）为例：

| 张量 | shape | 内容 |
|---|---|---|
| `hidden_states` | `[6, 2560]` | 上一层输出（Qwen hidden=2560；第 0 层是 embedding） |
| `q` | `[6, 24, 256]` | 6 个 token × 24 个 Q 头 |
| `k` | `[6, 2, 256]` | 6 个 token × 2 个 KV 头 |
| `v` | `[6, 2, 256]` | 同上 |
| `attn_out` | `[6, 24, 256]` | 本层 attention 输出 |
| `logits` | `[1, vocab]` | 只取最后一个位置（不要求 prompt logprobs 时） |

**写入 cache**：

```
slot_mapping = [672, 673, 674, 675, 676, 677]   # [6] int32
reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)
```

每层的 cache 视图（当前 vLLM 逻辑布局）：

```
GQA:  [num_blocks, Hkv=2, block_size=16, C=512]   # C 里包含 K 256 + V 256
MLA:  [num_blocks, 1,     block_size=16, C=576]   # 512 latent + 64 rope
```

对 `block_id=42`、`layer=0`，写入的是：

```
# 当前逻辑视图 [B, H, N, C] 把 K/V 打包在 C 里；
# 某些 kernel 会 split 成两个 [B, H, N, 256] 视图：
k_view[42, :, 0:6, :]  ← k[0:6]
v_view[42, :, 0:6, :]  ← v[0:6]
```

layer 1 写的是**另一块** cache（不同的 `layer_stride`），但同样使用 block 42 和 slot 672..677。

### 2.3 采样（step 2，真实 vLLM 中与 prefill 同一个 engine step）

> vLLM 的实际执行是 `execute_model`（prefill forward）→ `sample_tokens`（lm_head + sampler），两者在同一个 `EngineCore.step` 内；这里拆成两步只是为了分别看清 KV 和 logits 的 shape。

- 只取最后一个位置（pos 5）的 hidden state → `lm_head` → `logits [1, vocab]`；
- 采样得到第一个输出 token；
- **6 个 prompt token 的 KV 全部保留**，不释放。

### 2.4 Decode 一个 token（step 3）

新 token 的 position = 6，slot = 672 + 6 = 678。

| 张量 | shape | 内容 |
|---|---|---|
| `input_ids` | `[1]` | 新 token |
| `positions` | `[1]` | `[6]` |
| `slot_mapping` | `[1]` | `[678]` |
| `block_table` | `[1, max_blocks]` | `[[42, …]]`（没变） |
| `seq_lens` | `[1]` | `[7]` |
| `q` | `[1, 24, 256]` | 只算新 token |
| `k / v` | `[1, 2, 256]` | 只算新 token |

Attention 读 `block 42` 的 slot 672..678，共 7 个位置；写只写 678。

**一句话：decode 每层只新增 1 个 K/V，但 attention 要读历史全部 KV。**

## 3. 临时 tensor vs 持久 tensor

| 张量 | 谁创建 | 生命周期 | 备注 |
|---|---|---|---|
| `input_ids` / `positions` | processor / input batch | 每步 | 复用 buffer，内容覆盖 |
| `slot_mapping` | `BlockTables.compute_slot_mappings()` | 每步 | shape `[num_groups, max_num_tokens]`，每 group 一行 |
| `block_table` | `BlockTables` / scheduler | 每请求 | 物理 block id；group 内所有层共享 |
| `q/k/v` | 每层 projection | 每层 forward | 第 0 层算完就释放/复用 |
| `attn_out` | 每层 attention | 每层 | 传给 next layer |
| `logits` | `lm_head` | 每步 | 通常只算最后位置 |
| **KV cache** | 启动时 `allocate_kv_cache()` | 整个服务 | 每层一份，按 block 分页 |
| `seq_lens` / `num_computed_tokens` | scheduler / runner | 每请求 | 决定读多少历史 |

## 4. 每层都复用同一份 KV 吗？

**默认不是。** 每层有自己的 K/V（不同权重算出来的），不能复用。  
“复用”的准确说法是：

| 对象 | 同 group 的层是否共享 |
|---|---|
| block id（42） | 是 |
| block table | 是 |
| slot_mapping | 是 |
| seq_lens / num_computed_tokens | 是 |
| KV 内容 | **否**（每层独立 page） |
| KDA/GDN state | 否（每层独立 state） |

为什么共享 block id 不会串内容？因为每层 page 的物理地址不同：

```
layer 0 page 42: offset + 0 * layer_stride + 42 * block_stride
layer 1 page 42: offset + 1 * layer_stride + 42 * block_stride
```

`block_table` 只存 42；kernel 用 `layer_stride` 找到“这一层的 42”。

## 5. 为什么 group 是“跨层”的？

因为 **一个 block id 必须同时代表 group 内每一层的一个 page**。

调度器分配 `block 42` 时，不是只给 layer 0 分配，而是给 group 内所有层各分配一页；这样：

- 一张 block table 就能服务几十层；
- 分配/回收只需要操作一个 id；
- 所有层的 `slot_mapping` 完全一致（同一批 token 在同一位置）；
- `bytes_per_block = sum(page_size for layer in group)`，一次算清。

如果每层独立分配，就会有 L 张 block table、L 套碎片和 L 次调度，复杂度爆炸。  
所以 vLLM 按“生命周期/形状相同”把层分组：

- 全 attention 层 → 一个 group；
- SWA 层 → 另一个 group（窗口不同，页的生命周期不同）；
- KDA/GDN → 另一个 group（state 不按 token 增长）；
- compressed/indexer/ring → 各自 group。

**group 跨层，是为了“元数据共享 + 一次分配”，不是为了“内容共享”。**

## 6. 真正的跨层内容共享：DeepSeek V4.1

普通 GQA/MLA 不共享内容。但 DeepSeek V4.1 是例外：

- 只有 `kv_source_layer_ids=[2,8,14,20]` 拥有 compressed KV；
- layer 3–7 通过 `compressed_cache_prefix` 读 layer 2 的 cache；
- 这些 consumer 层不写自己的 compressed cache（`get_kv_cache_spec()` 返回 None）。

所以 V4.1 的 “group” 里既有“每层独立 SWA”，也有“source 层共享 compressed”。  
读代码时看到 `is_kv_source` / `compressed_cache_prefix`，就知道这是模型级共享，不是通用 group 行为。

## 7. 用请求演示页走一遍

按四个步骤走一遍：

1. **请求到达**：只有 token，没有 cache；
2. **Prefill**：block 42 分配，slot 672..677，六层各写一份 page；
3. **采样**：只取最后位置 logits，KV 保留；
4. **Decode**：新 token slot 678，每层追加 1 个。

每一步只问四个问题：

```
这个张量 shape 是什么？
它是临时还是持久？
谁写它？
谁读它？
```

然后做三个练习：

1. 用纸画 6 个 token、1 个 block、6 层 page；把 slot 填进去。
2. 把 `block_size` 从 16 改成 32，重新算 slot / block 数。
3. 把 GQA 切到 MLA，观察 `k/v` 变成 `kv_c + k_pe`，cache 从 `[B,2,16,512]` 变成 `[B,1,16,576]`。

## 8. 自测

1. T=6，`block_size=16`，`block_table=[42]`。pos 5 的 slot 是多少？
2. Decode 第 7 个 token 时，`slot_mapping` 的 shape 和内容？
3. layer 0 的 page 42 和 layer 1 的 page 42 是同一块显存吗？
4. 同 group 的层共享哪些东西？至少说 3 个。
5. 为什么 KDA 层不能和 attention 层放同一个 group？
6. DeepSeek V4.1 的 layer 3 有没有自己的 compressed KV？它的 cache 从哪来？
7. MLA 的 `kv_cache` 为什么是 `[B,1,N,576]`，而不是 `[B,Hkv,N,D]`？
8. 如果 `block_size=32`，T=6，pos 5 的 slot 是多少？

### 答案

1. `42*16+5=677`。
2. `[1]`，内容 `[678]`（`42*16+6`）。
3. 不是；同一 block id，不同 `layer_stride` 偏移。
4. block id、block table、slot_mapping、seq_lens/num_computed_tokens（任说 3 个）。
5. KDA 的 state 不按 token 增长，`MambaSpec` 的 page 生命周期/形状与 attention 不同，不能共享 block table。
6. 没有；它是 consumer，通过 `compressed_cache_prefix` 读 source 层（最近的上游 2）的 cache。
7. MLA 只有 1 个 latent head（`num_kv_heads=1`），576 = 512 latent + 64 rope；所有 Q 头共享这份 latent。
8. `42*32+5=1349`（如果物理 block id 仍是 42，且 block_size=32）。
