# KV Cache 速查表

> 用一张表装下“模型 → spec → group → tensor → slot”的主干。

## 1. 四模型总表

| | DeepSeek V4.1 | Kimi K3 | Qwen3.8 Flash Next | GLM-5.3 Flash |
|---|---|---|---|---|
| 总参 / 激活 | 552B / 8B 输入、16B 输出 | 2.8T / 104B | ≈180B / 6B | 320B / 18B |
| 层数 | 40 + 3 MTP | 93 | 48 | 45 + 1 MTP |
| attention | 40 层 sparse MLA | 24 层 MLA | 12 层 GQA | 11 层 sparse MLA |
| linear | 无 | 69 层 KDA | 36 层 GDN | 34 层 KDA |
| 主 KV 行宽 | 512（latent+rope） | 576（512+64） | 2 KV heads × 256 | 512（无 rope） |
| 跨层共享 | **4 个 source** | 否 | 否 | 否 |
| 额外 cache | SWA 128、indexer、compressor | 无 | QSA compressed/ring | indexer kpool/tail |

## 2. 四种 `KVCacheSpec`

| spec | 增长 | `block_size` | 关键字段 | 谁在用 |
|---|---|---|---|---|
| `FullAttentionSpec` | O(S) | token block | `num_kv_heads`, `head_size`, `head_size_v` | Qwen GQA |
| `MLAAttentionSpec` | O(S) | token block | `tokens_per_state`, `head_size`（通常 512/576） | Kimi/GLM/DS compressed |
| `SlidingWindowMLASpec` | O(S) 但只留窗口 | token block（DS=32） | `sliding_window`, `bounded_replay` | DSV4.1 SWA |
| `MambaSpec` | **O(1)** | state block | `shapes`, `dtypes`, `tp_replicated` | KDA/GDN |
| `CircularBufferSpec` | **O(1)** | ring 容量 | 1 block/req, no slot mapping | compressor/indexer tail/QSA ring |

## 3. 七步链路

```
get_kv_cache_spec()                         # 每层声明
  → GPUWorker.get_kv_cache_spec()           # 收集 AttentionLayerBase
  → get_kv_cache_groups()                   # 按 spec/page size 分组
  → get_kv_cache_config_from_groups()       # bytes_per_block, num_blocks, tensors
  → allocate_kv_cache()                     # int8 backing + as_strided
  → bind_kv_cache_to_layers()               # 每层 4D view
  → BlockTables.compute_slot_mappings()     # 运行时 slot
```

## 4. 五个公式

```
page_size_bytes   = num_heads × (block_size / tokens_per_state) × content_bytes
bytes_per_block   = max over groups( sum(page_size for layer in group) )
num_blocks        = available_kv_memory // bytes_per_block
slot(p)           = block_table[p // B] * B + (p % B)
ring_index(p)     = p % ring_capacity          # CircularBufferSpec
```

## 5. 四模型预期 group

| 模型 | 预期 group |
|---|---|
| DSV4.1 | SWA / compressed r2 / compressed r1 / indexer r2 / indexer r1 / compressor |
| Kimi | MLA / KDA |
| Qwen | GQA / QSA compressed / QSA ring / GDN / PLE |
| GLM | sparse MLA / indexer kpool / indexer tail / KDA |

## 6. 五个易错点

1. `page_size_bytes` 是**每层**的；`bytes_per_block` 才乘层数。
2. `block_table` 的真实 buffer 是 `[max_reqs, max_blocks]`，不是 `[1,1]`。
3. `slot_mapping` 的真实 buffer 是 `[num_groups, max_tokens]`；每层拿到的是一行切片。
4. 同 group 的层共享 block id / block table / slot_mapping，**不共享 KV 内容**。
5. `CircularBufferSpec` 的 `block_size` 是 ring 容量；`uses_slot_mapping=False`、`prefix_cacheable=False`、1 block/req。
