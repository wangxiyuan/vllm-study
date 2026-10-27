# KV Cache 全景：四个模型与七步链路

> 代码基线：`vllm-project/vllm` @ `4e964b2623`（2026-10-17）。

## 0. 为什么 KV cache 是核心

自回归生成时，每个新 token 都要和历史上所有 token 做 attention。历史 token 的 K/V 不会变，所以缓存它们，decode 每步只算新 token 的 K/V，再读缓存。这个缓存就是 KV cache。

它同时决定三件事：

- **显存容量**：能同时服务多少条请求、多长上下文；
- **带宽**：decode 每步要搬多少字节，直接决定 token/s；
- **调度**：块怎么分、前缀怎么复用、谁被抢占。

这四个模型没有一个是“纯 full attention”：

| 模型 | 参数量（总/激活） | 结构 | KV cache 的本质 |
|---|---|---|---|
| DeepSeek-V4.1-Flash | 552B / 8B 输入、16B 输出 | 40 层 sparse MLA；**跨层共享 compressed KV** | 每层一份 SWA（窗口 128）+ 4 个 source 层的 compressed KV + 8 个 indexer KV + compressor state |
| Kimi K3 | 2.8T / 104B | 93 层：24 层 MLA + 69 层 KDA | 24 份 MLA KV（576 维/头）+ 69 份**固定大小**的 KDA recurrent state |
| Qwen3.8-Flash-Next | ≈180B（125B 主干 + 51B n-gram + 4B MTP）/ 6B | 48 层：12 层 full attention + 36 层 GDN | 12 份 GQA KV（2 KV heads × 256）+ 12 份 QSA 压缩 key/ring + 36 份 GDN state |
| GLM-5.3-Flash | 320B / 18B | 45 层：11 层 sparse MLA + 34 层 KDA | 11 份 MLA KV（512 维）+ 11 份 indexer kpool cache + tail ring + 34 份 KDA state |

因此必须同时掌握四种 `KVCacheSpec`：

- `MLAAttentionSpec` / `SlidingWindowMLASpec`：按 token 线性增长的 paged KV；
- `FullAttentionSpec`：Qwen 的 GQA 主 KV；
- `MambaSpec`：不随序列长度增长的 recurrent state（KDA/GDN）；
- `CircularBufferSpec`：每请求固定一个 ring block（compressor/indexer tail/QSA ring）。

## 1. 四个模型的关键差异

1. **DeepSeek V4.1 的 KV 不在每一层都存一份。**  
   `kv_source_layer_ids=[2,8,14,20]`，只有这 4 层拥有 compressed KV；其余层通过 `compressed_cache_prefix` 读 source 的 cache。每层另有自己的 SWA cache（窗口 128）。这是“跨层 KV 共享 + 分层压缩”，和传统 MLA 完全不同。
2. **Kimi K3 的 69 层 KDA 不是 attention。**  
   它们没有 paged KV，只有固定大小的 conv state + recurrent state（`MambaSpec`）。序列从 8k 到 1M，这 69 层的 cache 大小不变。
3. **Qwen3.8 的 full attention 是 GQA，不是 MLA。**  
   12 层 full attention：`num_kv_heads=2, head_dim=256`；另外有 QSA 的压缩 key cache（`tokens_per_state=4`）和 ring（`CircularBufferSpec`），以及 36 层 GDN state。
4. **GLM-5.3 的 sparse attention 是“无 RoPE 的 MLA”。**  
   `qk_rope_head_dim=0`、`kv_lora_rank=512`、`qk_nope_head_dim=256`；每层 512 维 latent。indexer 用 `tokens_per_state=index_kpool=4`，tail 用一个 `CircularBufferSpec` 保存未满 pool 的 raw K。

## 2. vLLM 的七个环节

1. **层声明 spec**：每个 attention/cache 层实现 `get_kv_cache_spec()`，返回自己的 `KVCacheSpec` 或 `None`。`None` 表示这层不拥有 KV（共享 source 或 encoder-only），不代表没有 cache。
2. **worker 收集**：`GPUWorker.get_kv_cache_spec()` 遍历所有 `AttentionLayerBase`，跳过 KV-sharing 层，并用 backend 的 `customize_attention_spec()` 修正布局。
3. **core 分组**：`get_kv_cache_groups()` 把 layer spec 合并成 `KVCacheGroupSpec`。所有层可 merge → 1 组；同类型 → `UniformTypeKVCacheSpecs`；混合类型 → 按 spec/page size 拆组。
4. **算显存与布局**：`get_kv_cache_config_from_groups()` 算 `bytes_per_block`、`num_blocks`，并为每个 group 生成 `KVCacheTensor` 的 offset/layer_stride/block_stride。
5. **建 tensor**：`allocate_kv_cache()` 分配一块 int8 backing，`create_kv_cache_views()` 用 `as_strided` 生成 `[L, B, H, N, C]` 逻辑视图，再拆成每层 4D `[B, H, N, C]`。
6. **绑定到层**：`bind_kv_cache_to_layers()` 调每层的 `bind_kv_cache()`；MLA squeeze 掉 head 维，Mamba 拆成 conv/recurrent view。
7. **运行时寻址**：`BlockTables.compute_slot_mappings()` 为每个 token 算 `slot`；attention metadata builder 按 group 生成 block table、seq_lens、slot_mapping。

## 3. 关键对象

| 对象 | 作用 |
|---|---|
| `KVCacheSpec` | 一层 cache 的“类型 + 形状 + 生命周期”声明 |
| `AttentionSpec` | paged attention 的通用字段（heads/head_size/dtype/tokens_per_state） |
| `MLAAttentionSpec` | MLA latent cache（`head_size_v=0`，通常 1 个 KV head） |
| `SlidingWindowMLASpec` | 带窗口的 MLA cache |
| `MambaSpec` | KDA/GDN 的固定 state（conv + recurrent） |
| `CircularBufferSpec` | 每请求 1 个 ring block，不参与前缀缓存 |
| `KVCacheGroupSpec` | 共享同一 block table 的一组层 |
| `KVCacheConfig` | `num_blocks` + 每个 group + 每个 tensor 的布局 |
| `KVCacheTensor` | 一段 backing allocation 的 offset/layer_stride/block_stride |
| `BlockPool` / `KVCacheBlock` | 物理 block / ref_cnt / LRU |
| `HybridKVCacheCoordinator` | 多 group 的分配/前缀命中协调 |
| `AttentionGroup` | 同一 backend + spec + Q 头数的层组 |
| `BlockTables` | 每 group 一张 block table + slot mapping |

## 4. 主链路

```
get_kv_cache_spec()                         # 每层声明
  → GPUWorker.get_kv_cache_spec()           # 收集 AttentionLayerBase
  → get_kv_cache_groups()                   # 按 spec/page size 分组
  → get_kv_cache_config_from_groups()       # bytes_per_block, num_blocks, tensors
  → allocate_kv_cache()                     # int8 backing + as_strided
  → bind_kv_cache_to_layers()               # 每层 4D view
  → BlockTables.compute_slot_mappings()     # 运行时 slot
```

## 5. 三个关键入口

- **先看 `get_kv_cache_spec()`**：它决定这层有没有 cache、cache 长什么样、是否共享。
- **再看 group**：group 决定 block table 和分配粒度；同 group 共享 block id，不共享 KV 内容。
- **最后看 `slot_mapping`**：它把“逻辑位置”变成“物理 slot”，是运行时读写 cache 的桥。
