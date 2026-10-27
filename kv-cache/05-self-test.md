# KV Cache 专项自测 20 题

> 闭卷完成以下 20 题。

## A. 模型结构（1–8）

1. DeepSeek V4.1 有 40 层，为什么 `get_kv_cache_spec()` 只返回 4 个 compressed spec？其余层的 compressed KV 从哪来？
2. DeepSeek V4.1 每一层都有的 cache 是什么？谁负责它的 spec？
3. `get_kv_cache_spec()` 返回 `None` 有哪三种常见含义？
4. Kimi K3 的 69 层 KDA 用什么 spec？它的显存随 `max_model_len` 增长吗？
5. Kimi K3 的 MLA cache 每 token 多少维？`mla_use_nope=true` 会影响这个维度吗？
6. GLM-5.3 的 config 里 `head_dim=0`，它的 MLA cache 实际 head size 是多少？为什么？
7. Qwen3.8 的 full attention 是 MLA 还是 GQA？写出它每 token 每层的 KV 字节公式。
8. Qwen3.8 的 QSA compressed key cache 的 `tokens_per_state` 是多少？它和主 KV 的关系是什么？

## B. 字节/几何（9–13）

9. 写出 `MLAAttentionSpec.page_size_bytes` 的公式（含 `tokens_per_state`）。
10. 写出 `bytes_per_block` 的计算方式。为什么不是“单层 page size”？
11. 写出 `num_blocks` 的计算方式（用 `available_memory`）。
12. `create_kv_cache_views()` 生成的逻辑形状是什么？`C` 是元素数还是字节数？
13. 一个 `KVCacheTensor` 有 3 层，每层 `page_size=64 KiB`，`num_blocks=1000`，求 `bytes_per_block`、`tensor.size`、总 KV 字节。

## C. 分组/后端（14–18）

14. 一个 KV `group` 和一个 `AttentionGroup` 的区别是什么？
15. `get_kv_cache_groups()` 的三条主要路径是什么？
16. `UniformTypeKVCacheSpecs` 解决什么问题？它允许 head size 不同吗？
17. `CircularBufferSpec` 为什么 `uses_slot_mapping=False`？它怎么定位？
18. `kernel_block_size < block_size` 时，tensor 的第 0 维怎么变？谁负责映射回 manager block？

## D. 运行时（19–20）

19. 写出 slot mapping 公式。`B=16`、`block_table=[4,2,7]`、`pos=45`，slot 是多少？
20. 为什么多 group 模型的前缀缓存要满足 `block_size % hash_block_size == 0`？哪些 group 可以例外？

---

## 参考答案

1. 只有 `kv_source_layer_ids=[2,8,14,20]` 是 source，拥有 compressed KV；其余层通过 `compressed_cache_prefix` 读对应 source 的 cache（`max(s for s in sources if s <= layer_id)`）。
2. SWA cache（窗口 128，`SlidingWindowMLASpec`）；由独立的 `DeepseekV4SWACache` 模块声明（`sparse_swa.py:120`）。
3. ①这层不拥有 KV（共享 source）；②`kv_sharing_target_layer_name` 指向别层；③encoder-only / 不需要 cache。
4. `MambaSpec`（conv + recurrent state）；不随 `max_model_len` 增长，只随 state 形状/TP/投机 block 变。
5. `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`；`mla_use_nope=true` 不改变 `head_size` 的计算，仍然按 576 分配。
6. `512` = `kv_lora_rank + qk_rope_head_dim = 512 + 0`；`qk_nope_head_dim=256` 是 Q/K 的 nope 投影维度，不是 cache 行宽。
7. GQA。公式：`12 层 × 2 KV heads × (K 256 + V 256) × dtype_bytes`；bf16 时 `12×2×512×2 = 24,576 B/token`。
8. `tokens_per_state=indexer_compress_ratio=4`；它是 full attention 层的 side cache（indexer 的压缩 key），不是主 K/V，不参与主 attention 的 PV。
9. `num_heads × (block_size / tokens_per_state) × state_content_size_bytes`；`state_content = (head_size + head_size_v) × dtype_bytes`，或 `state_content_bytes` 覆盖（如 fp8_ds_mla 的 584/656）。
10. `bytes_per_block = max over groups( sum(page_size_bytes for layer in group) )`；因为一个 block id 要覆盖 group 内所有层的 page，且所有 group 共享同一 block 编号空间。
11. `num_blocks = available_memory // bytes_per_block`（`kv_cache_planning.py:681`）。
12. 逻辑 `[B, H, N, C]`；`B = num_blocks * (block_size/kernel_block_size)`，`N = block_size/tokens_per_state`（kernel 粒度），`C` 是**字节数**，最后 `.view(dtype)` 变元素。
13. `bytes_per_block = 3×64 KiB = 192 KiB`；`tensor.size = 192 KiB × 1000 = 187.5 MiB`；总 KV = 187.5 MiB（这个 tensor 本身）。
14. KV group 共享**同一张 block table**（`KVCacheGroupSpec`）；AttentionGroup 是同一 KV group 内按 `(backend, spec, num_q_heads)` 再拆的 metadata 组，可以有多个，但 block table 相同。
15. ①所有 spec 可 merge → 1 组；②同一类型 → `UniformTypeKVCacheSpecs` → 1 组；③混合类型 → 按 spec/page size 分多组（packed / uniform page size / fallback）。
16. 解决“同一类型但 head size/每层 page 不同”仍想共享一个 group/block table 的问题；允许 head size 不同，但要求 block_size、dcp_sharded、token slot 数兼容。
17. ring 由层自己按 `pos % ring` 寻址，只需要每请求 1 个 block；不需要 per-token slot，也不参与前缀缓存。
18. 第 0 维变成 `num_blocks * (block_size/kernel_block_size)`；`map_kv_caches_to_kernel_blocks()` 把 kernel 粒度视图按 manager block 分组。
19. `slot = block_table[pos//B]*B + pos%B`；`pos=45`，`block_idx=2`，`offset=13`，`slot=7*16+13=125`。
20. 前缀 hash 以 `hash_block_size` 为粒度；每个可缓存 group 的 block 必须能整除 hash 粒度，否则 hash 边界和 block 边界对不上。`CircularBufferSpec`（`prefix_cacheable=False`）可以例外，例如 GLM 的 indexer tail。
