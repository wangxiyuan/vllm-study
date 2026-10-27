# 验证与调试

> 下面的数值按 config 推导；真正部署时，再把 print 补丁加到 vLLM 里核对。

## 1. 四模型量级参考

按 config 直接推导的近似值（bf16，KV 按每请求、TP 仅影响 state 分片）：

| 模型 | 每 token 主 KV | 长上下文估算 | 固定 state（TP=8，每 rank） |
|---|---|---|---|
| DeepSeek V4.1 | SWA 584 B/层；compressed 584 B/state | 128k ≈ 249 MiB（含 indexer） | compressor ring，几 MB |
| Kimi K3 | 24 层 × 576 × 2 = 27 KiB | 128k ≈ 3.38 GiB | 69 层 KDA ≈ 54 MiB |
| Qwen3.8 | 12 层 × 2 heads × 512 × 2 = 24 KiB | 262k ≈ 6.0 GiB | 36 层 GDN ≈ 14 MiB |
| GLM-5.3 | 11 层 × 512 × 2 = 11 KiB | 128k ≈ 1.38 GiB | 34 层 KDA ≈ 18 MiB |

这些数字只用于建立量级；真实值受 FP8、block 对齐、前缀共享、TP 复制影响。

## 2. 手算三个数

1. **page_size**：`block_size=32`，MLA `tokens_per_state=2`，`head_size=512`，FP8（1 B），求一个 page 的字节。
2. **bytes_per_block**：一个 group 有 3 层，每层 page 64 KiB，求 bytes_per_block。
3. **slot**：`B=16`，`block_table=[4,2,7]`，`pos=45`，求 slot。

答案：

1. `1 head × (32/2) states × 512 = 8192 B`。
2. `3 × 64 KiB = 192 KiB`。
3. `block_idx=2`，`offset=13`，`slot=7×16+13=125`。

## 3. 代码阅读顺序

不看 forward，先看这四个位置：

| 目标 | 位置 |
|---|---|
| 分组决策 | `vllm/v1/core/kv_cache_planning.py:1113` |
| group 合并 | 同文件 `:187` 的 `create_kv_cache_group_specs()` |
| 混合类型分组 | 同文件 `:416` 的 `_get_kv_cache_groups_uniform_page_size()` |
| 多 group 协调 | `vllm/v1/core/kv_cache_coordinator.py:609` |

然后对照：

| 目标 | 位置 |
|---|---|
| `UniformTypeKVCacheSpecs` | `vllm/v1/kv_cache_interface.py:1235` |
| `KVCacheGroupSpec` | 同文件 `:1449` |
| `block_table` 宽度 | `vllm/v1/worker/gpu/model_runner.py:621` |
| slot mapping | `vllm/v1/worker/gpu/block_table.py:194` |

读每个函数时只回答三件事：输入是什么、返回什么、哪个字段不同会导致 group 数变化。

## 4. 运行时打印 spec / group

在 `vllm/v1/worker/gpu/attn_utils.py:139` 的 `get_kv_cache_spec()` 返回前：

```python
for name, spec in kv_cache_spec.items():
    print("SPEC", name, type(spec).__name__,
          "block", spec.block_size,
          "heads", getattr(spec, "num_kv_heads", None),
          "head_size", getattr(spec, "head_size", None),
          "tokens_per_state", getattr(spec, "tokens_per_state", None),
          "page", spec.page_size_bytes)
return kv_cache_spec
```

在 `vllm/v1/core/kv_cache_planning.py:1113` 的 `get_kv_cache_groups()` 返回前：

```python
for i, g in enumerate(groups):
    print("GROUP", i, type(g.kv_cache_spec).__name__, g.layer_names)
```

在 `vllm/v1/worker/gpu/model_runner.py:621` 的 `initialize_kv_cache()` 里：

```python
print("NUM_BLOCKS", kv_cache_config.num_blocks)
for t in kv_cache_config.kv_cache_tensors:
    print("TENSOR", t.layers, t.offset, t.layer_stride, t.block_stride)
```

## 5. 对照公式

拿到 `num_blocks`、`page_size` 后核对：

```
bytes_per_block = max over groups( sum(page_size for layer in group) )
num_blocks      = available_kv_bytes // bytes_per_block
```

对不上时依次检查：

1. `page_size_padded` / `alignment` 是否改了 page；
2. group 是否是 `UniformTypeKVCacheSpecs`（page 要逐层求和）；
3. `kernel_block_size` 是否让 tensor 第 0 维翻倍；
4. 是否走了 HiSparse / host pool 的独立预算。

## 6. 三个边界条件

- `block_size` 必须能被 backend 的 `kernel_block_size` 整除；
- 可前缀缓存 group 的 `block_size` 必须能被 `hash_block_size` 整除；
- `CircularBufferSpec` 的 ring 容量必须能整除 attention block size。
