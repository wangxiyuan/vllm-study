# vLLM KV Cache 落地全链路

> 从一个模型出发，沿 7 个环节看它的 KV cache 如何从声明走到 kernel。
> 行号会漂移，用函数名定位。

## 0. 七步总览

```
① 层声明 spec         layer.get_kv_cache_spec()
② worker 收集         GPUWorker.get_kv_cache_spec()
③ core 分组           get_kv_cache_groups()
④ core 算显存/布局    get_kv_cache_config_from_groups()
⑤ worker 建 tensor    allocate_kv_cache() → create_kv_cache_views()
⑥ 绑定到层            bind_kv_cache_to_layers() → layer.bind_kv_cache()
⑦ 运行时寻址          BlockTables.compute_slot_mappings() + attention metadata
```

每一步的输入、输出和关键变量如下。

同一个模型在七步里会以不同形态出现：

| 步骤 | 模型侧的对象 | 这一步之后变成什么 |
|---|---|---|
| ① 层声明 | `model.layers[i]` 里的 attention / linear / indexer / compressor 模块 | 每条模块返回一个 `KVCacheSpec`，或 `None` |
| ② worker 收集 | 所有 `AttentionLayerBase` 实例 | `dict[layer_name, KVCacheSpec]`，条目数可能大于层数 |
| ③ core 分组 | 上一步的 dict | `list[KVCacheGroupSpec]`，每个 group 对应一张 block table |
| ④ 算显存 | group 列表 | `bytes_per_block`、`num_blocks`、`KVCacheTensor` |
| ⑤ 建 tensor | `KVCacheTensor` + spec | 每层一个 4D view，指向同一块 int8 backing |
| ⑥ 绑定 | 4D view + layer 对象 | `layer.kv_cache` |
| ⑦ 运行时 | block table + slot mapping | kernel 按 slot 读写每层 page |

关键是：**模型层数不等于 spec 条目数**。一层可能注册多个 cache 模块（主 KV + indexer + ring），也可能一个都不拥有（consumer / KV-sharing）。

---

## ① 层声明：`get_kv_cache_spec()`

每个 attention/cache 层实现 `AttentionLayerBase.get_kv_cache_spec(vllm_config)`，返回自己的 `KVCacheSpec` 或 `None`。

典型实现：

- MLA：`vllm/model_executor/layers/attention/mla_attention.py:1357`。
- DeepSeek V4.1：`vllm/models/deepseek_v41/attention.py:1195`。
- DeepSeek V4.1 SWA：`vllm/v1/attention/backends/mla/sparse_swa.py:120`。
- Kimi MLA：`vllm/models/kimi_k3/nvidia/mla.py:417`。
- Kimi KDA：`vllm/models/kimi_k3/nvidia/kda.py:830`。
- GLM indexer：`vllm/models/glm5next/common/attention.py:112`。
- GLM tail：`:156`。
- Qwen full attention：`vllm/models/qwen4_exp/nvidia/qsa.py:552`。
- Qwen QSA compressed/ring：`vllm/models/qwen4_exp/common/qsa_cache.py:868`、`:844`。
- GDN：`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:705`。

**返回 `None` 的含义**：

- 这层不拥有 KV（例如 DeepSeek V4.1 的非 source 层）；
- 或者这层共享别的层的 cache（`kv_sharing_target_layer_name`）；
- 或者 encoder-only 层。

注意：DeepSeek V4.1 的非 source 层“没有 compressed cache”，但仍然有独立的 SWA cache 模块（`DeepseekV4SWACache`），所以 worker 会看到两条 layer 记录：`...self_attn`（返回 None）和 `...self_attn.swa_cache`（返回 `SlidingWindowMLASpec`）。

### 1.1 一层为什么会有多条 spec

以 Qwen3.8 的一个 full attention 层为例（层号 3、7、…、47）：

```
model.layers.3.self_attn          → FullAttentionSpec（GQA 主 KV）
model.layers.3.self_attn.qsa      → MLAAttentionSpec（QSA compressed key）
model.layers.3.self_attn.qsa_ring → CircularBufferSpec（未满 group 的 ring）
```

这三条记录的 layer name 前缀相同，但 key 不同，所以 `kv_cache_spec` 里是三条 spec，而不是一条。后面的分组会把它们拆到三个 group：主 KV 按 token 增长，QSA compressed 按 4 个 token 一个 state，ring 则每请求固定 1 block。

DeepSeek V4.1 的 source 层 2 更典型：

```
model.layers.2.self_attn           → MLAAttentionSpec（compressed，ratio 2）
model.layers.2.self_attn.swa_cache → SlidingWindowMLASpec（窗口 128）
model.layers.2.self_attn.indexer   → MLAAttentionSpec（indexer，head 128）
model.layers.2.self_attn.compressor→ CircularBufferSpec（压缩器 ring）
```

同一层贡献 4 条 spec。非 source 层只贡献 SWA 一条，`self_attn` 本身返回 `None`。Kimi 的 MLA 层贡献 `MLAAttentionSpec`，KDA 层贡献 `MambaSpec`；GLM 的 sparse MLA 层额外贡献 indexer kpool 与 tail ring。

---

## ② worker 收集：`GPUWorker.get_kv_cache_spec()`

`vllm/v1/worker/gpu/attn_utils.py:139`：

```python
def get_kv_cache_spec(vllm_config):
    kv_cache_spec = {}
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            continue
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            if isinstance(spec, AttentionSpec):
                spec = customize_attention_spec(attn_module.get_attn_backend(), spec)
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec
```

关键点：

1. **收集所有 `AttentionLayerBase`**，不只是 `Attention`；Mamba/KDA、indexer、SWA cache 都是独立模块。
2. **跳过 KV-sharing 层**：被共享的层不重复分配。
3. **`customize_attention_spec`**：让 backend 修改 spec，例如加 `block_stride_alignment`、调整 head size/quant layout。见 `vllm/v1/worker/utils.py:460` 附近。
4. 输出：`dict[layer_name, KVCacheSpec]`。

**变量**：`kv_cache_spec` 的 key 是完整 layer name（如 `model.layers.3.self_attn`、`model.layers.3.self_attn.swa_cache`、`model.layers.3.self_attn.indexer.k_cache`）。

---

## ③ core 分组：`get_kv_cache_groups()`

`vllm/v1/core/kv_cache_planning.py:1113`：

```python
if is_kv_cache_spec_uniform(kv_cache_spec):
    return _get_kv_cache_groups_uniform_spec(kv_cache_spec)       # 所有层一个 group
elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
    return _get_kv_cache_groups_uniform_type(uniform_spec)        # 同类型一个 group
...
groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)    # 按 spec/page size 分组
```

三条路径：

1. **所有层 spec 可 merge**（典型纯 full attention）：一个 group，所有层共享一张 block table。
2. **同一类型、但 head size 不同**：一个 `UniformTypeKVCacheSpecs` group；`page_size_bytes` 是该 group 内每层 page 之和。
3. **混合类型**（KDA + MLA、SWA + compressed）：先按 spec 分桶，再尝试按 page size 合并；不能合并就多 group。

分组单位是 `KVCacheGroupSpec`（`vllm/v1/kv_cache_interface.py:1449`）：

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False
    host_resident: bool = False
    ...
```

**一个 group = 一张 block table = 一个 `SingleTypeKVCacheManager`。**
同一 group 的层共享 block id 序列；不同 group 可以有不同的 `block_size`、`page_size`、`max_num_blocks_per_req`。

对四个模型的预期：

| 模型 | 预期 group |
|---|---|
| DSV4.1 | SWA、compressed-ratio2、compressed-ratio1、indexer-ratio2、indexer-ratio1、compressor-state（约 6） |
| Kimi K3 | MLA、KDA（2） |
| Qwen3.8 | GQA、QSA-compressed、QSA-ring、GDN、PLE（约 5） |
| GLM-5.3 | MLA、indexer-kpool、indexer-tail、KDA（4） |

> 实际 group 数由 `merge()` 和 page size 统一结果决定；以 `KVCacheConfig.kv_cache_groups` 为准。

### 3.1 从模型层到 group 的数量关系

| 模型 | 层数 | spec 条目 | group | 条目为什么多于/少于层数 |
|---|---:|---:|---:|---|
| DeepSeek V4.1 | 40 | 56 | 6 | 每层都有 SWA；4 个 source 另带 compressed + indexer + compressor |
| Kimi K3 | 93 | 93 | 2 | 每层恰好一种 cache：MLA 或 KDA |
| Qwen3.8 | 48 | 73 | 5 | 12 个 full attention 层各带 GQA + QSA compressed + QSA ring |
| GLM-5.3 | 45 | 67 | 4 | 11 个 sparse MLA 层各带主 MLA + indexer kpool + tail ring |

分组过程可以概括成三句话：

1. **先按 spec 类型分桶**：`FullAttentionSpec`、`MLAAttentionSpec`、`MambaSpec`、`CircularBufferSpec` 不会进同一个 group。
2. **同类型再按 page size / tokens_per_state / state shape 合并**：能 merge 的合成一个 group；不能 merge 的继续拆。
3. **每个 group 生成一张 block table**：group 内共享 block id，group 间独立分配。

Qwen3.8 的 73 条 spec 最后收缩成 5 个 group，是因为：

- 12 条 GQA spec 完全相同 → 1 组；
- 12 条 QSA compressed 完全相同 → 1 组；
- 12 条 QSA ring 完全相同 → 1 组；
- 36 条 GDN 完全相同 → 1 组；
- layer 2 的 PLE 是 `MambaSpec`，但 `tp_replicated`/shape 不同 → 单独 1 组。

分组实验可以逐层点开看这个收缩过程。

---

## ④ core 算显存与布局：`get_kv_cache_config_from_groups()`

`vllm/v1/core/kv_cache_planning.py:642`。

### 4.1 `bytes_per_block`

`_get_kv_cache_bytes_per_block()`（`:574`）：

```python
bytes_per_block = max(
    sum(spec.page_size_bytes for layer in group.layer_names)
    for group in kv_cache_groups
)
```

理解：

- 一个 block id 在**每个 group** 里都对应一段物理空间；
- group 内的所有层共享这个 block id，但每层都有自己的 page；
- 所以 group 的 block 字节 = 该 group 所有 layer 的 page 之和；
- 取所有 group 的最大值，保证每个 block id 都有足够空间。

### 4.2 `num_blocks`

`:681`：

```python
num_blocks = available_memory // bytes_per_block
```

`available_memory` 来自 worker 的显存 profiling（`vllm/v1/worker/gpu_worker.py:638-744`）：`requested_memory - 非KV峰值 - CUDA graph`。

### 4.3 `KVCacheTensor`

`:690` 之后为每个 group 的每种 spec 生成 `KVCacheTensor`：

```python
KVCacheTensor(
    size=size,                 # bytes_per_block * num_blocks
    layers=layer_names,
    layer_stride=...,
    block_stride=...,
    offset=byte_offset,
)
```

- `layer_stride`：同一 block 内，相邻 layer 的 page 间隔；
- `block_stride`：相邻 block 的间隔（block-outer 布局下等于 `bytes_per_block`）；
- `offset`：该 tensor 在共享 backing allocation 中的起点。

**所有 group/tensor 共享一块 `torch.zeros(buf_size, dtype=int8)`**（`vllm/v1/worker/utils.py:564`），group 之间可以 alias 同一段字节。注释说得很清楚：

```
group 0: | blk0 [ A | B | pad ] | blk1 [...]
group 1: | blk0 [ C | D ]       | blk1 [...]
```

### 4.4 关键变量

| 变量 | 含义 |
|---|---|
| `available_memory` | KV 可用字节（profiling 后） |
| `bytes_per_block` | 一个 block id 的物理字节 |
| `num_blocks` | 物理 block 数（所有 group 共用这个编号空间） |
| `KVCacheTensor.size` | `bytes_per_block * num_blocks` |
| `layer_stride` / `block_stride` / `offset` | 5D 视图的步长 |

---

## ⑤ worker 建 tensor：`allocate_kv_cache()` → `create_kv_cache_views()`

`vllm/v1/worker/gpu/attn_utils.py:398`：

```python
kv_caches = allocate_kv_cache(kv_cache_config, device, layout, kernel_block_sizes)
```

`allocate_kv_cache()`（`vllm/v1/worker/utils.py:564`）：

1. 校验所有 `KVCacheTensor.size` 相同；
2. 分配 `buf = torch.zeros(buf_size, dtype=torch.int8, device=device)`；
3. 对每个 `KVCacheTensor`，按 group/spec 调 `create_kv_cache_views()`；
4. 返回 `dict[layer_name, torch.Tensor]`（每个 layer 一个 4D view）。

`create_kv_cache_views()`（`vllm/v1/kv_cache_interface.py:381`）：

```python
shape_bytes = compute_layer_kv_cache_shape_bytes(spec, num_blocks, kernel_block_size)
# (num_blocks * blocks_per_page, num_heads, num_states, content_bytes)

strides = compute_layout_strides(spec, num_blocks, num_layers, layout, ...)
view_5d = torch.as_strided(raw, size=(num_layers, *shape_bytes), stride=strides, ...)

for layer_idx in range(num_layers):
    cache_logical = view_5d[layer_idx]
    if dtype is not None:
        cache_logical = cache_logical.view(dtype)
    views.append(cache_logical)
```

逻辑形状（`kv_cache_interface.py:326`）：

```
[B, H, N, C] 其中 C 是“字节”
B = num_blocks * (spec.block_size // kernel_block_size)
H = spec.num_heads
N = spec.get_num_kernel_states(kernel_block_size)
C = spec.state_content_size_bytes
```

然后 `.view(dtype)` 把字节维变成元素维（例如 `C=1024` bytes → 512 个 bf16）。

**为什么要 kernel_block_size**：manager block 可能是 128，但后端 kernel 只支持 64；`blocks_per_page=2`，视图第 0 维变成 `num_blocks*2`，`map_kv_caches_to_kernel_blocks()` 再把 manager 粒度映射成 kernel 粒度。

**关键变量**：

- `kernel_block_size`：由 `prepare_kernel_block_sizes()` 选（见 ⑥）；
- `num_layers`：这个 `KVCacheTensor` 包含的层数；
- `layer_stride`/`block_stride`：决定 `as_strided` 的物理布局；
- `dtype`：`spec.dtype`，决定最后 `.view()`。

---

## ⑥ 绑定到层：`bind_kv_cache_to_layers()`

`vllm/v1/worker/utils.py:868`：

```python
for layer_name, kv_cache in kv_caches.items():
    forward_context[layer_name].bind_kv_cache(kv_cache)
```

每个层的 `bind_kv_cache` 自己决定怎么解释这个 4D view：

- MLA（Kimi/GLM/DeepSeek）：`[B, H=1, N, C] → [B, N, C]`（squeeze H）。
- Mamba/KDA：把一块 tensor 拆成 conv state / recurrent state（可能多个 view）。
- GLM indexer：`[B, N, H, C]` 或 squeeze。
- Qwen QSA：`transpose(1,2)` 后 split key/value。

**这就是“同一块物理显存，不同层有不同逻辑视图”的地方。**

---

## ⑦ 运行时：block table、slot mapping、attention metadata

### 7.1 每个 group 一张 block table

`GPUModelRunner.initialize_kv_cache()`（`vllm/v1/worker/gpu/model_runner.py:621`）在拿到 `KVCacheConfig` 后：

1. 为每个 group 算 `max_num_blocks_per_group`（`spec.max_num_blocks_per_req`）；
2. 调 `init_attn_backend()`（`attn_utils.py:198`）建 `AttentionGroup` + metadata builder；
3. 建 `BlockTables`（`model_runner.py:709`）：

```python
self.block_tables = BlockTables(
    block_sizes=[group.spec.block_size ...],
    max_num_blocks_per_group=[...],
    slot_mapping_enabled=[...],
    ...
)
```

### 7.2 slot mapping

`vllm/v1/worker/gpu/block_table.py:194` 的 `compute_slot_mappings()`，核心公式在 `:355`：

```python
block_indices = local_positions // kv_block_size
block_offsets = local_positions % kv_block_size
block_numbers = block_table[req_idx, block_indices]
slot_ids = block_numbers * kv_block_size + block_offsets
```

`PAD_SLOT_ID = -1` 给 padding / CUDA graph 占位。

**`CircularBufferSpec.uses_slot_mapping=False`**（`kv_cache_interface.py:917`）：ring 由层自己按 `pos % ring` 寻址，worker 不为它生成 slot mapping。

### 7.3 attention backend 分组

`init_attn_backend()`（`attn_utils.py:198`）：

```python
for kv_cache_group in kv_cache_config.kv_cache_groups:
    for layer_name in kv_cache_group.layer_names:
        attn_backend = attn_layers[layer_name].get_attn_backend()
        key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)
        group_map[key] = AttentionGroup(...)
```

同一个 KV group 内，可能因为 backend 不同或 Q 头数不同，拆成多个 `AttentionGroup`；每个 `AttentionGroup` 有自己的 metadata builder。

### 7.4 metadata

`AttentionGroup.create_metadata_builders()` 把 `kernel_block_size` 传给 builder；builder 在每步构造：

- `block_table`（per group）；
- `seq_lens` / `query_start_loc`；
- `slot_mapping`（如果 `uses_slot_mapping`）；
- 模型特定字段（MLA 的 `chunked_context`、sparse 的 `topk_indices`、QSA 的 index 等）。

---

## 8. 一张图总结

```
模型 config
  → 每层 get_kv_cache_spec()
  → dict[layer, spec]
  → get_kv_cache_groups()
  → [KVCacheGroupSpec(layers, merged_spec), ...]
  → get_kv_cache_config_from_groups()
  → KVCacheConfig{num_blocks, tensors[], groups[]}
  → allocate_kv_cache()
  → raw int8 buffer + as_strided 5D [L,B,H,N,C]
  → 每层 4D [B,H,N,C] view
  → bind_kv_cache()
  → layer.kv_cache
  → BlockTables + AttentionGroup metadata
  → kernel
```

---

## 9. 调试清单（实际跑模型时）

1. **打印 spec**：在 `GPUWorker.get_kv_cache_spec()` 返回前打印 `{name: type(spec).__name__, block_size, num_kv_heads, head_size, tokens_per_state, page_size_bytes}`。
2. **打印 group**：在 `get_kv_cache_groups()` 返回前打印每个 group 的 `layer_names` 和 `spec`。
3. **打印 config**：`num_blocks`、`bytes_per_block`、每个 `KVCacheTensor` 的 `layers/offset/layer_stride/block_stride`。
4. **打印 attention group**：`init_attn_backend()` 后打印 `(group_id, backend.get_name(), [layer_names], kernel_block_size)`。
5. **对照显存**：`num_blocks × bytes_per_block` 应接近 `available_kv_cache_memory_bytes`。
6. **对照日志**：vLLM 启动会打印 “KV cache size: ... tokens, Maximum concurrency ...”（`vllm/v1/core/kv_cache_utils.py:1074` 附近）。

---

## 10. 常见“看起来对但不对”的点

| 现象 | 真实原因 |
|---|---|
| `get_kv_cache_spec()` 返回 `None`，以为没 cache | 这层可能共享 source，或 cache 在另一个子模块（SWA/indexer） |
| `num_blocks` 比“总显存/page_size”小很多 | `bytes_per_block` 是 group 内所有层 page 之和，且取最大 group |
| 同一模型有多个 `block_size` | 不同 group 可以有不同 block size（DSV4 SWA=32，compressed 用 cache_config） |
| `KVCacheTensor.size` 都一样 | 所有 group 共享同一个 block id 空间，size = `bytes_per_block*num_blocks` |
| `slot_mapping` 有 -1 | padding / CUDA graph / ring buffer 不参与 |
| MLA 的 cache 是 `[B,1,N,C]` | `bind_kv_cache` squeeze 掉 head 维；kernel 把 1 个 head 当 MQA |

---

## 11. 一个具体例子：Qwen3MoE 30B，5 个 token

以 `qwen3_moe_inference_flow.md` 的输入为例：

```
序列 0："你好"   → [14, 26]       positions [0, 1]
序列 1："你是谁" → [14, 37, 52]   positions [0, 1, 2]

展平：tokens [14, 26, 14, 37, 52]
```

Qwen3MoE 30B 的 attention 配置：

```
48 层，每层都是 full attention（sliding_window=null）
hidden_size = 2048
32 个 Q heads，4 个 KV heads（GQA）
head_dim = 128
dtype = bfloat16
```

### 11.1 cache 里存的是什么

对每一层，`k_proj` / `v_proj` 把 5 个 token 的 hidden `[5, 2048]` 投影成：

```
K = [5, 512] → reshape [5, 4, 128]
V = [5, 512] → reshape [5, 4, 128]
```

K 还会做 k_norm + RoPE；V 不做 RoPE。cache 只存这两个结果：

- 不存 hidden state `[5, 2048]`；
- 不存 Q `[5, 32, 128]`；
- 不存 attention 输出 `[5, 2048]`；
- 不存 MoE 的专家输出；
- 不存权重矩阵本身。

每个 token、每个 KV head、每个 head_dim 的 K 和 V 都进 cache。32 个 Q heads 不单独存，它们共享 4 个 KV heads。

### 11.2 放到什么 tensor

所有 48 层的 `FullAttentionSpec` 完全相同，所以只有 **1 个 group**：

```python
KVCacheGroupSpec(
    layer_names=[48 层],
    kv_cache_spec=FullAttentionSpec(
        block_size=16,          # 假设默认 block_size=16
        num_kv_heads=4,
        head_size=128,
        head_size_v=128,
        dtype=bfloat16,
    ),
)
```

每层的逻辑 view 是：

```
[num_blocks, 4, 16, 256]   # bf16 元素
       B      H   N    C
```

- `B` 是 `num_blocks`，不是 batch=2；
- `H=4` 是 KV heads；
- `N=16` 是一个 block 的 token 槽位；
- `C=256` 个 bf16 元素 = 512 B，其中 K 128 个元素、V 128 个元素。

逻辑上 `[B,H,N,C]`，物理 stride 由 `KVCacheLayout` 决定；kernel 可以把最后维拆成 K view 和 V view，但底层是同一页。

每层一个 page 的字节数：

```
4 heads × 16 tokens × (128 + 128) × 2 B = 32 KiB
```

48 层共享一个 `KVCacheTensor` 描述符：

```
bytes_per_block = 48 × 32 KiB = 1.5 MiB
```

实际显存仍然只有一块 int8 backing；`KVCacheTensor` 只是描述 48 层的 offset / `layer_stride` / `block_stride`。

### 11.3 block table 和 slot

假设两个序列各分到 1 个 block（`block_size=16`）：

```
seq 0: block_table = [7]
seq 1: block_table = [8]
```

展平后的 5 个 token 对应：

```
slot = block_table[req][pos // 16] * 16 + (pos % 16)

token 14 (seq0, pos0) → 7*16+0 = 112
token 26 (seq0, pos1) → 7*16+1 = 113
token 14 (seq1, pos0) → 8*16+0 = 128
token 37 (seq1, pos1) → 8*16+1 = 129
token 52 (seq1, pos2) → 8*16+2 = 130
```

`slot_mapping = [112, 113, 128, 129, 130]`。

每个 slot 在每层 page 里对应一个 K/V 位置；48 层各写一份。5 个 token 的实际 K/V 数据量：

```
5 × 4 heads × 128 × 2 (K,V) × 2 B × 48 layers = 480 KiB
```

但物理上按 block 分配了 2 个 block：

```
2 × 1.5 MiB = 3 MiB
```

其中大部分 slot 暂时没有 token；这是 paged KV 的块粒度开销，不是 cache 存了 3 MiB 的有效 K/V。

### 11.4 对象对应关系

| 层 | 对象 |
|---|---|
| 模型层 | `Qwen3MoeAttention` 的 `k_proj` / `v_proj` |
| cache 声明 | 每层一个 `FullAttentionSpec` |
| 分组 | 48 层 merge 成 1 个 `KVCacheGroupSpec` |
| 显存布局 | 1 个 `KVCacheTensor`，`layers=48` |
| 运行时 page | `layer.kv_cache`，每层一个 `[num_blocks, 4, 16, 256]` view |
| 请求到 block | `BlockTable`，seq0 `[7]`，seq1 `[8]` |
| token 到 slot | `slot_mapping`，`[112, 113, 128, 129, 130]` |
| 物理 block | `KVCacheBlock` / `BlockPool` 里的 block 7、8 |
| 计算 | paged attention kernel 按 block table + slot 读每层 K/V |

注意：这是 **Qwen3MoE 30B**（48 层全 full attention，1 个 group）。不要把它和 Qwen3.8-Flash-Next 的 12 GQA + 36 GDN、5 个 group 混在一起。
