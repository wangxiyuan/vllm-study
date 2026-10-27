# 分组与 attention backend：谁和谁共享一张 block table

> 分组与后端解决两个问题：
> 1. 为什么同一个模型会有 2–6 个 KV cache group？
> 2. 每个 group 用哪个 attention backend、block table 怎么建？

## 0. 一句话

**group 的边界 = “是否共享同一张 block table”**。  
同 group 的层共享 block id 序列和分配/回收；不同 group 可以有不同 `block_size`、`page_size`、`max_num_blocks_per_req`、甚至不同 backend。

---

## 1. 分组算法：`get_kv_cache_groups()`

`vllm/v1/core/kv_cache_planning.py:1113`，决策顺序：

```
disable_hybrid_kv_cache_manager?
  └─ 是：把所有 spec 强行统一成 full attention 风格（只用于兼容）
attention-free?
  └─ 是：返回 []
hisparse?
  └─ 是：HiSparse 自己的分组
所有层 spec 可 merge?
  └─ 是：1 个 group（最常见：纯 full attention 模型）
所有层是同一“类型”?
  └─ 是：1 个 UniformTypeKVCacheSpecs group
否则：
  ├─ 尝试 packed groups（block-outermost）
  ├─ 尝试统一 page size，按 page size 分组
  ├─ 失败则 full-allocation fallback
  └─ hidden-state 层单独 group
```

### 1.1 什么叫“可 merge”

`KVCacheSpec.merge(specs)` 要求所有字段兼容。例如 `MLAAttentionSpec.merge` 要求：

- `block_size` 相同；
- `num_kv_heads`、`head_size`、`head_size_v` 相同；
- `dtype`、`tokens_per_state`、`kv_quant_mode` 相同；
- `cache_dtype_str`、`model_version`、`alignment` 相同。

DeepSeek V4.1 的 SWA 和 compressed 显然不满足（窗口 vs 全量、head 512 vs 512 但 tokens_per_state 不同），所以不同 group。

### 1.2 什么叫“同一类型，但 head size 不同”

`UniformTypeKVCacheSpecs.from_specs()`（`kv_cache_interface.py:1235`）允许：

- 同一个大类（都是 `FullAttentionSpec` / 都是 `SlidingWindowSpec` / 都是 `MambaSpec`）；
- `block_size`、`dcp_sharded` 相同；
- `is_uniform_with_collection()` 为真（同类型，token slot 数一致）。

这时所有层放一个 group，`page_size_bytes` 是各层 page 之和，`max_memory_usage_bytes` 取最大 page 数。

**例子**：纯 full attention 模型里，有些层 head_size 不同但都是 full attention；或者 Qwen 的 12 层 full attention 完全同构，直接 merge 成一个 group。

### 1.3 混合类型：按 page size 分组

`_get_kv_cache_groups_uniform_page_size()`（`:416`）的假设：

1. 每个 block 的物理字节在 group 间必须相同（否则碎片/分配复杂）；
2. `block_size` 当前用全局 `cache_config.block_size`；
3. 每 token 每层字节、每组层数尽量相同；
4. 同 group 内 attention 类型一致。

它先按 `layer_spec` 分桶，再尝试 merge；然后按 page size 拼成 group。  
**这就是为什么四个模型会拆成多个 group，而不是“一层一个 group”。**

---

## 2. 四个模型的预期 group

> 下面是按代码逻辑推导的“预期”；实际以 `KVCacheConfig.kv_cache_groups` 为准；可在启动时打印 groups 验证。

### 2.1 DeepSeek V4.1（约 6 组）

| # | 层 | spec | block_size | 说明 |
|---|---|---|---|---|
| 1 | 所有 40 层 `.swa_cache` | `SlidingWindowMLASpec` | 32 | window=128, head=512, fp8 packed |
| 2 | source 2/8/14 `.self_attn` | `MLAAttentionSpec` | `cache_config.block_size` | `tokens_per_state=2` |
| 3 | source 20 `.self_attn` | `MLAAttentionSpec` | 同上 | `tokens_per_state=1` |
| 4 | index source 2/8/14 | `MLAAttentionSpec` | 同上 | index head=128, ratio=2 |
| 5 | index source 20/24/28/32/36 | `MLAAttentionSpec` | 同上 | ratio=1 |
| 6 | source 2/8/14/20 `.compressor` | `CircularBufferSpec` | ring | 每请求 1 block |

注意：

- 层 3–7 等消费者**不在** compressed group 里；它们只读 source 的 cache。
- SWA group 的 block_size=32，其他 group 用 `cache_config.block_size`；这是少数允许 group 间 block_size 不同的场景。
- indexer 的 8 个 source 分成 ratio2/ratio1 两组，因为 `tokens_per_state` 不同。

### 2.2 Kimi K3（2 组）

| # | 层 | spec | 说明 |
|---|---|---|---|
| 1 | 24 层 full MLA | `MLAAttentionSpec` | `head=576`, `num_kv_heads=1`, bf16/fp8 |
| 2 | 69 层 KDA | `MambaSpec` | conv + recurrent，固定大小 |

为什么不是 93 组：24 层 MLA 的 spec 完全相同，可以 merge；69 层 KDA 的 shapes/dtypes 相同，也可以 merge。

### 2.3 Qwen3.8（约 5 组）

| # | 层 | spec | 说明 |
|---|---|---|---|
| 1 | 12 层 full attention | `FullAttentionSpec` | GQA 2 KV heads × 256 |
| 2 | 12 层 QSA compressed key | `MLAAttentionSpec` | `tokens_per_state=4`, head=128 |
| 3 | 12 层 QSA ring | `CircularBufferSpec` | 每请求 1 block |
| 4 | 36 层 GDN | `MambaSpec` | conv + recurrent |
| 5 | layer 2 PLE | `MambaSpec` | `tp_replicated=True`，短 conv |

QSA compressed 和 QSA ring 是 full attention 层的 side cache，不是主 KV；它们各自成为独立 group。

### 2.4 GLM-5.3（4 组）

| # | 层 | spec | 说明 |
|---|---|---|---|
| 1 | 11 层 sparse MLA | `MLAAttentionSpec` | head=512, 无 RoPE |
| 2 | 11 层 indexer kpool | `MLAAttentionSpec` | `tokens_per_state=4`, head=128 |
| 3 | 11 层 indexer tail | `CircularBufferSpec` | 2 heads × 128，bf16 ring |
| 4 | 34 层 KDA | `MambaSpec` | conv + recurrent |

---

## 3. 为什么 group 内要共享 block table

`KVCacheManager` 对每个 group 建一个 `SingleTypeKVCacheManager`（`vllm/v1/core/single_type_kv_cache_manager.py`）：

- 维护该 group 的 `block_table`；
- 负责 `allocate_slots` / `free` / 前缀命中；
- 每个 group 的 `block_size`、`max_num_blocks_per_req` 可以不同。

模型 runner 的 `BlockTables`（`vllm/v1/worker/gpu/block_table.py:17`）为每个 group 保存：

- `block_sizes[group]`；
- `max_num_blocks_per_group[group]`；
- `block_table_ptrs[group]`；
- `slot_mappings[group]`。

forward 时，同一 group 的所有层用同一张 block table；不同 group 的层用各自的表。

**好处**：

- 一个 group 的压缩/窗口策略可以独立演进；
- 不同 group 可以有不同的 block size（DSV4 SWA=32 vs compressed=64/128）；
- 线性注意力 state 不需要按 token 分配，单独 group 管理。

**代价**：

- 多 group 的前缀命中要协调（`HybridKVCacheCoordinator`，`kv_cache_coordinator.py:609`）；
- 每个 group 都要满足 `block_size % hash_block_size == 0`（不可前缀缓存的 ring 除外，见 `kv_cache_coordinator.py:650-666`）。

### 3.1 group 不是“每组单独分配一块显存”

`get_kv_cache_config_from_groups()` 先算：

```
bytes_per_block = max over groups( sum(page_size for layer in group) )
num_blocks      = available_kv_bytes // bytes_per_block
size            = bytes_per_block * num_blocks
```

然后为每个 group 生成一个 `KVCacheTensor`（`UniformTypeKVCacheSpecs` 内部如果 page 不同，会拆成多个 tensor）：

```python
KVCacheTensor(
    size=size,                 # 所有 tensor 的 size 相同
    layers=[...],              # 这个 group 里的层
    layer_stride=...,          # 相邻 layer 的 page 间隔
    block_stride=...,          # 相邻 block 的间隔
    offset=...,                # 在这块 backing 里的起点
)
```

真正分配显存只有一次（`allocate_kv_cache()`）：

```python
assert len({t.size for t in kv_cache_tensors}) == 1
buf = torch.zeros(size, dtype=torch.int8, device=device)
for tensor in kv_cache_tensors:
    views = create_kv_cache_views(buf, spec, ..., tensor)
    kv_caches.update(zip(tensor.layers, views))
```

所以准确的层级是：

| 对象 | 数量 | 说明 |
|---|---:|---|
| backing buffer | **1** | 一块 `int8` 显存，所有 group 共享 |
| `KVCacheTensor` | 每个 group 1 个（UniformType 可能多个） | 描述 offset / layer_stride / block_stride，不是独立显存 |
| layer view | 每个 spec 条目 1 个 | 每层拿到的 4D `[B,H,N,C]` 视图 |
| block table | 每个 group 1 张 | group 内所有层共用 |

`KVCacheTensor` 的注释写得很直接：地址范围重叠的 tensor **alias 同一段字节**；不同 group 的 tensor 从同一个 backing 的 byte 0 开始叠放，因为一个 block id 在任一时刻只属于一个 group。

### 3.2 每层一个 tensor 行不行？

分两层看：

- **view 层**：可以，而且现在就是每层一个 view（`layer.kv_cache` 是 4D tensor）。同 group 的层页大小相同，用 `layer_stride` 串起来即可。
- **物理分配层**：没必要。每层单独 `torch.zeros` 会变成几十次分配，带来对齐/碎片开销，而且 block id 仍然是按 group 分配的，单独分配并不会让寻址更简单。

真正不能做的是“每层一张 block table”。因为：

1. 一个 block id 要同时代表该 group 内所有层的 page；分开后同一个逻辑 token block 在每层会有不同 block id，scheduler/前缀缓存要维护 N 套映射。
2. 不同 spec 的 block 语义不同：Mamba state 不随 token 增长，ring 每请求只有 1 block，SWA 有窗口；它们无法和 paged KV 共用一张 block table。
3. attention metadata 按 group 构建；kernel 一次拿一张 group block table，不会为每层各拿一张。

所以 group 的本质不是“给每层分配 tensor”，而是**把可以共享同一张 block table 的层放在一起，作为一个分配/回收单元**。tensor 只是这个单元在显存里的布局描述。

---

## 4. attention backend 怎么选

### 4.1 通用流程

`init_attn_backend()`（`vllm/v1/worker/gpu/attn_utils.py:198`）：

```
for kv_cache_group in config.kv_cache_groups:
    for layer in group.layer_names:
        backend = layer.get_attn_backend()
        spec = group.spec or group.spec.kv_cache_specs[layer]
        key = (backend.full_cls_name(), spec, layer.num_heads)
        AttentionGroup(backend, [layer], spec, group_id)
```

同一 group 内如果 backend 不同（例如 MLA 和 indexer 在同一个 group？通常不会），或 Q 头数不同，会拆成多个 `AttentionGroup`。

然后：

1. `prepare_kernel_block_sizes()` 为每个 group 选一个所有 backend 都支持的 kernel block size；
2. 每个 `AttentionGroup.create_metadata_builders()` 建 metadata builder；
3. builder 在运行时生成 block table / seq_lens / slot_mapping。

### 4.2 四模型 backend 映射

| 模型 / 层 | 后端类 | 选择位置 |
|---|---|---|
| DSV4.1 sparse MLA | `DeepseekV4FlashMLAAttention` / `DeepseekV4FlashInferMLAAttention` / `DeepseekV4FlashInferSM120Attention` / `DeepseekV4MegaAttnAttention` | `vllm/models/deepseek_v41/nvidia/model.py:198` |
| DSV4.1 SWA | `DeepseekSparseSWABackend`（FlashMLA/FlashInfer 子类） | `sparse_swa.py:155` |
| DSV4.1 indexer | `DeepseekV4IndexerBackend`（或 sparse logits 变体） | `deepseek_v41/attention.py` |
| Kimi MLA | `get_attn_backend(use_mla=True)` → FlashMLA / FlashInfer / Cutlass MLA | `vllm/models/kimi_k3/nvidia/mla.py:320` |
| Kimi KDA | `KimiK3KDAAttentionBackend`（FlashKDA / Triton） | `vllm/models/kimi_k3/nvidia/kda.py:547` |
| Qwen full attention | `Qwen4ExpQSAFlashAttentionBackend`（继承 FlashAttentionBackend） | `vllm/models/qwen4_exp/nvidia/qsa.py:146` |
| Qwen QSA side | `QSAStateBackend`（out-of-band，无 attention impl） | `vllm/models/qwen4_exp/common/qsa_cache.py:727` |
| Qwen GDN | GDN Mamba backend | `qwen_gdn_linear_attn.py` |
| GLM sparse MLA | 通用 MLA backend（FlashMLA / FlashInfer sparse） | `vllm/models/glm5next/common/attention.py:534` |
| GLM indexer | `Glm5NextIndexerBackend` | `vllm/models/glm5next/common/attention.py:121` |
| GLM tail | `KpoolTailBackend` | `vllm/models/glm5next/common/attention.py:181` |
| GLM KDA | FlashKDA / Triton | `glm5next/common/kda.py` |

### 4.3 kernel block size

后端通过 `get_supported_kernel_block_sizes(spec)` 声明：

- 返回 `int`：必须是精确值；
- 返回 `MultipleOf(n)`：接受 `n` 的倍数。

`select_common_block_size()`（`vllm/v1/worker/utils.py` 内）选一个所有 backend 都支持、且能整除 manager block size 的最大值。

例子：

- DSV4.1 SWA backend：`[MultipleOf(32)]`，preferred 256；
- GLM indexer：32/64（SM120 只支持 64）；
- 普通 FlashAttention：通常 128/256。

### 4.4 为什么 `kernel_block_size` 会小于 `block_size`

manager 用 128 token 一个 block，但某个 kernel 只支持 64；`create_kv_cache_views` 把视图第 0 维拆成 `num_blocks*2`，`map_kv_caches_to_kernel_blocks()` 再按 manager block 分组。  
**这就是你在 tensor shape 里看到 `B*2` 的原因。**

---

## 5. block table 与 slot mapping

### 5.0 block、page、slot、block table 到底分别是什么

先把四个词分开：

| 概念 | 是什么 | 不是什么 |
|---|---|---|
| block | 一条序列的 **16 个 token 位置**（`block_size=16`）的分配单位 | 不是 16 个 hidden state，也不是 16 个 token 的全局集合 |
| page | **某一层**在一个 block 里的 K/V 存储区 | 不是整层的 cache，也不是整个模型的 cache |
| block id | 物理 block 的编号，例如 7、8 | 不是 token id，也不是层号 |
| block table | 每条请求的“逻辑 block 序号 → 物理 block id”映射 | 不是 K/V 数据本身 |
| slot | 一个 token 在物理 page 里的具体位置 | 不是 token 在序列里的位置；它是物理地址的一部分 |

`block_size=16` 的准确含义是：

> 一条序列最多把 16 个连续 token 放进一个 block；需要第 17 个 token 时，再分配下一个 block。

#### block id 是每层 view 的第一维下标

每层拿到的 cache view 是：

```text
layer.kv_cache: [num_blocks, H, N, C]
                       ^
                     block id
```

`block id = 7` 就是第 7 号物理 block 在这个 view 第一维的下标：

```python
layer0.kv_cache[7]   # layer 0 的 block 7 对应的 page
layer1.kv_cache[7]   # layer 1 的 block 7 对应的 page
...
layer47.kv_cache[7]  # layer 47 的 block 7 对应的 page
```

它们是**不同的物理 page**，只是共用同一个 block id。物理地址由 `layer_stride` 和 `block_stride` 算出来。

block id 由 `BlockPool` 分配，是一个 0..num_blocks-1 的整数句柄，不是字节地址，也不是 token id。

`block_table = [7]` 的意思是：

```text
这条请求的逻辑 block 0  →  物理 block 7
```

它只描述“请求的逻辑块到物理块的映射”，不包含 token，也不包含 K/V。真实 buffer 里每行有 `max_num_blocks_per_req` 个位置；只有前面的有效项有值，后面不用。

例如 Qwen3MoE 30B，48 层、4 个 KV heads、head_dim=128、bf16：

```text
一个 block id = 7
  layer 0 的 page 7：16 个 token 槽位 × 4 heads × 128 维 K/V
  layer 1 的 page 7：16 个 token 槽位 × 4 heads × 128 维 K/V
  ...
  layer 47 的 page 7：16 个 token 槽位 × 4 heads × 128 维 K/V
```

同一个 block id 7 在每一层都有一页，但页的物理地址不同：

```text
layer l, block 7 的起始字节 =
    tensor.offset + l * layer_stride + 7 * block_stride
```

所以“block 7”不是一块连续的 1.5 MiB；它是 **48 个 page 的逻辑集合**，每层一页。`KVCacheTensor` 的 `layer_stride` / `block_stride` 就是用来算这些地址的。

#### 一个 block 里到底存了多少 token

- 容量：最多 16 个 token；
- 实际：如果序列只有 2 个 token，这个 block 只写了 2 个 slot，另外 14 个 slot 空着；
- 空 slot 仍然占着物理 page 的空间，这是 block 粒度的内部碎片。

上一个例子里：

```text
seq0 "你好"   → block 7，只用了 slot 0、1
seq1 "你是谁" → block 8，只用了 slot 0、1、2
```

两个序列各自有自己的 block table：

```text
seq0: [7]
seq1: [8]
```

如果 seq1 继续生成到 20 个 token：

```text
token 0..15  → block 8  （逻辑 block 0）
token 16..19 → block 9  （逻辑 block 1，新分配）

seq1 的 block_table = [8, 9]
```

第 16 个 token 的 slot：

```text
block_table[1][16 // 16] = 9
slot = 9 * 16 + (16 % 16) = 144
```

#### block table 的 tensor 形状

`BlockTables` 为每个 group 维护：

```text
block_table: [max_num_reqs, max_num_blocks_per_req]  # int32
```

每一行是一条请求；行里的第 `k` 个元素是逻辑 block `k` 对应的物理 block id。序列长度不够时，后面的槽位不用。

#### block 的元数据在 BlockPool 里

`KVCacheBlock` 只存管理信息：

```text
block_id、ref_cnt、block_hash、is_null ...
```

K/V 数据不在 `KVCacheBlock` 里，而在 `KVCacheTensor` 的 page 里。`BlockPool` 负责分配/回收 block id，并维护引用计数。

#### 前缀缓存时 block 会共享

如果两条请求的前缀完全相同，某个**完整 block** 的 hash 相同，`BlockPool` 会让两条请求的 block table 指向同一个物理 block id，并增加 `ref_cnt`：

```text
seq0 block_table = [7, 9]
seq1 block_table = [7, 11]   # 第一个 block 共享 7
```

共享的是 block id 和它对应的所有层 page；不是把 K/V 复制一份。部分填充的 block 不能共享，因为 hash 只对完整 block 稳定。

#### 5 层玩具模型的一次 forward

为了看清结构，把模型缩成 5 层、2 个 KV heads、head_dim=2、bf16、block_size=16。输入：

```text
"你好" → tokens [14, 26]，positions [0, 1]
```

**第 0 步：分配 block**

`BlockPool` 从空闲块里分配一个物理 block，假设是 7：

```text
KVCacheBlock(block_id=7, ref_cnt=1)

block_table[req] = [7]      # 逻辑 block 0 → 物理 block 7
slot_mapping     = [112, 113]  # 7*16+0, 7*16+1
```

这 5 层属于同一个 group，所以只有一张 block table；5 层都用 block id 7。

**第 1 步：layer 0 forward**

```text
H0 = [2, hidden]
K0 = H0 @ Wk0 → [2, 2, 2]    # token, KV head, head_dim
V0 = H0 @ Wv0 → [2, 2, 2]
Q0 = H0 @ Wq0 → [2, Q head, 2]   # 不存
```

写入 layer 0 的 page：

```text
layer0.kv_cache[7]   # 逻辑形状 [2 heads, 16 slots, 4]（K 2 + V 2）

slot 0: K0[0], V0[0]   # token 14
slot 1: K0[1], V0[1]   # token 26
slot 2..15: 空
```

layer 0 的 attention 读自己的 page 7：token 26 看 slot 0/1，token 14 看 slot 0。

**第 2 步：layer 1 forward**

layer 0 的输出进入 layer 1：

```text
K1 = H1 @ Wk1 → [2, 2, 2]
V1 = H1 @ Wv1 → [2, 2, 2]
```

写入 **layer 1 自己的 page 7**：

```text
layer1.kv_cache[7]

slot 0: K1[0], V1[0]
slot 1: K1[1], V1[1]
```

layer 0 的 page 7 不会被覆盖；layer 1 的 page 7 是另一段地址。

**第 3 步：layer 2..4**

重复同样的事。每层都写自己的 page 7，slot 0/1 分别对应 token 14/26。

**第 4 步：forward 结束后**

```text
block_table = [7]
slot_mapping = [112, 113]

block 7:
  layer0 page7: slot0=K/V(14), slot1=K/V(26)
  layer1 page7: slot0=K/V(14), slot1=K/V(26)
  layer2 page7: slot0=K/V(14), slot1=K/V(26)
  layer3 page7: slot0=K/V(14), slot1=K/V(26)
  layer4 page7: slot0=K/V(14), slot1=K/V(26)
```

所以：

- `block_table=[7]` 只有 **1 个 entry**，表示逻辑块 0 → 物理块 7；
- block 7 本身不是“一个 page”，而是 **5 个 layer page 的逻辑集合**；
- 每层 page 7 的物理地址不同，但都用同一个 block id 寻址；
- 2 个 token 只用了 slot 0、1，slot 2..15 空着。

**第 5 步：再 decode 一个 token**

新 token 在 pos 2，仍然在同一个 block：

```text
slot = 7*16 + 2 = 114
block_table 仍然是 [7]
```

每层在 page 7 的 slot 2 追加自己的 K/V。

如果继续到第 17 个 token（pos 16），需要新 block 8：

```text
block_table = [7, 8]
slot(16) = 8*16 + 0 = 128
```

这时 5 层各有 page 7 和 page 8 两页。

#### 为什么逻辑 block 是 0

“逻辑 block”是**序列内部的编号**，不是物理编号：

```text
logical_block = position // block_size
```

所以：

```text
pos 0..15  → logical block 0
pos 16..31 → logical block 1
```

任何请求都从 `position=0` 开始，因此任何请求都有自己的逻辑 block 0。它是 block table 的行内下标：

```text
block_table[req][0] = 这条请求第 1 个逻辑块用的物理 block id
```

两条不同的请求可以都有逻辑 block 0：

```text
seq0 logical block 0 → physical block 7
seq1 logical block 0 → physical block 8
```

逻辑编号相同，物理编号不同，这正是 block table 存在的意义：把稳定的“序列内第几块”映射到实际分配的“物理第几块”。

#### “5 层共用一张 block table”怎么理解

共用的只是**索引表**，不是 K/V 数据。

```text
                 block_table[req]
                       [7]
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
 layer0.kv_cache[7]  layer1.kv_cache[7] ... layer4.kv_cache[7]
       page A             page B             page E
```

- block table 里只有整数 7，没有 K/V；
- 5 层都读同一个 entry 7；
- 每层用自己的 view 取第 7 行，得到自己的 page；
- page A/B/E 是不同物理地址，K/V 内容也不共享。

所以“共用 block table”= 共用同一套“逻辑块 → 物理块”的编号；不是共用同一页，也不是共用 K/V。

#### 一个 block 多大，什么时候分配

**大小**：一个 block id 的大小不是一个 page，而是 group 内所有 layer 的 page 之和：

```text
page_size(layer) = num_kv_heads × block_size × (head_size + head_size_v)
                   × dtype_bytes / tokens_per_state

bytes_per_block  = sum(page_size(layer) for layer in group)
```

5 层玩具模型（`H=2, D=2, block=16, bf16`）：

```text
page_size = 2 × 16 × (2 + 2) × 2 = 256 B
bytes_per_block = 5 × 256 = 1280 B
```

真实 Qwen3MoE 30B（48 层，`H=4, D=128, block=16, bf16`）：

```text
page_size = 4 × 16 × (128 + 128) × 2 = 32 KiB
bytes_per_block = 48 × 32 KiB = 1.5 MiB
```

**什么时候分配**：分两个阶段，不要混：

1. **启动时分配物理显存**  
   `get_kv_cache_config_from_groups()` 用可用显存算 `num_blocks` 和 `size = bytes_per_block × num_blocks`；`allocate_kv_cache()` 一次性 `torch.zeros(size, int8)` 分配整块 backing。所有 block 的物理空间在这一刻就有了，`BlockPool` 把它们加入空闲链表。

2. **请求到来时分配 block id**  
   prefill 调度时，`allocate_slots()` 按 `ceil(tokens / block_size)` 计算需要几个逻辑 block，从 `BlockPool` 取空闲 block id，写进该请求的 `block_table`，并把 `ref_cnt` 加 1。这时才决定“逻辑 block 0 → 物理 block 7”。

所以“你好”两个 token：

```text
启动时：物理显存里已经有 num_blocks 个 block 的空间
请求时：BlockPool 分配 block id 7
prefill：K/V 写进 block 7 的每层 page
```

block 7 的物理空间不是请求时临时 `malloc` 的；请求时只是把一个已有 block id 映射给这条请求。decode 超过 16 个 token 时，再分配下一个 block id（例如 8），追加到 block table。

#### 写 page 的真实 tensor 操作

`layer0 page7 slot0 ← K/V(14)` 不是 PyTorch 的 `tensor[slot] = kv`，而是调用 CUDA 的 scatter kernel：

```python
# vllm/v1/attention/backends/flash_attn.py
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    # kv_cache: [num_blocks, H, N, 2*D]
    # transpose -> [num_blocks, N, H, 2*D]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(
        self.head_size, dim=-1
    )
    reshape_and_cache_flash(
        key, value,
        key_cache, value_cache,
        slot_mapping,
        self.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )
```

`key_cache` / `value_cache` 是 `[num_blocks, block_size, H, D]` 的 view。`reshape_and_cache_flash` 的 CUDA kernel 按 token 做：

```cpp
slot_idx = slot_mapping[token_idx];
block_idx = slot_idx / block_size;
block_offset = slot_idx % block_size;

key_dst = key_cache
        + block_idx * block_stride
        + block_offset * page_stride;

copy(key[token_idx], key_dst, num_heads * head_size);
copy(value[token_idx], value_dst, num_heads * head_size);
```

对 token 14（slot 112，block_size=16）：

```text
block_idx = 112 / 16 = 7
block_offset = 112 % 16 = 0

key_cache[7, 0, :, :] = key[0]
value_cache[7, 0, :, :] = value[0]
```

对 token 26（slot 113）：

```text
key_cache[7, 1, :, :] = key[1]
```

- 这是真正的显存写；目标地址已经在启动时分配好；
- `slot_mapping` 是 `[num_tokens]`，kernel 忽略 `-1` 的 padding token；
- 5 层各调用一次同样的 kernel，只是 `kv_cache` view 不同；
- Qwen3 在写之前已经做完 `k_norm` 和 RoPE，所以写进去的是投影 + norm + RoPE 后的 K；V 是投影后的 V。

#### 多 group 的玩具例子

模型只有 3 层，但有两组：

```text
layer 0, 1：full attention
  H = 2, D = 2, block_size = 16, bf16
  page_size = 2 × 16 × (2+2) × 2 = 256 B

layer 2：sliding-window attention
  H = 1, D = 2, block_size = 8, bf16
  page_size = 1 × 8 × (2+2) × 2 = 64 B
```

分组：

```text
G0 = [layer0, layer1]   FullAttentionSpec, block_size 16
G1 = [layer2]           SlidingWindowMLASpec, block_size 8
```

`bytes_per_block = max(2×256, 64) = 512 B`。假设 `num_blocks = 4`，backing 一共 2048 B。

输入仍然是：

```text
"你好" → tokens [14, 26]，positions [0, 1]
```

**分配两个物理 block**（同一个 BlockPool，id 全局唯一）：

```text
G0 拿到 block 2
G1 拿到 block 1
```

两张 block table：

```text
G0 block_table[req] = [2]   # 逻辑 block 0 → 物理 2
G1 block_table[req] = [1]   # 逻辑 block 0 → 物理 1
```

两张 slot mapping：

```text
G0: slot(0) = 2*16 + 0 = 32
    slot(1) = 2*16 + 1 = 33
    slot_mapping = [32, 33]

G1: slot(0) = 1*8 + 0 = 8
    slot(1) = 1*8 + 1 = 9
    slot_mapping = [8, 9]
```

写入：

```text
layer0 page2 slot0/1 ← K/V(14), K/V(26)   # G0
layer1 page2 slot0/1 ← K/V(14), K/V(26)   # G0
layer2 page1 slot0/1 ← K/V(14), K/V(26)   # G1
```

注意：

- **同一个请求有两张 block table**，不是一张；
- layer0/1 用 G0 的表，layer2 用 G1 的表；
- 同一个 token 14 在两个 group 的 slot 不同：G0 是 32，G1 是 8；
- 因为 G0 的 block_size=16、G1 的 block_size=8，所以“逻辑 block 0”覆盖的 token 范围也不同。

如果序列有 10 个 token：

```text
G0: token 0..15 都在逻辑 block 0 → block_table [2]，只需 1 个 block
G1: token 0..7  在逻辑 block 0 → 物理 1
    token 8..15 在逻辑 block 1 → 物理 3

G1 block_table = [1, 3]
```

同一个 pos=8 的 token：

```text
G0 slot = 2*16 + 8 = 40
G1 slot = 3*8 + 0 = 24
```

所以 slot 必须按“group 自己的 block_size”算，不能拿一个全局公式套所有 group。

#### 这个 max 会浪费空间吗

会。上面的例子里：

```text
G0 block = 512 B
G1 block = 64 B
bytes_per_block = 512 B
```

当 G1 拿到一个 block 时，它只用 64 B，剩下 448 B 在**这个 block 槽位内部**空着。这是 block 级内部碎片，确实浪费。

为什么不能让 G1 的 block 只占 64 B？因为 block id 是全局共享的：

```text
block 1 今天可能属于 G1，明天释放后可能分配给 G0
```

如果 G1 的 block 1 只从 byte 64 开始，而 G0 的 block 0 占 0..512，两者会重叠。为了保证**任意 block id 在任意时刻都能被任意 group 使用**，每个 block 槽位都必须按最大的 group 留足空间：

```text
block slot 大小 = max over groups(bytes_per_block)
```

这是用“少量内部碎片”换：

- 一套全局 block id；
- 一套共享 BlockPool；
- 简单的按 block 寻址；
- 不需要为每个 group 维护独立的地址空间。

代价是：如果某个 group 的 page 远小于最大 group，它持有的 block 尾部会浪费。实际模型的 group page 差距通常没有玩具例子这么夸张；但如果一个小 page group 占满所有 block，浪费就会很明显。这是统一 block id 空间的设计取舍，不是没有代价。

---
 
### 5.1 block table 是 per-group 的

`BlockTables` 构造（`vllm/v1/worker/gpu/model_runner.py:709`）时传入：

```python
block_sizes=[group.spec.block_size ...]
max_num_blocks_per_group=[...]
slot_mapping_enabled=[group.spec.uses_slot_mapping ...]
dcp_sharded=[...]
```

每个 group 一行 `max_num_blocks_per_group`，存物理 block id。

### 5.2 slot mapping 是 per-token 的

`compute_slot_mappings()`（`block_table.py:194`）为**每个可写 token** 算：

```
slot = block_table[req, pos // B] * B + (pos % B)
```

`CircularBufferSpec.uses_slot_mapping=False`，所以 ring 不生成 slot；层自己按 `pos % ring` 寻址。

### 5.3 group 与 layer 的映射

`build_slot_mappings_by_layer()`（`attn_utils.py:448`）把 group 的 slot mapping 分发给 group 内每个 layer：

```python
for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
    for layer_name in kv_cache_group.layer_names:
        slot_mappings_by_layer[layer_name] = slot_mapping
```

forward 时 `forward_context.slot_mapping[layer_name]` 就是它。

---

## 6. 四个模型的“分组 + backend”速记

```
DSV4.1:
  SWA(40) ──────────────── FlashMLA sparse SWA, block 32
  compressed ratio2(3) ─── FlashMLA/FlashInfer sparse MLA
  compressed ratio1(1) ─── 同上
  indexer ratio2(3) ────── DeepSeekV4IndexerBackend
  indexer ratio1(5) ────── 同上
  compressor state(4) ──── CircularBuffer, 无 attention

Kimi K3:
  MLA(24) ──────────────── FlashMLA / FlashInfer / Cutlass MLA
  KDA(69) ──────────────── KimiK3KDAAttentionBackend

Qwen3.8:
  GQA(12) ──────────────── Qwen4ExpQSAFlashAttentionBackend
  QSA compressed(12) ───── QSAStateBackend（side cache）
  QSA ring(12) ─────────── QSAStateBackend
  GDN(36) ──────────────── GDN backend
  PLE(1) ───────────────── Mamba/ShortConv backend

GLM-5.3:
  sparse MLA(11) ───────── FlashMLA / FlashInfer sparse
  indexer kpool(11) ────── Glm5NextIndexerBackend
  indexer tail(11) ─────── KpoolTailBackend
  KDA(34) ──────────────── FlashKDA / Triton
```

---

## 7. 易错点

1. **group 不是按“层类型”简单分的**，而是按 spec merge / page size；同是 MLA，ratio 不同就会分 group。
2. **`UniformTypeKVCacheSpecs` 不是“所有层一样”**；它允许 head size 不同，只要 token slot 数一致。
3. **同一 group 内不同 backend** 会被拆成多个 `AttentionGroup`，但仍是同一 KV group（同一 block table）。
4. **`block_size` 可以 group 间不同**；不要假设全局只有一个。
5. **`CircularBufferSpec` 不参与前缀缓存**，也不生成 slot mapping；它每请求固定 1 block。
6. **`MambaSpec` 的 `block_size` 不是 token block**；它的 page 是 state，`max_memory_usage_bytes` 与 `max_model_len` 无关。
