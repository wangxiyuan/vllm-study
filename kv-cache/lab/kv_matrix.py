"""Approximate KV/state matrix for the four target models.

Pure Python + the configs in ../configs. No vLLM install, no GPU.

    python3 kv_matrix.py

These are *estimates* from config fields, meant to build intuition and to
cross-check the real `KVCacheConfig`. They are not a substitute for vLLM's
own `page_size_bytes` / `num_blocks` computation (packing, padding, TP,
prefix sharing, and dtype overrides all move the real numbers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent / "configs"


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def gib(n: int | float) -> str:
    return f"{n / 2**30:.2f} GiB"


def mb(n: int | float) -> str:
    return f"{n / 2**20:.1f} MiB"


def load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text())


def ds41() -> None:
    c = load("deepseek_v41_flash.json")["text_config"]
    layers = c["num_hidden_layers"]
    ratios = c["compress_ratios"][:layers]
    swa_bytes = 584  # fp8_ds_mla packed record used by DeepseekV4SWACache
    state_bytes = 584
    ctx = 128 * 1024

    swa = layers * c["sliding_window"] * swa_bytes
    comp2 = sum(1 for r in ratios if r == 2)  # consumers; sources own the cache
    comp1 = sum(1 for r in ratios if r == 1)
    # Only the sources own the compressed cache: 3 x ratio2 + 1 x ratio1.
    comp = 3 * (ctx // 2) * state_bytes + 1 * ctx * state_bytes
    # Indexer: 128-dim, 8 source layers, roughly ratio 2/1; assume bf16.
    indexer = 8 * (ctx // 2) * 128  # uint8 packed
    total = swa + comp + indexer

    print("DeepSeek-V4.1-Flash")
    print(f"  layers={layers}, ratios={dict((r, ratios.count(r)) for r in set(ratios))}")
    print(f"  SWA (all layers, window 128): {mb(swa)} per request")
    print(f"  compressed (3x ratio2 + 1x ratio1): {mb(comp)} at 128k")
    print(f"  indexer (8 source layers, rough): {mb(indexer)} at 128k")
    print(f"  total rough: {mb(total)} per 128k request")
    print("  expected groups: SWA, comp-r2, comp-r1, idx-r2, idx-r1, compressor")
    print()


def kimi() -> None:
    c = load("kimi_k3.json")["text_config"]
    full = c["linear_attn_config"]["full_attn_layers"]
    kda = c["linear_attn_config"]["kda_layers"]
    heads = c["linear_attn_config"]["num_heads"]
    hd = c["linear_attn_config"]["head_dim"]
    ctx = 128 * 1024

    # MLA: kv_lora 512 + rope 64 = 576, one latent head, bf16.
    mla_per_tok = len(full) * 576 * 2
    mla_total = mla_per_tok * ctx

    # KDA state, per rank at TP=8. conv bf16, recurrent fp32.
    tp = 8
    conv_dim = 3 * heads * hd
    conv = (conv_dim // tp) * (4 - 1) * 2
    rec = (heads // tp) * hd * hd * 4
    state_per_layer = conv + rec
    state_total = state_per_layer * len(kda)

    print("Kimi-K3")
    print(f"  full MLA layers={len(full)}, KDA layers={len(kda)}")
    print(f"  MLA KV/token: {mla_per_tok / 1024:.1f} KiB; 128k: {gib(mla_total)}")
    print(f"  KDA state/layer/rank(TP8): {state_per_layer / 2**20:.2f} MiB; "
          f"all KDA: {state_total / 2**20:.1f} MiB")
    print("  expected groups: MLA, KDA")
    print()


def qwen() -> None:
    c = load("qwen3_8_flash_next.json")["text_config"]
    full = sum(1 for t in c["layer_types"] if t == "full_attention")
    linear = sum(1 for t in c["layer_types"] if t == "linear_attention")
    ctx = 262144

    # GQA: 2 KV heads x (K 256 + V 256) x bf16.
    kv_per_tok = full * c["num_key_value_heads"] * 2 * c["head_dim"] * 2
    kv_total = kv_per_tok * ctx
    qsa_per_tok = full * c["indexer_head_dim"] * 2 // c["indexer_compress_ratio"]
    qsa_total = qsa_per_tok * ctx

    # GDN state per rank at TP=8.
    tp = 8
    v_heads = c["linear_num_value_heads"]
    k_heads = c["linear_num_key_heads"]
    hd = c["linear_value_head_dim"]
    conv_dim = v_heads * hd + 2 * k_heads * hd
    conv = (conv_dim // tp) * (c["linear_conv_kernel_dim"] - 1) * 2
    rec = (v_heads // tp) * hd * hd * 4
    state_total = (conv + rec) * linear

    print("Qwen3.8-Flash-Next")
    print(f"  full={full}, linear={linear}")
    print(f"  GQA KV/token: {kv_per_tok / 1024:.1f} KiB; 262k: {gib(kv_total)}")
    print(f"  QSA compressed/token: {qsa_per_tok / 1024:.2f} KiB; 262k: {gib(qsa_total)}")
    print(f"  GDN state/rank(TP8): {state_total / 2**20:.1f} MiB")
    print("  expected groups: GQA, QSA-compressed, QSA-ring, GDN, PLE")
    print()


def glm() -> None:
    c = load("glm5_3_flash.json")["text_config"]
    full = sum(1 for t in c["layer_types"] if t == "deepseek_sparse_attention")
    linear = sum(1 for t in c["layer_types"] if t == "linear_attention")
    ctx = 128 * 1024

    # MLA: kv_lora 512, no rope, bf16.
    mla_per_tok = full * 512 * 2
    mla_total = mla_per_tok * ctx
    # indexer kpool: 128 dim every 4 tokens, bf16.
    idx_per_tok = full * 128 * 2 // c["index_kpool"]
    idx_total = idx_per_tok * ctx

    tp = 8
    heads = c["linear_attn_config"]["num_heads"]
    hd = c["linear_attn_config"]["head_dim"]
    conv_dim = 3 * heads * hd
    conv = (conv_dim // tp) * (4 - 1) * 2
    rec = (heads // tp) * hd * hd * 4
    state_total = (conv + rec) * linear

    print("GLM-5.3-Flash")
    print(f"  sparse MLA={full}, KDA={linear}")
    print(f"  MLA KV/token: {mla_per_tok / 1024:.1f} KiB; 128k: {gib(mla_total)}")
    print(f"  indexer/token: {idx_per_tok / 1024:.2f} KiB; 128k: {gib(idx_total)}")
    print(f"  KDA state/rank(TP8): {state_total / 2**20:.1f} MiB")
    print("  expected groups: sparse MLA, indexer kpool, indexer tail, KDA")
    print()


if __name__ == "__main__":
    ds41()
    kimi()
    qwen()
    glm()
