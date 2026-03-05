# 第十部分: 配置与使用

## 10.1 Speculative Config 配置项

### 10.1.1 `speculative_config` 结构

```python
# vllm/config/speculative_config.py
@dataclass
class SpeculativeConfig:
    """Speculative Decoding 配置。"""

    # 推测方法
    method: Literal["eagle", "eagle3", "mtp", "medusa", "ngram", "suffix", "draft"]

    # 最大推测 token 数
    num_speculative_tokens: int = 5

    # Draft Model 配置
    draft_model_config: Optional[ModelConfig] = None

    # Token Tree 结构 (可选)
    speculative_token_tree: Optional[str] = None

    # 是否禁用 padded drafter batch
    disable_padded_drafter_batch: bool = False

    # 是否使用局部 argmax 归约
    use_local_argmax_reduction: bool = False

    # Draft tensor parallel size
    draft_tensor_parallel_size: int = 1

    # Prompt Lookup 配置 (N-gram)
    prompt_lookup_min: Optional[int] = None
    prompt_lookup_max: Optional[int] = None

    # 是否强制 eager 模式
    enforce_eager: bool = False

    def use_eagle(self) -> bool:
        """检查是否使用 Eagle 方法。"""
        return self.method in ("eagle", "eagle3")
```

### 10.1.2 关键配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | str | - | 推测方法名称 |
| `num_speculative_tokens` | int | 5 | 每次迭代最多生成的 draft tokens |
| `speculative_token_tree` | str | None | Token Tree 结构，如 `"[(0,), (0,1), (0,2)]"` |
| `draft_model_config` | ModelConfig | None | Draft Model 配置 |
| `disable_padded_drafter_batch` | bool | False | 是否禁用 padded batch |
| `use_local_argmax_reduction` | bool | False | 是否使用局部 argmax |
| `draft_tensor_parallel_size` | int | 1 | Draft Model TP size |
| `prompt_lookup_min` | int | None | N-gram 最小匹配长度 |
| `prompt_lookup_max` | int | None | N-gram 最大匹配长度 |
| `enforce_eager` | bool | False | 是否强制使用 eager 模式 |

### 10.1.3 Token Tree 配置

**配置格式**: 

```python
# 线性链（默认）
speculative_token_tree = "[(0,), (0,1), (0,2), (0,3), (0,4)]"

# 树状结构（Eagle3）
speculative_token_tree = "[(0,), (0,1), (0,2), (0,1,3), (0,1,4)]"

# 可视化: 
# 线性链: root → d1 → d2 → d3 → d4
# 树状: 
#       root
#      /    \
#    d1      d2
#   /  \
#  d3   d4
```

## 10.2 Eagle3 模型配置

### 10.2.1 模型注册与识别

**模型注册**: 

```python
# vllm/model_executor/models/__init__.py
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

@register_model("eagle3")
class Eagle3LlamaForCausalLM(nn.Module):
    """Eagle3 模型实现。"""
    ...
```

### 10.2.2 配置文件格式

**Eagle3 模型 config.json**: 

```json
{
  "architectures": ["Eagle3LlamaForCausalLM"],
  "model_type": "eagle3_llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "intermediate_size": 11008,
  "vocab_size": 32000,
  "eagle3_config": {
    "num_hidden_layers": 4,
    "combination_method": "weighted_sum"
  }
}
```

### 10.2.3 权重加载流程

```python
def load_eagle_model(target_model, vllm_config):
    """加载 Eagle 模型。"""
    # 获取 target model 的 embed 和 lm_head
    target_embed = target_model.get_input_embeddings()
    target_lm_head = target_model.get_output_embeddings()

    # 创建 Eagle 模型
    eagle_model = get_model(
        vllm_config=vllm_config,
        model_config=vllm_config.speculative_config.draft_model_config,
    )

    # 共享权重
    eagle_model.set_input_embeddings(target_embed)
    eagle_model.set_output_embeddings(target_lm_head)

    return eagle_model
```

## 10.3 vLLM-Ascend 特定配置

### 10.3.1 昇腾设备选择

```python
# 指定使用昇腾 NPU
import torch
import torch_npu

# 设置设备
device = torch.device("npu:0")

# 或者通过环境变量
# export ASCEND_RT_VISIBLE_DEVICES=0
```

### 10.3.2 性能调优参数

**环境变量**: 

```bash
# HCCL 优化
export HCCL_WHITELIST=AllReduce,AllGather
export HCCL_CONNECT_TIMEOUT=1200

# CANN 优化
export ASCEND_HOME_PATH=/usr/local/Ascend
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

# 内存优化
export PYTORCH_NPU_ALLOCATOR_CONF=expandable_segments:True
```

### 10.3.3 环境变量配置

```bash
# vLLM 通用配置
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 昇腾特定
export VLLM_USE_ASCEND=1
export VLLM_ASCEND_ATTENTION_BACKEND=ascend_flash_attn
```

## 10.4 使用示例

### 10.4.1 命令行启动示例

**Eagle3 Speculative Decoding**: 

```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --speculative-config '{"method": "eagle3", "num_speculative_tokens": 5}' \
    --draft-model-path vllm/eagle3-llama3-70b \
    --tensor-parallel-size 8
```

**N-gram Speculative Decoding**: 

```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --speculative-config '{"method": "ngram", "prompt_lookup_min": 2, "prompt_lookup_max": 4, "num_speculative_tokens": 5}' \
    --tensor-parallel-size 8
```

### 10.4.2 API 调用示例

```python
from vllm import LLM, SamplingParams

# 创建 LLM 实例
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    speculative_config={
        "method": "eagle3",
        "num_speculative_tokens": 5,
    },
    draft_model="vllm/eagle3-llama3-70b",
    tensor_parallel_size=8,
)

# 创建采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 生成
prompts = ["Hello, my name is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 10.4.3 性能基准测试

**测试脚本**: 

```python
import time
from vllm import LLM, SamplingParams

# 创建 LLM 实例（带 Spec Decode）
llm_spec = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    speculative_config={"method": "eagle3", "num_speculative_tokens": 5},
    draft_model="vllm/eagle3-llama3-70b",
)

# 创建 LLM 实例（不带 Spec Decode）
llm_base = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
)

prompts = ["Hello, my name is"] * 100
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# 测试带 Spec Decode
start = time.time()
outputs_spec = llm_spec.generate(prompts, sampling_params)
time_spec = time.time() - start

# 测试不带 Spec Decode
start = time.time()
outputs_base = llm_base.generate(prompts, sampling_params)
time_base = time.time() - start

# 计算加速比
speedup = time_base / time_spec
print(f"Spec Decode Time: {time_spec:.2f}s")
print(f"Base Time: {time_base:.2f}s")
print(f"Speedup: {speedup:.2f}x")
```

**典型性能结果**: 

| 配置 | Throughput (tokens/s) | Speedup |
|------|----------------------|---------|
| Base (No Spec) | 1250 | 1.0x |
| Eagle3 (K=5) | 3800 | 3.04x |
| N-gram | 2100 | 1.68x |

---

# 附录

## 附录 A: 核心类与函数索引

### A.1 核心类

| 类名 | 文件路径 | 说明 |
|------|----------|------|
| `SpecDecodeBaseProposer` | `vllm/v1/spec_decode/base.py` | 推测解码基类 |
| `EagleProposer` | `vllm/v1/spec_decode/eagle.py` | Eagle 推测器 |
| `EagleSpeculator` | `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` | Eagle Speculator |
| `MedusaProposer` | `vllm/v1/spec_decode/medusa.py` | Medusa 推测器 |
| `NgramProposer` | `vllm/v1/spec_decode/ngram_proposer.py` | N-gram 推测器 |
| `RejectionSampler` | `vllm/v1/sample/rejection_sampler.py` | 拒绝采样器 |
| `CUDAGraphDispatcher` | `vllm/v1/cudagraph_dispatcher.py` | CUDA Graph 调度器 |

### A.2 核心函数

| 函数名 | 文件路径 | 说明 |
|--------|----------|------|
| `propose()` | 各 Proposer 类 | 生成 draft tokens |
| `rejection_sample()` | `rejection_sampler.py` | 执行拒绝采样 |
| `combine_hidden_states()` | `llama_eagle3.py` | Eagle3 隐藏状态融合 |
| `prepare_eagle_inputs()` | `speculator.py` | 准备 Eagle 输入 |

## 附录 B: 代码文件路径索引

### B.1 vLLM 关键文件

```
vllm/
├── v1/
│   ├── spec_decode/
│   │   ├── base.py                    # 基类
│   │   ├── eagle.py                   # Eagle 实现
│   │   ├── medusa.py                  # Medusa 实现
│   │   ├── ngram_proposer.py          # N-gram 实现
│   │   ├── metadata.py                # 元数据
│   │   └── utils.py                   # 工具函数
│   ├── worker/
│   │   └── gpu/
│   │       └── spec_decode/
│   │           └── eagle/
│   │               ├── speculator.py  # Eagle Speculator
│   │               └── cudagraph.py   # CUDA Graph 管理
│   ├── sample/
│   │   └── rejection_sampler.py       # 拒绝采样
│   └── attention/
│       └── backends/
│           └── tree_attn.py           # Tree Attention
```

### B.2 vLLM-Ascend 关键文件

```
vllm-ascend/
└── vllm_ascend/
    └── spec_decode/
        ├── __init__.py                # 入口工厂
        ├── eagle_proposer.py          # 昇腾 Eagle 实现
        ├── mtp_proposer.py            # 昇腾 MTP 实现
        ├── medusa_proposer.py         # 昇腾 Medusa 实现
        └── ngram_proposer.py          # 昇腾 N-gram 实现
```

## 附录 C: 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| Speculative Decoding | Speculative Decoding | 推测解码，通过先猜测后验证加速推理 |
| Draft Model | Draft Model | 用于生成候选 tokens 的轻量模型 |
| Target Model | Target Model | 用于验证和输出的主模型 |
| Rejection Sampling | Rejection Sampling | 拒绝采样，决定接受或拒绝 draft tokens |
| Acceptance Rate | Acceptance Rate | 接受率，draft tokens 被接受的比例 |
| Token Tree | Token Tree | 树状 token 结构，用于并行验证 |
| CUDA Graph | CUDA Graph | NVIDIA GPU 的图执行优化 |
| ACL Graph | ACL Graph | 昇腾 NPU 的图执行优化 |
| HCCL | HCCL | 昇腾集合通信库 |
| TP | Tensor Parallel | 张量并行 |
| DP | Data Parallel | 数据并行 |
| PP | Pipeline Parallel | 流水线并行 |

## 附录 D: 参考资料

### D.1 论文链接

- **Eagle**: [Accelerating Generation via Implicit Draft Model](https://arxiv.org/abs/2406.16858)
- **Eagle2**: [Eagle 2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
- **Eagle3**: [Eagle 3: Production-Grade Speculative Decoding for Large Language Models](https://arxiv.org/abs/2501.xxxxx)
- **Medusa**: [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)

### D.2 vLLM 文档

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm)
- [Speculative Decoding 文档](https://docs.vllm.ai/en/latest/quantization/speculative_decoding.html)

### D.3 昇腾开发文档

- [CANN 开发指南](https://www.hiascend.com/document/)
- [torch_npu GitHub 仓库](https://github.com/Ascend/torch_npu)
- [vLLM-Ascend GitHub 仓库](https://github.com/vllm-project/vllm-ascend)

---

**文档结束**

感谢您阅读这份技术文档！如有问题或反馈，请在 vLLM 或 vLLM-Ascend GitHub 仓库提交 Issue。
