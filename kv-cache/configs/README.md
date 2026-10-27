# 原始 config（离线副本）

抓取时间：2026-10-17；来源 ModelScope。

| 文件 | 模型 ID | 说明 |
|---|---|---|
| `deepseek_v41_flash.json` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | 用户提供的 config |
| `kimi_k3.json` | `moonshotai/Kimi-K3` | Kimi K3（含 vision_config） |
| `qwen3_8_flash_next.json` | `Qwen/Qwen3.8-Flash-Next` | 架构名 `Qwen4ExpForConditionalGeneration` |
| `glm5_3_flash.json` | `ZhipuAI/GLM-5.3-Flash` | 架构名 `Glm5NextForConditionalGeneration` |

ModelScope raw 地址模板：

```
https://modelscope.cn/models/<org>/<name>/resolve/master/config.json
```

注意：Qwen 的模型名是 `Qwen3.8-Flash-Next`，不是 `Qwen3.8-Flash`；GLM 的组织是 `ZhipuAI`。如果模型更新，config 字段可能变化，以仓库代码对应的 `transformers_utils/configs` 为准。
