# Code Mode - 项目编码规则（仅非显而易见内容）

## 关键约定

### 工具函数返回值
- 必须返回 [`ToolResponse`](../../mori/tool/internal_tools/example_tools.py:20) 对象，不能返回普通字符串
- 使用 `TextBlock` 包装文本内容

### 长期记忆集成
- 主模型必须用 [`NonStreamingModelWrapper`](../../mori/utils/model_wrapper.py:9) 包装（Mem0 不支持流式）
- 嵌入模型必须显式设置 `dimensions` 参数
- 向量存储必须通过 `VectorStoreConfig` 传递 `embedding_model_dims`

### 模型配置
- OpenAI 兼容接口：`model_type: openai` + `base_url` 在 `client_args` 中
- 嵌入模型 base_url：OpenAI 直接传参，Ollama 通过 `client_args`

### 响应处理
- Agent 响应可能是字符串或 `TextBlock` 列表
- 统一使用 [`_extract_text_from_response()`](../../mori/mori.py:383) 处理
