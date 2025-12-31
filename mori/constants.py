"""Mori 项目常量定义

包含项目中使用的各种默认常量，便于统一管理和修改。
"""

# 向量存储相关常量
DEFAULT_VECTOR_STORE_PROVIDER = "qdrant"
DEFAULT_COLLECTION_NAME = "mem0migrations"

# 嵌入模型默认维度
DEFAULT_EMBEDDING_DIMENSIONS = {
    "openai": 1536,
    "dashscope": 1536,
    "gemini": 768,
    "ollama": 768,
}

# 默认嵌入维度（用于回退）
DEFAULT_EMBEDDING_DIM = 1536

# 日志相关常量
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"

# 服务器相关常量
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 7860

# 存储相关常量
DEFAULT_MEMORY_STORAGE_PATH = "data/memory"

# 记忆模式常量
MEMORY_MODE_AGENT_CONTROL = "agent_control"
MEMORY_MODE_STATIC_CONTROL = "static_control"
MEMORY_MODE_BOTH = "both"

VALID_MEMORY_MODES = {
    MEMORY_MODE_AGENT_CONTROL,
    MEMORY_MODE_STATIC_CONTROL,
    MEMORY_MODE_BOTH,
}
