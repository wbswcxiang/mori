# Ask Mode - 项目文档规则（仅非显而易见内容）

## 项目结构理解

### 配置系统
- 配置文件必须分离：`models.yaml` (模型), `agents.yaml` (agent), `config.yaml` (全局)
- Agent 不直接配置模型，而是通过 `model` 字段引用 `models.yaml` 中的模型名称
- 环境变量使用 `${ENV_VAR_NAME}` 格式，在 [`config.py:resolve_env_var()`](../../mori/config.py:27) 中解析

### 模板系统
- 模板查找有优先级：自定义模板 (`config/template/`) > 内置模板 (`mori/template/internal_template/`)
- 简短名称（如 `mori`）会自动添加 `.jinja2` 扩展名并按优先级查找
- 运行时信息（时间、日期）在代码中注入，不在模板中硬编码

### 长期记忆架构
- 嵌入模型配置独立于主模型，在 `models.yaml` 的 `embedding_models` 部分
- 长期记忆有三种模式：`agent_control`（Agent 自主）、`static_control`（代码控制）、`both`（两者）
- 存储可以是内存或磁盘持久化，通过 `on_disk` 配置
