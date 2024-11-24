from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class FieldSchema:
    """字段模式定义"""
    name: str                    # 字段名称
    field_type: str             # 字段类型 (text, number, list, radio, checkbox)
    label: str                  # 显示标签
    default: Any = None         # 默认值
    options: List[Any] = None   # 选项（用于radio, list等）
    validators: List[str] = None  # 验证器列表
    depends_on: List[str] = None # 依赖的其他字段
    min_value: Optional[float] = None  # 最小值（用于number类型）
    max_value: Optional[float] = None  # 最大值（用于number类型）

# 示例配置模式
DEFAULT_SCHEMA = {
    "server": {
        "host": FieldSchema(
            name="host",
            field_type="text",
            label="服务器地址",
            default="localhost",
            validators=["required", "ip_or_hostname"]
        ),
        "port": FieldSchema(
            name="port",
            field_type="number",
            label="端口号",
            default=8080,
            min_value=1,
            max_value=65535,
            validators=["required", "port_range"]
        ),
        "debug": FieldSchema(
            name="debug",
            field_type="radio",
            label="调试模式",
            default="False",
            options=["True", "False"]
        )
    },
    "database": {
        "type": FieldSchema(
            name="type",
            field_type="radio",
            label="数据库类型",
            default="sqlite",
            options=["mysql", "postgresql", "sqlite"]
        ),
        "name": FieldSchema(
            name="name",
            field_type="text",
            label="数据库名称",
            default="app.db",
            validators=["required"]
        )
    }
} 