from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class FieldSchema:
    """JSON字段模式定义"""
    name: str                    # 字段名称
    field_type: str             # 字段类型 (text, number, list, radio, checkbox, etc)
    label: str                  # 显示标签
    default: Any = None         # 默认值
    options: List[Any] = None   # 选项（用于radio, list等）
    validators: List[str] = None  # 验证器列表
    description: str = ""       # 字段描述
    required: bool = False      # 是否必填
    min_value: Optional[float] = None  # 最小值（用于number类型）
    max_value: Optional[float] = None  # 最大值（用于number类型） 