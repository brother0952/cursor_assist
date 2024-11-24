import json
import os
from typing import Dict, Any
from config_json.validators import VALIDATORS, ValidationError

class ConfigManager:
    def __init__(self, config_file: str, schema: Dict):
        self.config_file = config_file
        self.schema = schema
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._get_default_config()

    def save_config(self) -> None:
        """保存配置到文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        config = {}
        for section, fields in self.schema.items():
            config[section] = {}
            for field_name, field_schema in fields.items():
                config[section][field_name] = field_schema.default
        return config

    def validate_field(self, section: str, field: str, value: Any) -> None:
        """验证字段值"""
        field_schema = self.schema[section][field]
        if field_schema.validators:
            for validator_name in field_schema.validators:
                validator = VALIDATORS.get(validator_name)
                if validator:
                    try:
                        validator(value)
                    except ValidationError as e:
                        raise ValidationError(f"{field_schema.label}: {str(e)}")

    def update_field(self, section: str, field: str, value: Any) -> None:
        """更新配置字段"""
        self.validate_field(section, field, value)
        if section not in self.config:
            self.config[section] = {}
        self.config[section][field] = value 