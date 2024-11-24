import gradio as gr
import json
from config_json.config_manager import ConfigManager
from config_json.config_schema import DEFAULT_SCHEMA
from config_json.validators import ValidationError

class ConfigApp:
    def __init__(self, config_file: str = "config.json"):
        self.config_manager = ConfigManager(config_file, DEFAULT_SCHEMA)
        self.interface = None

    def update_value(self, section: str, field: str, value: str):
        """更新配置值"""
        try:
            self.config_manager.update_field(section, field, value)
            self.config_manager.save_config()
            return f"更新成功: {section}.{field} = {value}", None
        except ValidationError as e:
            return None, str(e)

    def create_interface(self):
        """创建图形界面"""
        with gr.Blocks() as interface:
            gr.Markdown("# JSON配置工具")
            
            # 创建配置部分
            for section, fields in DEFAULT_SCHEMA.items():
                with gr.Tab(section.upper()):
                    for field_name, field_schema in fields.items():
                        with gr.Row():
                            if field_schema.field_type == "text":
                                input_component = gr.Textbox(
                                    label=field_schema.label,
                                    value=self.config_manager.config[section][field_name]
                                )
                            elif field_schema.field_type == "number":
                                input_component = gr.Number(
                                    label=field_schema.label,
                                    value=self.config_manager.config[section][field_name],
                                    minimum=field_schema.min_value,
                                    maximum=field_schema.max_value
                                )
                            elif field_schema.field_type == "radio":
                                input_component = gr.Radio(
                                    choices=field_schema.options,
                                    label=field_schema.label,
                                    value=self.config_manager.config[section][field_name]
                                )
                            
                            # 更新按钮和状态显示
                            update_btn = gr.Button(f"更新{field_schema.label}")
                            status_output = gr.Textbox(label="状态", interactive=False)
                            error_output = gr.Textbox(label="错误信息", interactive=False)
                            
                            # 绑定更新事件
                            update_btn.click(
                                fn=lambda v, s=section, f=field_name: self.update_value(s, f, v),
                                inputs=[input_component],
                                outputs=[status_output, error_output]
                            )
            
            # 显示当前配置
            with gr.Tab("当前配置"):
                gr.JSON(value=lambda: self.config_manager.config)

        self.interface = interface
        return interface

    def launch(self):
        """启动应用"""
        if not self.interface:
            self.create_interface()
        self.interface.launch()

def main():
    app = ConfigApp()
    app.launch()

if __name__ == "__main__":
    main() 