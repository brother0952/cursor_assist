import gradio as gr
import json
from pathlib import Path
from typing import Dict, Any, List

class JSONEditor:
    def __init__(self):
        self.current_file = None
        self.json_data = {}
        
    def load_json(self, file_obj) -> tuple[Dict[str, Any], str]:
        """加载JSON文件"""
        try:
            if file_obj is None:
                return None, "请选择JSON文件"
            
            # 读取文件内容
            if hasattr(file_obj, 'name'):  # 如果是文件对象
                with open(file_obj.name, 'r', encoding='utf-8') as f:
                    self.json_data = json.load(f)
            else:  # 如果是文件路径
                with open(file_obj, 'r', encoding='utf-8') as f:
                    self.json_data = json.load(f)
                    
            self.current_file = file_obj
            return self.json_data, "文件加载成功"
        except Exception as e:
            return None, f"加载文件失败: {str(e)}"
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks() as interface:
            gr.Markdown("# JSON配置编辑器")
            
            # 文件输入和状态显示
            with gr.Row():
                file_input = gr.File(label="选择JSON文件", file_types=[".json"])
                load_status = gr.Textbox(label="状态", interactive=False)
            
            # JSON内容显示
            json_display = gr.JSON(label="当前JSON内容")
            
            # 创建固定的表单组件
            with gr.Tabs() as tabs:
                with gr.Tab("服务器配置"):
                    host = gr.Textbox(label="主机地址")
                    port = gr.Number(label="端口", precision=0)
                    debug = gr.Checkbox(label="调试模式")
                
                with gr.Tab("数据库配置"):
                    db_type = gr.Dropdown(
                        label="数据库类型",
                        choices=["mysql", "postgresql", "sqlite"],
                    )
                    db_name = gr.Textbox(label="数据库名称")
                    db_user = gr.Textbox(label="用户名")
                    db_password = gr.Textbox(label="密码", type="password")
            
            def update_form(file_obj):
                if file_obj is None:
                    return "请选择文件", None
                
                json_data, message = self.load_json(file_obj)
                if json_data is None:
                    return message, None
                
                return message, json_data
            
            # 文件上传事件
            file_input.change(
                fn=update_form,
                inputs=[file_input],
                outputs=[load_status, json_display]
            )
            
            # 保存按钮
            save_button = gr.Button("保存修改")
            save_status = gr.Textbox(label="保存状态", interactive=False)
            
            def save_changes(host_val, port_val, debug_val, 
                           db_type_val, db_name_val, db_user_val, db_pass_val):
                try:
                    new_config = {
                        "server": {
                            "host": host_val,
                            "port": port_val,
                            "debug": debug_val
                        },
                        "database": {
                            "type": db_type_val,
                            "name": db_name_val,
                            "user": db_user_val,
                            "password": db_pass_val
                        }
                    }
                    
                    # 保存到文件
                    with open("config.json", "w", encoding="utf-8") as f:
                        json.dump(new_config, f, indent=4, ensure_ascii=False)
                    
                    return "配置已保存"
                except Exception as e:
                    return f"保存失败: {str(e)}"
            
            # 保存按钮事件
            save_button.click(
                fn=save_changes,
                inputs=[host, port, debug, db_type, db_name, db_user, db_password],
                outputs=[save_status]
            )
        
        return interface

def main():
    editor = JSONEditor()
    interface = editor.create_interface()
    interface.launch()

if __name__ == "__main__":
    main() 