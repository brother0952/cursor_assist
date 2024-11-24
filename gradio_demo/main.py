import gradio as gr
from basic_demo import create_demo as create_basic_demo
from advanced_demo import create_demo as create_advanced_demo
from interactive_demo import create_demo as create_interactive_demo
from style_demo import create_demo as create_style_demo

def main():
    # 创建主界面，包含所有demo
    with gr.Blocks() as demo:
        gr.Markdown("# Gradio 功能展示")
        
        with gr.Tabs():
            with gr.TabItem("基础功能"):
                create_basic_demo()
            
            with gr.TabItem("高级功能"):
                create_advanced_demo()
            
            with gr.TabItem("交互式图表"):
                create_interactive_demo()
            
            with gr.TabItem("样式定制"):
                create_style_demo()

    # 启动应用
    demo.launch()

if __name__ == "__main__":
    main() 