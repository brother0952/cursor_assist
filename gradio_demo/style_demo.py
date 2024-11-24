import gradio as gr
import random

def random_style():
    """返回随机样式"""
    colors = ["red", "blue", "green", "purple", "orange"]
    return random.choice(colors)

def create_demo():
    with gr.Blocks(css="""
        .important { color: red; font-weight: bold; }
        .custom-box { border: 2px solid blue; padding: 10px; }
    """) as demo:
        gr.Markdown("# 样式演示")
        
        with gr.Row():
            with gr.Column(elem_classes="custom-box"):
                gr.Markdown("## 自定义样式示例", elem_classes="important")
                text_input = gr.Textbox(
                    label="带样式的输入框",
                    placeholder="请输入文本...",
                    container=True
                )
        
        with gr.Row():
            with gr.Column():
                color_button = gr.Button(
                    "更改颜色",
                    variant="primary"
                )
                style_output = gr.Textbox(
                    label="样式输出",
                    container=True
                )
        
        color_button.click(
            fn=random_style,
            outputs=style_output
        )

if __name__ == "__main__":
    demo = create_demo()
    demo.launch() 