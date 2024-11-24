import gradio as gr

def greet(name, intensity):
    """基础文本处理演示"""
    return f"Hello {name}!" * intensity

def image_processor(img):
    """基础图像处理演示"""
    return img.rotate(90)

def create_demo():
    # 创建多个演示并组合
    with gr.Blocks() as demo:
        gr.Markdown("# 基础功能演示")
        
        with gr.Tab("文本处理"):
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label="输入名字")
                    num = gr.Slider(minimum=1, maximum=5, step=1, label="重复次数")
                    greet_btn = gr.Button("问候")
                with gr.Column():
                    output = gr.Textbox(label="输出结果")
            
            greet_btn.click(fn=greet, 
                          inputs=[name, num], 
                          outputs=output)
        
        with gr.Tab("图像处理"):
            with gr.Row():
                image_input = gr.Image(label="上传图片")
                image_output = gr.Image(label="处理结果")
            
            image_input.change(fn=image_processor,
                             inputs=image_input,
                             outputs=image_output)

if __name__ == "__main__":
    demo = create_demo()
    demo.launch() 