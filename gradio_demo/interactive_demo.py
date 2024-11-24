import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px

def create_plot(chart_type, data_size):
    """创建交互式图表"""
    # 生成示例数据
    x = np.linspace(0, 10, data_size)
    y = np.sin(x) + np.random.normal(0, 0.1, data_size)
    df = pd.DataFrame({"x": x, "y": y})
    
    if chart_type == "折线图":
        fig = px.line(df, x="x", y="y")
    elif chart_type == "散点图":
        fig = px.scatter(df, x="x", y="y")
    else:
        fig = px.bar(df, x="x", y="y")
    
    return fig

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 交互式图表演示")
        
        with gr.Row():
            with gr.Column():
                chart_type = gr.Dropdown(
                    choices=["折线图", "散点图", "柱状图"],
                    value="折线图",
                    label="图表类型"
                )
                data_size = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=10,
                    label="数据点数量"
                )
                update_btn = gr.Button("更新图表")
            
            with gr.Column():
                plot_output = gr.Plot(label="图表显示")
        
        update_btn.click(fn=create_plot,
                        inputs=[chart_type, data_size],
                        outputs=plot_output)

if __name__ == "__main__":
    demo = create_demo()
    demo.launch() 