import gradio as gr
import requests
import json
from config import API_KEY

def chat_with_kimi(message, history):
    # 构建完整的对话历史
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    
    # API配置
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 准备请求数据
    data = {
        "model": "moonshot-v1-8k",
        "messages": messages,
        "temperature": 0.7
    }
    
    try:
        # 发送API请求
        response = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            # 获取回复
            kimi_response = response.json()["choices"][0]["message"]["content"]
            return kimi_response
        else:
            return f"错误: API调用失败\n状态码: {response.status_code}"
            
    except Exception as e:
        return f"错误: {str(e)}"

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Kimi 聊天助手")
    gr.Markdown("与 Kimi AI 进行对话")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="在这里输入消息...", label="用户输入")
    clear = gr.Button("清除对话")
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history):
        user_message = history[-1][0]
        bot_message = chat_with_kimi(user_message, history[:-1])
        history[-1][1] = bot_message
        return history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

# 启动应用
if __name__ == "__main__":
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7860,
        quiet=True
    ) 