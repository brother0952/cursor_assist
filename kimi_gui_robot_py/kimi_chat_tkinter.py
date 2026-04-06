import tkinter as tk
from tkinter import scrolledtext
import requests
import json

class KimiChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kimi 聊天")
        self.root.geometry("600x400")
        
        # API密钥
        self.api_key = "sk-0PnppjKvTYr7uxYOigwfrLOY5xsYPMP6lNhKWKGccXAoFEgq"  # 请替换成你的实际API密钥
        
        # 创建聊天显示区域
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20)
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 创建输入框
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.input_box = tk.Entry(self.input_frame)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.send_button = tk.Button(self.input_frame, text="发送", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # 绑定回车键
        self.input_box.bind("<Return>", lambda e: self.send_message())

    def send_message(self):
        user_message = self.input_box.get()
        if not user_message.strip():
            return
            
        # 显示用户消息
        self.chat_display.insert(tk.END, "你: " + user_message + "\n\n")
        self.chat_display.see(tk.END)
        self.input_box.delete(0, tk.END)
        
        # 调用Kimi API
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "moonshot-v1-8k",  # 添加模型参数
                "messages": [{"role": "user", "content": user_message}],
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                kimi_response = response.json()["choices"][0]["message"]["content"]
                self.chat_display.insert(tk.END, "Kimi: " + kimi_response + "\n\n")
            else:
                error_message = f"错误: API调用失败\n状态码: {response.status_code}\n响应内容: {response.text}\n\n"
                self.chat_display.insert(tk.END, error_message)
                
        except Exception as e:
            self.chat_display.insert(tk.END, f"错误: {str(e)}\n\n")
            
        self.chat_display.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = KimiChatGUI(root)
    root.mainloop() 