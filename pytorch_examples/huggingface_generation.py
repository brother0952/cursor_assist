from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

class TextGenerationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc='训练中')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def generate_text(model, tokenizer, prompt, max_length=100, num_return_sequences=1):
    # 创建文本生成pipeline
    generator = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=model.device
    )
    
    # 生成文本
    outputs = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        temperature=0.7
    )
    
    return [output['generated_text'] for output in outputs]

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型和tokenizer
    model_name = 'uer/gpt2-chinese-cluecorpussmall'  # 使用中文GPT2
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    # 如果tokenizer没有pad_token，设置为eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # 准备示例数据
    train_texts = [
        "人工智能正在改变我们的生活方式，",
        "春天来了，樱花开满了整个公园，",
        "科技的发展带来了很多便利，",
        "在这个信息化的时代，"
    ]
    
    # 创建数据集
    dataset = TextGenerationDataset(train_texts, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 训练参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    # 训练循环
    print("开始训练...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        avg_loss = train_epoch(model, data_loader, optimizer, device)
        print(f'平均损失: {avg_loss:.4f}')
    
    # 保存模型
    model.save_pretrained('./text_generation_model')
    tokenizer.save_pretrained('./text_generation_model')
    print("\n模型已保存")
    
    # 测试文本生成
    test_prompts = [
        "人工智能的未来发展",
        "在一个阳光明媚的早晨",
        "随着科技的进步"
    ]
    
    print("\n测试文本生成:")
    for prompt in test_prompts:
        print(f"\n提示语: {prompt}")
        generated_texts = generate_text(model, tokenizer, prompt)
        for i, text in enumerate(generated_texts, 1):
            print(f"生成文本 {i}: {text}")

def interactive_generation():
    # 加载模型和tokenizer
    model_name = 'uer/gpt2-chinese-cluecorpussmall'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("交互式文本生成 (输入'quit'退出)")
    while True:
        prompt = input("\n请输入提示语: ")
        if prompt.lower() == 'quit':
            break
        
        generated_texts = generate_text(model, tokenizer, prompt)
        for i, text in enumerate(generated_texts, 1):
            print(f"\n生成文本 {i}:")
            print(text)

if __name__ == '__main__':
    main()
    # 取消下面的注释来启用交互式生成
    # interactive_generation() 