from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
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
        logits = outputs.logits
        
        _, predictions = torch.max(logits, dim=1)
        correct_predictions += torch.sum(predictions == labels)
        total_predictions += len(labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_predictions
    
    return avg_loss, accuracy

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='评估中'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, predictions = torch.max(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_predictions
    
    return avg_loss, accuracy

def predict_text(model, tokenizer, text, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
    
    return prediction.item()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型和tokenizer
    model_name = 'bert-base-chinese'  # 使用中文BERT
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 二分类任务
    ).to(device)
    
    # 准备示例数据（这里使用简单的示例数据，实际应用中需要替换为真实数据）
    texts = [
        "这部电影很好看，情节紧凑，演技出色",
        "画面很差，故事情节混乱，浪费时间",
        "服务态度很好，环境优美，值得推荐",
        "价格太贵了，质量一般，不会再来"
    ]
    labels = [1, 0, 1, 0]  # 1表示正面评价，0表示负面评价
    
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据集
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # 训练参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    
    # 训练循环
    print("开始训练...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f'训练集 - 平均损失: {train_loss:.4f}, 准确率: {train_acc:.4f}')
        
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f'验证集 - 平均损失: {val_loss:.4f}, 准确率: {val_acc:.4f}')
    
    # 保存模型
    model.save_pretrained('./text_classification_model')
    tokenizer.save_pretrained('./text_classification_model')
    print("\n模型已保存")
    
    # 测试预测
    test_texts = [
        "这家餐厅的菜品非常美味，服务也很周到",
        "商品质量太差了，而且客服态度很不好"
    ]
    
    print("\n测试预测:")
    for text in test_texts:
        prediction = predict_text(model, tokenizer, text, device)
        sentiment = "正面" if prediction == 1 else "负面"
        print(f"文本: {text}")
        print(f"预测情感: {sentiment}\n")

if __name__ == '__main__':
    main() 