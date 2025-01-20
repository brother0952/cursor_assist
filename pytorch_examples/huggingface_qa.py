from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

class QuestionAnsweringDataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer, max_length=384):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        context = str(self.contexts[idx])
        answer = self.answers[idx]
        
        # 使用tokenizer处理问题和上下文
        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 获取答案的起始和结束位置
        answer_start = answer['answer_start']
        answer_end = answer_start + len(answer['text'])
        
        # 将原始文本位置转换为token位置
        tokenized_answer = self.tokenizer.encode_plus(
            answer['text'],
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        # 找到答案在token序列中的位置
        input_ids = encoding['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # 注意：这是一个简化的实现，实际应用中需要更复杂的对齐逻辑
        start_positions = torch.tensor([answer_start])
        end_positions = torch.tensor([answer_end])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': start_positions,
            'end_positions': end_positions
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc='训练中')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def get_answer(model, tokenizer, question, context, device):
    model.eval()
    
    # 处理输入
    encoding = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        max_length=384,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
    
    # 获取最可能的答案范围
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    
    # 将token ID转换回文本
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
    
    return answer

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型和tokenizer
    model_name = 'bert-base-chinese'  # 使用中文BERT
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
    
    # 准备示例数据
    contexts = [
        "北京是中国的首都，有着悠久的历史文化。故宫是中国明清两代的皇宫，位于北京中心，占地72万平方米。",
        "人工智能是计算机科学的一个重要分支，它致力于研究和开发能模拟人类智能的系统。深度学习是人工智能的一个重要方法。"
    ]
    
    questions = [
        "故宫在哪里？",
        "什么是人工智能的重要方法？"
    ]
    
    answers = [
        {'text': '北京中心', 'answer_start': 35},
        {'text': '深度学习', 'answer_start': 42}
    ]
    
    # 创建数据集
    dataset = QuestionAnsweringDataset(contexts, questions, answers, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 训练参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    
    # 训练循环
    print("开始训练...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        avg_loss = train_epoch(model, data_loader, optimizer, device)
        print(f'平均损失: {avg_loss:.4f}')
    
    # 保存模型
    model.save_pretrained('./qa_model')
    tokenizer.save_pretrained('./qa_model')
    print("\n模型已保存")
    
    # 测试问答
    test_contexts = [
        "张三是一名程序员，他在一家科技公司工作。他主要负责开发人工智能相关的项目。",
        "太阳系有八大行星，其中地球是第三颗行星。月球是地球唯一的天然卫星。"
    ]
    
    test_questions = [
        "张三是做什么工作的？",
        "月球是什么？"
    ]
    
    print("\n测试问答:")
    for context, question in zip(test_contexts, test_questions):
        answer = get_answer(model, tokenizer, question, context, device)
        print(f"\n上下文: {context}")
        print(f"问题: {question}")
        print(f"预测答案: {answer}")

if __name__ == '__main__':
    main() 