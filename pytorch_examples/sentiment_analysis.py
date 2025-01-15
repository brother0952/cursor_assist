import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
from collections import Counter
import random

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, sentence length]
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch size, sentence length, embedding dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # hidden shape: [n layers, batch size, hidden dim]
        
        hidden = hidden[-1, :, :]
        # hidden shape: [batch size, hidden dim]
        
        return self.fc(hidden)

class IMDBDataset:
    def __init__(self, split='train'):
        self.tokenizer = get_tokenizer('basic_english')
        self.split = split
        self.dataset = IMDB(split=split)
        
        # 构建词汇表
        def yield_tokens(data_iter):
            for label, text in data_iter:
                yield self.tokenizer(text)
        
        self.vocab = build_vocab_from_iterator(
            yield_tokens(self.dataset),
            specials=['<unk>', '<pad>'],
            min_freq=10
        )
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        # 转换数据集
        self.examples = []
        for label, text in self.dataset:
            tokens = self.tokenizer(text)
            ids = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)
            self.examples.append((ids, 1 if label == 'pos' else 0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(_text)
    
    # 填充序列
    text_list = pad_sequence(text_list, batch_first=True, padding_value=1)  # 1 is <pad>
    label_list = torch.tensor(label_list)
    return text_list, label_list

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for text, labels in iterator:
        text, labels = text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(text)
        
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for text, labels in iterator:
            text, labels = text.to(device), labels.to(device)
            
            predictions = model(text)
            
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def main():
    # 设置随机种子
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = IMDBDataset('train')
    test_dataset = IMDBDataset('test')
    
    # 创建数据加载器
    BATCH_SIZE = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # 模型参数
    VOCAB_SIZE = len(train_dataset.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT = 0.5
    
    # 创建模型
    model = SentimentLSTM(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        N_LAYERS,
        DROPOUT
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练模型
    N_EPOCHS = 5
    best_valid_loss = float('inf')
    
    print("开始训练...")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'sentiment_model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
    
    print('训练完成!')

if __name__ == '__main__':
    main() 