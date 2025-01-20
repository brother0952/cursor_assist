import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
import random
import json
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (batch_size, sequence_length, embedding_size)
        
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, cell):
        # x shape: (batch_size, 1)
        x = x.unsqueeze(1)
        
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (batch_size, 1, embedding_size)
        
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (batch_size, 1, hidden_size)
        
        predictions = self.fc(outputs)
        # predictions shape: (batch_size, 1, output_size)
        predictions = predictions.squeeze(1)
        
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(source)
        
        # 第一个解码器输入是特殊的开始符号
        decoder_input = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output
            
            # 是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1
        
        return outputs

class TranslationDataset(Dataset):
    def __init__(self, data_path, src_lang='en', tgt_lang='zh', max_length=50):
        self.data_path = data_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 初始化分词器
        self.src_tokenizer = get_tokenizer('basic_english')
        self.tgt_tokenizer = get_tokenizer('spacy', language='zh')
        
        # 构建词汇表
        self.src_vocab = self.build_vocabulary(
            [item[src_lang] for item in self.data],
            self.src_tokenizer
        )
        self.tgt_vocab = self.build_vocabulary(
            [item[tgt_lang] for item in self.data],
            self.tgt_tokenizer
        )
    
    def build_vocabulary(self, texts, tokenizer, min_freq=2):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for word, count in counter.items():
            if count >= min_freq:
                vocab[word] = len(vocab)
        
        return vocab
    
    def text_to_sequence(self, text, tokenizer, vocab):
        tokens = tokenizer(text)
        sequence = [vocab.get(token, vocab['<unk>']) for token in tokens]
        sequence = [vocab['<sos>']] + sequence + [vocab['<eos>']]
        
        if len(sequence) < self.max_length:
            sequence += [vocab['<pad>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length-1] + [vocab['<eos>']]
        
        return torch.tensor(sequence)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]
        
        src_seq = self.text_to_sequence(src_text, self.src_tokenizer, self.src_vocab)
        tgt_seq = self.text_to_sequence(tgt_text, self.tgt_tokenizer, self.tgt_vocab)
        
        return src_seq, tgt_seq

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        running_loss = 0.0
        
        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            
            output = output[:, 1:].reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

def translate(model, sentence, src_tokenizer, src_vocab, tgt_vocab, device, max_length=50):
    model.eval()
    
    # 将词转换为索引
    tokens = src_tokenizer(sentence)
    indexes = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    indexes = [src_vocab['<sos>']] + indexes + [src_vocab['<eos>']]
    src_tensor = torch.tensor(indexes).unsqueeze(0).to(device)
    
    # 获取编码器输出
    with torch.no_grad():
        encoder_hidden, encoder_cell = model.encoder(src_tensor)
    
    # 开始解码
    tgt_indexes = [tgt_vocab['<sos>']]
    for _ in range(max_length):
        tgt_tensor = torch.tensor([tgt_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(tgt_tensor, encoder_hidden, encoder_cell)
            
        pred_token = output.argmax(1).item()
        tgt_indexes.append(pred_token)
        
        if pred_token == tgt_vocab['<eos>']:
            break
    
    # 将索引转换回词
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}
    translated_tokens = [tgt_vocab_inv[i] for i in tgt_indexes[1:-1]]
    
    return ''.join(translated_tokens)

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
    dataset = TranslationDataset('translation_data.json')  # 请替换为实际的数据文件路径
    
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    
    # 模型参数
    INPUT_DIM = len(dataset.src_vocab)
    OUTPUT_DIM = len(dataset.tgt_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    # 创建模型
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab['<pad>'])
    
    # 训练模型
    print("开始训练...")
    train_model(model, train_loader, optimizer, criterion, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'translation_model.pth')
    
    # 测试翻译
    test_sentence = "Hello, how are you?"
    translation = translate(
        model,
        test_sentence,
        dataset.src_tokenizer,
        dataset.src_vocab,
        dataset.tgt_vocab,
        device
    )
    print(f"\n测试翻译:")
    print(f"输入: {test_sentence}")
    print(f"输出: {translation}")
    
    print("\n训练完成！模型已保存。")

if __name__ == '__main__':
    main() 