import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
class TextDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.tokenizer(text)
        token_ids = [self.vocab[token] for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels

data = [
    ("I love this movie", 1),
    ("This film is amazing", 1),
    ("I hate this movie", 0),
    ("This movie is terrible", 0)
]

# 定义分词器和词汇表
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator([tokenizer(text) for text, _ in data], specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])
dataset = TextDataset(data, vocab, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_batch)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text)  # [batch_size, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)   # Global Average Pooling
        return self.fc(pooled)
data = [
    ("I love this movie", 1),
    ("This film is amazing", 1),
    ("I hate this movie", 0),
    ("This movie is terrible", 0)
]
# 定义分词器和词汇表
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator([tokenizer(text) for text, _ in data], specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])
dataset = TextDataset(data, vocab, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_batch)
# 参数
vocab_size = len(vocab)
embed_dim = 16
num_class = 2
model = TextClassifier(vocab_size, embed_dim, num_class)
model = model.cuda()
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
epochs = 100
for epoch in range(epochs):
    for texts, labels in tqdm(dataloader, desc="Training Progress"):
        texts = texts.cuda()
        labels = labels.cuda()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

test_sentence = "This movie is amazing"
tokens = [vocab[token] for token in tokenizer(test_sentence)]
tokens_tensor = torch.tensor(tokens).unsqueeze(0).cuda()  # Add batch dimension
output = model(tokens_tensor)
# predicted_label = torch.argmax(output, dim=1).item()
predicted_score,predicted_label = torch.max(output, dim=1)
print(f"Predicted Label: {predicted_label} {predicted_score}")
