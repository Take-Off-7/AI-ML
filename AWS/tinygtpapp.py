import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Sample text - bigger example works better
text = "hello world this is a mini GPT hello world this is fun GPT"

# Split text into words (tokens)
words = text.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

# Create word to index and index to word mappings
stoi = { w:i for i,w in enumerate(vocab) }
itos = { i:w for i,w in enumerate(vocab) }

def encode(words_list):
    return [stoi[w] for w in words_list]

def decode(indices):
    return ' '.join([itos[i] for i in indices])

data = torch.tensor(encode(words), dtype=torch.long)

block_size = 4  # context window of 4 words
X, Y = [], []

for i in range(len(data) - block_size):
    X.append(data[i:i+block_size])
    Y.append(data[i+1:i+block_size+1])

X = torch.stack(X)
Y = torch.stack(Y)

class SelfAttentionHead(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) / math.sqrt(C)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class GPTBlock(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.sa = SelfAttentionHead(embed_size, embed_size)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[GPTBlock(embed_size) for _ in range(2)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok = self.token_embed(idx)
        pos = self.pos_embed(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        else:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            return logits, loss

model = TinyGPT(vocab_size, embed_size=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1000):
    idx = torch.randint(len(X), (16,))
    xb, yb = X[idx], Y[idx]
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")
        model.eval()
        context = torch.tensor([encode(["hello", "world", "this", "is"])], dtype=torch.long)
        generated = context
        for _ in range(10):  # generate 10 words
            logits = model(generated[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
        print("Generated:", decode(generated[0].tolist()))
        model.train()