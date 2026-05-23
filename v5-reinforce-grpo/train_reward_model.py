import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RMSNorm, precompute_freqs_cis, TransformerBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
vocab_size = vocab['vocab_size']
stoi = vocab['stoi']
itos = {int(k): v for k, v in vocab['itos'].items()}
encode = lambda s: [stoi[c] for c in s]

class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, seq_len=64, device='cpu'):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.score_head = nn.Linear(d_model, 1, bias=False)
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, seq_len * 2).to(device)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.norm(x)
        return self.score_head(x[:, -1, :]).squeeze(-1)

rm = RewardModel(vocab_size=vocab_size, device=device).to(device)

pt = torch.load('pretrain.pt', map_location=device)
rm.tok_emb.weight.data.copy_(pt['tok_emb.weight'])
rm_layers = {k[7:]: v for k, v in pt.items() if k.startswith('layers.')}
rm.layers.load_state_dict(rm_layers)
rm.norm.weight.data.copy_(pt['norm.weight'])
print("Initialized RM from pretrain.pt")

optimizer = torch.optim.AdamW(rm.parameters(), lr=1e-4)

problems = [(a, b, a+b) for a in range(1, 10) for b in range(1, 10)]
random.shuffle(problems)

training_data = []
for a, b, correct in problems:
    prompt = f"<Q>{a}+{b}=？<A>"
    training_data.append((prompt + str(correct), 1.0))
    for delta in [1, 2, -1, -2]:
        wrong = correct + delta
        if wrong >= 2:
            score = max(0.0, 0.5 - 0.1 * abs(delta))
            training_data.append((prompt + str(wrong), score))
    for wrong in [correct // 2, correct * 2, 99]:
        if wrong != correct and wrong >= 0:
            training_data.append((prompt + str(wrong), 0.0))

random.shuffle(training_data)
print(f"RM training samples: {len(training_data)}")

rm.train()
for epoch in range(15):
    total_loss = 0
    for text, score in training_data:
        ids = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)
        pred = rm(ids)
        loss = F.mse_loss(pred.squeeze(), torch.tensor(score, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0 or epoch == 14:
        print(f"RM Epoch {epoch:2d} | MSE: {total_loss / len(training_data):.6f}")

torch.save(rm.cpu().state_dict(), 'reward_model.pt')
rm.to(device)
print("Saved reward_model.pt")

rm.eval()
test_set = [
    ("<Q>3+5=？<A>8", 1.0),
    ("<Q>3+5=？<A>7", 0.4),
    ("<Q>3+5=？<A>3", 0.0),
    ("<Q>3+5=？<A>99", 0.0),
]
print("\nRM Test:")
for text, expected in test_set:
    ids = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)
    pred = rm(ids).item()
    print(f"  {text:<30} expected={expected:.2f}  predicted={pred:.3f}")
