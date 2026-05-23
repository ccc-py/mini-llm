import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModernLanguageModel, RMSNorm, precompute_freqs_cis, TransformerBlock

batch_size = 8
seq_len = 64
max_iters = 500
learning_rate = 1e-5
max_gen_len = 20
eval_every = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
vocab_size = vocab['vocab_size']
stoi = vocab['stoi']
itos = {int(k): v for k, v in vocab['itos'].items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

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

def load_qa(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '<Q>' in line and '<A>' in line:
                q, a = line.split('<A>', 1)
                pairs.append((q + '<A>', a.strip()))
    return pairs

seen_pairs = load_qa('finetune.txt')
unseen_pairs = load_qa('rl_unseen.txt')
print(f"Seen (SFT): {len(seen_pairs)} pairs, Unseen (RL): {len(unseen_pairs)} pairs")

model = ModernLanguageModel(vocab_size=vocab_size, seq_len=seq_len, device=device).to(device)
model.load_state_dict(torch.load('finetune.pt', map_location=device))
print("Loaded finetune.pt")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

rm = RewardModel(vocab_size=vocab_size, device=device).to(device)
rm.load_state_dict(torch.load('reward_model.pt', map_location=device, weights_only=True), strict=False)
rm.eval()
print("Loaded reward_model.pt")

@torch.no_grad()
def evaluate(pairs, num=100):
    model.eval()
    n = min(num, len(pairs))
    correct = 0
    for i in range(n):
        prompt, expected = pairs[i]
        prompt_ids = encode(prompt)
        gen = prompt_ids.copy()
        for _ in range(max_gen_len):
            ctx = torch.tensor(gen[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(ctx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            token = torch.multinomial(probs, 1)
            gen.append(token.item())
            if itos.get(token.item(), '') == '\n':
                break
        resp = decode(gen)[len(prompt):].strip()
        if expected.strip() in resp:
            correct += 1
    return correct / n

init_seen = evaluate(seen_pairs)
init_unseen = evaluate(unseen_pairs)
print(f"Before RL → Seen: {init_seen*100:.1f}%, Unseen: {init_unseen*100:.1f}%")
print()

baseline = 0.1
for step in range(max_iters):
    model.train()
    total_loss = 0.0
    rewards = []

    for _ in range(batch_size):
        prompt, _ = random.choice(unseen_pairs)
        prompt_ids = encode(prompt)
        gen = prompt_ids.copy()
        log_probs = []

        for _ in range(max_gen_len):
            ctx = torch.tensor(gen[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(ctx)
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            token = torch.multinomial(probs, 1)
            log_prob = -F.cross_entropy(logits_last, token.view(-1))
            log_probs.append(log_prob)
            gen.append(token.item())
            if itos.get(token.item(), '') == '\n':
                break

        full_text = decode(gen)
        full_ids = torch.tensor(encode(full_text), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            reward = rm(full_ids).item()
        reward = max(0.0, reward)
        rewards.append(reward)
        advantage = reward - baseline
        loss = -torch.stack(log_probs).sum() * advantage
        total_loss = total_loss + loss

    avg_loss = total_loss / batch_size
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    avg_reward = sum(rewards) / len(rewards)
    baseline = 0.95 * baseline + 0.05 * avg_reward

    if step % 25 == 0 or step == max_iters - 1:
        print(f"Step {step:4d} | Loss: {avg_loss.item():+.4f} | Reward: {avg_reward:.3f} | Baseline: {baseline:.3f}")

    if (step + 1) % eval_every == 0:
        seen_acc = evaluate(seen_pairs, 50)
        unseen_acc = evaluate(unseen_pairs, 50)
        print(f"  Eval → Seen: {seen_acc*100:.1f}%, Unseen: {unseen_acc*100:.1f}%")
        model.train()

torch.save(model.state_dict(), 'reinforce.pt')
print()

final_seen = evaluate(seen_pairs)
final_unseen = evaluate(unseen_pairs)

print("=" * 45)
print(f"  {'Metric':<20} {'Before':>10} {'After':>10}")
print("=" * 45)
print(f"  {'Seen accuracy':<20} {init_seen*100:>9.1f}% {final_seen*100:>9.1f}%")
print(f"  {'Unseen accuracy':<20} {init_unseen*100:>9.1f}% {final_unseen*100:>9.1f}%")
print("=" * 45)
print("Saved reinforce.pt")
