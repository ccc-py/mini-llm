import json
import random
import argparse
import torch
import torch.nn.functional as F
from model import ModernLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch', type=int, default=8)
args = parser.parse_args()

seq_len = 64
max_iters = args.steps
learning_rate = args.lr
batch_size = args.batch
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
print(f"Seen: {len(seen_pairs)} pairs, Unseen: {len(unseen_pairs)} pairs")

model = ModernLanguageModel(vocab_size=vocab_size, seq_len=seq_len, device=device).to(device)
model.load_state_dict(torch.load('finetune.pt', map_location=device))
print("Loaded finetune.pt")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def compute_reward(response, expected):
    if expected.strip() in response.strip():
        return 1.0
    return 0.0

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
print(f"REINFORCE(EMA): steps={max_iters}, batch={batch_size}, lr={learning_rate}")
print(f"Before → Seen: {init_seen*100:.1f}%, Unseen: {init_unseen*100:.1f}%")
print()

baseline = 0.1
for step in range(max_iters):
    model.train()
    total_loss = 0.0
    rewards = []

    for _ in range(batch_size):
        prompt, expected = random.choice(unseen_pairs)
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

        resp = decode(gen)[len(prompt):]
        reward = compute_reward(resp, expected)
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
        s = evaluate(seen_pairs, 50)
        u = evaluate(unseen_pairs, 50)
        print(f"  Eval → Seen: {s*100:.1f}%, Unseen: {u*100:.1f}%")
        model.train()

torch.save(model.state_dict(), 'reinforce_ema.pt')
print()

fs = evaluate(seen_pairs)
fu = evaluate(unseen_pairs)
print("=" * 45)
print(f"  {'Metric':<20} {'Before':>10} {'After':>10}")
print("=" * 45)
print(f"  {'Seen accuracy':<20} {init_seen*100:>9.1f}% {fs*100:>9.1f}%")
print(f"  {'Unseen accuracy':<20} {init_unseen*100:>9.1f}% {fu*100:>9.1f}%")
print("=" * 45)
print(f"RESULT|REINFORCE(EMA)|{init_seen*100:.1f}|{fs*100:.1f}|{init_unseen*100:.1f}|{fu*100:.1f}")
import json; json.dump({"method":"REINFORCE(EMA)","seen_before":round(init_seen*100,1),"seen_after":round(fs*100,1),"unseen_before":round(init_unseen*100,1),"unseen_after":round(fu*100,1)}, open('metrics_ema.json','w'))
print("Saved reinforce_ema.pt")
