import json
import random
import torch
import torch.nn.functional as F
from model import ModernLanguageModel

batch_size = 8
seq_len = 64
max_iters = 300
learning_rate = 1e-5
max_gen_len = 30
eval_every = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
vocab_size = vocab['vocab_size']
stoi = vocab['stoi']
itos = {int(k): v for k, v in vocab['itos'].items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

with open('finetune.txt', 'r', encoding='utf-8') as f:
    lines = f.read().strip().split('\n')
qa_pairs = []
for line in lines:
    line = line.strip()
    if '<Q>' in line and '<A>' in line:
        q, a = line.split('<A>', 1)
        qa_pairs.append((q + '<A>', a.strip()))

if not qa_pairs:
    raise ValueError("No QA pairs found in finetune.txt")
print(f"Loaded {len(qa_pairs)} QA pairs")

model = ModernLanguageModel(vocab_size=vocab_size, seq_len=seq_len, device=device).to(device)
model.load_state_dict(torch.load('finetune.pt', map_location=device))
print("Loaded finetune.pt")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def compute_reward(response, expected):
    r = 0.0
    rs = response.strip()
    es = expected.strip()
    if es in rs:
        r = 1.0
    elif rs and any(c.isdigit() for c in es) and any(c.isdigit() for c in rs[:5]):
        r = 0.3
    return r

@torch.no_grad()
def evaluate(num=100):
    model.eval()
    correct = 0
    for i in range(min(num, len(qa_pairs))):
        prompt, expected = qa_pairs[i]
        prompt_ids = encode(prompt)
        gen = prompt_ids.copy()
        for _ in range(max_gen_len):
            ctx = torch.tensor(gen[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(ctx)
            token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            gen.append(token.item())
            if itos.get(token.item(), '') == '\n':
                break
        resp = decode(gen)[len(prompt):].strip()
        if expected.strip() in resp:
            correct += 1
    return correct / (min(num, len(qa_pairs)))

baseline = 0.1
for step in range(max_iters):
    model.train()
    total_loss = 0.0
    rewards = []

    for _ in range(batch_size):
        prompt, expected = random.choice(qa_pairs)
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

    if step % 10 == 0 or step == max_iters - 1:
        print(f"Reinforce Step {step:4d} | Loss: {avg_loss.item():.4f} | Reward: {avg_reward:.3f} | Baseline: {baseline:.3f}")

    if eval_every > 0 and (step + 1) % eval_every == 0:
        acc = evaluate(50)
        print(f"  Eval Accuracy: {acc*100:.1f}%")
        model.train()

torch.save(model.state_dict(), 'reinforce.pt')
print("\nSaved reinforce.pt")

final_acc = evaluate(200)
print(f"Final Accuracy: {final_acc*100:.1f}%")
print("RL training complete!")
