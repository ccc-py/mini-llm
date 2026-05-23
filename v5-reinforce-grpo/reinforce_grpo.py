import json
import random
import argparse
import torch
import torch.nn.functional as F
from model import ModernLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--beta', type=float, default=0.04)
parser.add_argument('--group', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-5)
args = parser.parse_args()

G = args.group
seq_len = 64
max_iters = args.steps
learning_rate = args.lr
max_gen_len = 20
beta_kl = args.beta
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
print(f"GRPO: steps={max_iters}, G={G}, β={beta_kl}, lr={learning_rate}")

model = ModernLanguageModel(vocab_size=vocab_size, seq_len=seq_len, device=device).to(device)
model.load_state_dict(torch.load('finetune.pt', map_location=device))
ref_model = ModernLanguageModel(vocab_size=vocab_size, seq_len=seq_len, device=device).to(device)
ref_model.load_state_dict(torch.load('finetune.pt', map_location=device))
ref_model.eval()
print("Loaded finetune.pt (policy + reference)")

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
print(f"Before RL → Seen: {init_seen*100:.1f}%, Unseen: {init_unseen*100:.1f}%")
print()

for step in range(max_iters):
    model.train()
    prompt, expected = random.choice(unseen_pairs)
    prompt_ids = encode(prompt)

    group_log_probs = []
    group_kls = []
    group_rewards = []

    for _ in range(G):
        gen = prompt_ids.copy()
        traj_log_probs = []
        traj_kl = 0.0

        for _ in range(max_gen_len):
            ctx = torch.tensor(gen[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits_theta, _ = model(ctx)
            logits_last = logits_theta[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            token = torch.multinomial(probs, 1)
            log_prob = -F.cross_entropy(logits_last, token.view(-1))
            traj_log_probs.append(log_prob)

            with torch.no_grad():
                logits_ref, _ = ref_model(ctx)
            lp_theta = F.log_softmax(logits_last, dim=-1)
            p_ref = F.softmax(logits_ref[:, -1, :], dim=-1)
            traj_kl = traj_kl + (p_ref * (p_ref.log() - lp_theta)).sum()

            gen.append(token.item())
            if itos.get(token.item(), '') == '\n':
                break

        resp = decode(gen)[len(prompt):]
        reward = compute_reward(resp, expected)
        group_log_probs.append(torch.stack(traj_log_probs))
        group_kls.append(traj_kl)
        group_rewards.append(reward)

    # Group advantage
    rewards_t = torch.tensor(group_rewards, device=device, dtype=torch.float)
    adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    # Loss
    total_pg = 0.0
    total_kl = 0.0
    for i in range(G):
        total_pg = total_pg - group_log_probs[i].sum() * adv[i]
        total_kl = total_kl + group_kls[i]
    avg_pg = total_pg / G
    avg_kl = total_kl / G
    loss = avg_pg + beta_kl * avg_kl

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    avg_r = sum(group_rewards) / G
    if step % 25 == 0 or step == max_iters - 1:
        print(f"Step {step:4d} | Loss: {loss.item():+.4f} | Reward: {avg_r:.3f} | KL: {avg_kl.item():.4f}")

    if (step + 1) % eval_every == 0:
        s = evaluate(seen_pairs, 50)
        u = evaluate(unseen_pairs, 50)
        print(f"  Eval → Seen: {s*100:.1f}%, Unseen: {u*100:.1f}%")
        model.train()

torch.save(model.state_dict(), 'reinforce_grpo.pt')
print()

fs = evaluate(seen_pairs)
fu = evaluate(unseen_pairs)
import json as _j
_j.dump({"method":"GRPO","seen_before":round(init_seen*100,1),"seen_after":round(fs*100,1),"unseen_before":round(init_unseen*100,1),"unseen_after":round(fu*100,1)}, open('metrics_grpo.json','w'))
print("=" * 45)
print(f"  {'Metric':<20} {'Before':>10} {'After':>10}")
print("=" * 45)
print(f"  {'Seen accuracy':<20} {init_seen*100:>9.1f}% {fs*100:>9.1f}%")
print(f"  {'Unseen accuracy':<20} {init_unseen*100:>9.1f}% {fu*100:>9.1f}%")
print("=" * 45)
print("Saved reinforce_grpo.pt")
