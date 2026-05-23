import random

random.seed(42)

problems = []
for a in range(1, 10):
    for b in range(1, 10):
        problems.append((a, b, a + b))

random.shuffle(problems)

sft_pool = problems[:50]
rl_pool = problems[50:]

pretrain_facts = []
for a, b, r in problems:
    pretrain_facts.append(f"{a}+{b}={r}。")

sft_qa = [f"<Q>{a}+{b}=？<A>{r}" for a, b, r in sft_pool]
rl_qa   = [f"<Q>{a}+{b}=？<A>{r}" for a, b, r in rl_pool]

pretrain_facts += ["加法是把兩個數字合併成一個數。"]

PRETRAIN_TARGET = 100000
FINETUNE_TARGET = 30000

pretrain_text = ""
while len(pretrain_text) < PRETRAIN_TARGET:
    random.shuffle(pretrain_facts)
    pretrain_text += "\n".join(pretrain_facts) + "\n"

finetune_text = ""
while len(finetune_text) < FINETUNE_TARGET:
    random.shuffle(sft_qa)
    finetune_text += "\n".join(sft_qa) + "\n"

with open('pretrain.txt', 'w', encoding='utf-8') as f:
    f.write(pretrain_text[:PRETRAIN_TARGET])

with open('finetune.txt', 'w', encoding='utf-8') as f:
    f.write(finetune_text[:FINETUNE_TARGET])

with open('rl_unseen.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(rl_qa) + "\n")

print(f"SFT seen: {len(sft_pool)} problems, RL unseen: {len(rl_pool)} problems")
print(f"pretrain.txt ({PRETRAIN_TARGET}), finetune.txt ({FINETUNE_TARGET}), rl_unseen.txt written")
