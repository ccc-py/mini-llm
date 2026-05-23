import random

print("=== 開始生成數學訓練資料 ===")

random.seed(42)

problems = []
for a in range(1, 31):
    for b in range(1, 31):
        problems.append(('+', a, b, a + b))
        if a > b:
            problems.append(('-', a, b, a - b))

random.shuffle(problems)
seed_problems = problems[:80]

pretrain_facts = []
finetune_qa = []

for op, a, b, r in seed_problems:
    fact = f"{a}{op}{b}={r}。"
    qa = f"<Q>{a}{op}{b}=？<A>{r}"
    pretrain_facts.append(fact)
    finetune_qa.append(qa)

extra_facts = [
    "加法是把兩個數字合併成一個數。",
    "減法是從一個數去掉另一個數。",
    "數學是重要的學科。",
]
pretrain_facts.extend(extra_facts)

PRETRAIN_TARGET = 100000
FINETUNE_TARGET = 30000

pretrain_text = ""
while len(pretrain_text) < PRETRAIN_TARGET:
    random.shuffle(pretrain_facts)
    pretrain_text += "\n".join(pretrain_facts) + "\n"

finetune_text = ""
while len(finetune_text) < FINETUNE_TARGET:
    random.shuffle(finetune_qa)
    finetune_text += "\n".join(finetune_qa) + "\n"

with open('pretrain.txt', 'w', encoding='utf-8') as f:
    f.write(pretrain_text[:PRETRAIN_TARGET])

with open('finetune.txt', 'w', encoding='utf-8') as f:
    f.write(finetune_text[:FINETUNE_TARGET])

print(f"生成完成: pretrain.txt ({PRETRAIN_TARGET} chars), finetune.txt ({FINETUNE_TARGET} chars)")
