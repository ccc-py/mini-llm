import pickle
import torch
import os
from datasets import load_dataset

# ==========================================
# 1. 準備 Pre-train 資料 (官方最新版中文維基百科)
# ==========================================
print("1. 從 Hugging Face 下載 Pre-train 語料 (中文維基百科前 1000 篇)...")
# 改用官方維護的 wikimedia/wikipedia，並使用 Parquet 格式 (20231101.zh)
wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train[:1000]")

# 將維基百科的 text 欄位全部串接起來
pretrain_text = "".join(wiki_dataset['text'])

# 為了避免一般電腦記憶體爆掉，我們這裡限制最多取前 100 萬個字元 (約 2MB 的純文字)
pretrain_text = pretrain_text[:1000000] 
print(f"-> Pre-train 資料準備完畢，總字元數: {len(pretrain_text)}")


# ==========================================
# 2. 準備 Fine-tune 資料 (中文 Alpaca 問答集)
# ==========================================
print("\n2. 從 Hugging Face 下載 Fine-tune 語料 (Alpaca-zh 前 1000 筆)...")
# 移除 trust_remote_code=True，因為這個資料集已經是標準 Parquet 格式
alpaca_dataset = load_dataset("shibing624/alpaca-zh", split="train[:1000]")

finetune_text = ""
for row in alpaca_dataset:
    # Alpaca 資料集的結構通常包含 instruction (指令), input (補充輸入), output (回答)
    instruction = row.get('instruction', '')
    inp = row.get('input', '')
    output = row.get('output', '')
    
    # 組合問題
    question = instruction + ("\n" + inp if inp else "")
    
    # 轉換成我們的 <Q> 和 <A> 格式
    finetune_text += f"<Q>{question}<A>{output}\n"

print(f"-> Fine-tune 資料準備完畢，總字元數: {len(finetune_text)}")


# ==========================================
# 3. 建立共用 Tokenizer (詞彙表)
# ==========================================
print("\n3. 建立共用詞表 (Vocabulary)...")
all_chars = sorted(list(set(pretrain_text + finetune_text)))
vocab_size = len(all_chars)
print(f"-> 總詞表大小 (不重複字元數): {vocab_size}")

stoi = {ch: i for i, ch in enumerate(all_chars)}
itos = {i: ch for i, ch in enumerate(all_chars)}

# 儲存 Tokenizer (字典)
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)


# ==========================================
# 4. 轉換為 Tensor 並儲存
# ==========================================
print("\n4. 將文字轉換為 Tensor 並儲存至硬碟...")
encode = lambda s: [stoi[c] for c in s]

torch.save(torch.tensor(encode(pretrain_text), dtype=torch.long), 'pretrain_data.pt')
torch.save(torch.tensor(encode(finetune_text), dtype=torch.long), 'finetune_data.pt')

print("\n✅ 資料準備完成！已產生: vocab.pkl, pretrain_data.pt, finetune_data.pt")