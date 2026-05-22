import torch
import pickle
from model import ModernLanguageModel

# 參數設定
batch_size = 32
seq_len = 64
max_iters = 500 #3000
learning_rate = 5e-4 # Pretrain 學習率通常較高
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 載入詞表與資料
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = vocab['vocab_size']

data = torch.load('pretrain_data.pt')
print(f"載入 Pretrain 資料，長度: {len(data)}")

def get_batch(data):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# 2. 初始化模型
model = ModernLanguageModel(vocab_size=vocab_size, seq_len=seq_len, device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"使用硬體: {device} | 開始 Pre-training...")

# 3. 訓練迴圈
for iter in range(max_iters):
    model.train()
    xb, yb = get_batch(data)
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if iter % 100 == 0 or iter == max_iters - 1:
        print(f"Pretrain Step {iter:4d} | Loss: {loss.item():.4f}")

# 4. 儲存模型權重
torch.save(model.state_dict(), 'pretrain.pt')
print("預訓練完成！模型已儲存為 pretrain.pt")