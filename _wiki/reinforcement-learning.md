# Policy Gradient (REINFORCE) — 策略梯度強化學習

## 概述

**策略梯度（Policy Gradient）** 是強化學習（Reinforcement Learning, RL）中的一大類方法。不同於傳統的最佳化問題有明確的損失函數可以直接計算梯度，RL 的最佳化目標是最大化**累積獎勵（cumulative reward）**，而獎勵是由環境（或獎勵函數）在模型完成一系列動作後給出的 — 這個過程不可微分。策略梯度的核心想法是：用**對數機率 × 獎勵**作為替代損失函數，讓梯度穿過不可微分的取樣步驟，更新策略的參數。

mini-llm v4 實作了最基礎的策略梯度演算法 **REINFORCE（Monte Carlo Policy Gradient）**，在微調後的模型上進一步用 RL 提升數學問答的準確率。

## 動機：為什麼監督式微調還不夠？

v2 和 v3 的流程是：

```
預訓練（語言建模） → 監督式微調（Q&A 格式學習）
```

監督式微調（SFT）的問題在於：

1. **損失函數與最終目標不一致** — SFT 最小化 cross-entropy loss（預測每個 token 的機率），但真正關心的是答案是否正確。模型可能以 80% 的機率預測正確答案 token，但 cross-entropy 仍然較低 — 無法鼓勵模型對正確答案更有信心。

2. **沒有探索機制** — SFT 只從訓練資料中學習固定模式。如果訓練資料中 `1+1=2` 出現 100 次，模型學會 `1+1=2`，但它從未嘗試過其他可能性，也不知道「為什麼 2 是對的而 3 是錯的」。

3. **訓練資料分布固定** — SFT 的訓練資料在訓練開始前就固定了。模型無法針對自己的弱點（回答錯誤的題目）進行針對性訓練。

RL 的優勢：
- **直接最佳化任務指標** — 如果目標是「正確回答問題」，RL 的獎勵函數可以直接用正確性作為信號
- **自我探索** — 模型透過取樣產生各種回應，並從成功/失敗的經驗中學習
- **動態訓練** — 每次迭代都對當前策略的弱點進行改進

## 強化學習基本框架

RL 問題由四個核心元素構成：

| 元素 | 在語言模型中的對應 |
|------|-------------------|
| **Agent（智能體）** | 語言模型本身 |
| **Environment（環境）** | 上下文視窗 + 獎勵函數 |
| **Action（動作）** | 生成下一個 token |
| **State（狀態）** | 目前已生成的 token 序列 |
| **Reward（獎勵）** | 最終回答的正確性評分 |

語言模型的 RL 與傳統 RL 的關鍵差異：語言模型的「環境」非常簡單 — 模型生成一串 token 後，由獎勵函數一次性給出評分。這被稱為 **Episodic RL（情節式強化學習）**，每個 episode 就是一次完整的生成過程。

## REINFORCE 演算法

### 直觀理解

REINFORCE 的想法非常直觀：

- 讓模型對同一個問題產生多個回答
- 對每個回答計算獎勵分數（越高越好）
- 如果某個回答得到了好分數，就提高產生該回答的機率（增加 log probability）
- 如果得到了差的分數，就降低產生該回答的機率（減少 log probability）

用一句話概括：**多做好事，少做壞事**。

### 數學推導

策略梯度定理的核心公式：

```
∇θ J(θ) = Eτ~πθ [ Σt ∇θ log πθ(at|st) · R(τ) ]
```

其中：
- `θ` — 策略參數（模型權重）
- `J(θ)` — 期望獎勵（objective）
- `τ` — 一條完整軌跡（generated sequence）
- `πθ(at|st)` — 在狀態 st 下選擇動作 at 的機率
- `R(τ)` — 軌跡總獎勵
- `∇θ` — 對 θ 的梯度

實際實現時，公式轉換為損失函數：

```
L(θ) = - Σt log πθ(at|st) · R(τ)
```

注意負號 — 因為 PyTorch 的 optimizer 做的是**梯度下降**（最小化損失），而 RL 要做的是**梯度上升**（最大化獎勵），所以用負號將最大化問題轉換為最小化問題。

### mini-llm v4 的實作

```python
# 對每個 prompt，逐 token 生成並累積 log prob
log_probs = []
for _ in range(max_gen_len):
    ctx = torch.tensor(gen[-seq_len:], ...).unsqueeze(0)
    logits, _ = model(ctx)
    logits_last = logits[:, -1, :]          # (1, vocab_size)
    probs = F.softmax(logits_last, dim=-1)
    token = torch.multinomial(probs, 1)     # 從機率分布取樣
    log_prob = -F.cross_entropy(logits_last, token.view(-1))
    log_probs.append(log_prob)
    gen.append(token.item())

# 計算獎勵
reward = compute_reward(resp, expected)

# REINFORCE 損失
advantage = reward - baseline
loss = -torch.stack(log_probs).sum() * advantage
```

**關鍵設計細節**：

1. **`torch.multinomial` 保持可微分** — 雖然取樣操作本身不可微分，但 `-F.cross_entropy(logits, token)` 計算的是「取到的 token 的 log probability」，其梯度可以反向傳播到 logits，再傳播到整個模型。

2. **逐 token 累積 log prob** — REINFORCE 的理論要求是「整條軌跡的聯合機率的對數」。由於語言模型是自迴歸的：`log P(y|x) = Σt log P(yt | y<t, x)`，所以累積每個 token 的 log prob 就得到整條軌跡的 log prob。

3. **獎勵是整個序列的總分** — 語言模型的獎勵通常只在序列生成完畢後計算一次（最後一個 token 的正確性），而不是每個 token 單獨給獎勵。這是因為單個 token 的正確性很難定義 — 「2」這個 token 在 `1+1=` 後面是正確的，但在 `2+2=` 後面是錯誤的。因此我們將相同的總獎勵應用到每個 token 的 log prob 上。

## 獎勵函數設計

獎勵函數是 RL 訓練的「北極星」— 它定義了什麼是好的行為。如果獎勵函數設計不當，模型會學到完全不同的行為。

### v4 的獎勵函數

```python
def compute_reward(response, expected):
    r = 0.0
    rs = response.strip()
    es = expected.strip()
    if es in rs:                          # 答案完全正確
        r = 1.0
    elif rs and any(c.isdigit() for c in es) and any(c.isdigit() for c in rs[:5]):
        r = 0.3                            # 產生了數字但答案不對
    return r
```

**設計邏輯**：

- **精確匹配（+1.0）** — 期望答案出現在回應中。因為任務是數字問答（如 `3+5=？`，答案 `8`），這是一個明確的正確標準。

- **數字獎勵（+0.3）** — 如果模型輸出中包含數字（任何數字），給一個小的部分獎勵。這防止了「獎勵稀疏性（Reward Sparsity）」問題：如果模型從未得到任何正獎勵，梯度始終為零，模型永遠不會改進。數字獎勵像一個「形狀獎勵（Shaping Reward）」— 引導模型至少嘗試輸出數字。

### 獎勵函數設計原則

| 原則 | 說明 | v4 的實踐 |
|------|------|-----------|
| **明確性** | 獎勵應該清楚對應目標行為 | 精確匹配 = 1.0，明確無歧義 |
| **形狀（Shaping）** | 中間獎勵引導學習 | 數字獎勵（0.3）維持探索動力 |
| **稀疏性控制** | 避免全 0 或全 1 的獎勵 | 混合獎勵結構 |
| **對齊（Alignment）** | 獎勵引導的行為應該是真實想要的 | 正確回答數字問題 |

**一個常見錯誤**：如果只給精確匹配獎勵（0 或 1），模型在訓練初期幾乎從未得到正獎勵（因為還沒有學會正確回答），梯度幾乎為零，訓練陷入停滯。加上部分獎勵（0.3）後，即使模型胡亂輸出數字，也能得到一些梯度信號，逐漸學會先輸出數字、再輸出正確數字。

## 基線（Baseline）技術

### 為什麼需要基線？

REINFORCE 的梯度估計是：

```
∇θ J(θ) = E[ ∇θ log πθ(a|s) · R ]
```

這個估計的**變異數（variance）**非常大。同一個 prompt，模型的不同取樣可能得到截然不同的獎勵。想像以下情況：

- 對於問題 `3+5=？`，在當前的策略下：
  - 取樣 1：輸出 `8` → 獎勵 1.0 → 梯度更新增加 `8` 的機率
  - 取樣 2：輸出 `7` → 獎勵 0.0 → 梯度更新減少 `7` 的機率
  - 取樣 3：輸出 `9` → 獎勵 0.0 → 梯度更新減少 `9` 的機率

如果總獎勵平均值是 0.3，那麼對於取樣 1：
- R = 1.0 的梯度更新量遠大於理想的更新量（因為相對於 0.3 的平均值，1.0 偏離很大）

### 基線的數學

引入基線 `b` 後：

```
∇θ J(θ) = E[ ∇θ log πθ(a|s) · (R - b) ]
```

其中 `b` 是與動作無關的常數（或函數）。因為：

```
E[ ∇θ log πθ(a|s) · b ] = b · E[ ∇θ log πθ(a|s) ]
                         = b · Σa πθ(a|s) · ∇θ log πθ(a|s)
                         = b · Σa ∇θ πθ(a|s)
                         = b · ∇θ 1
                         = 0
```

基線的期望值為零，所以梯度估計仍然是**無偏的（unbiased）**，但變異數大大降低。

### 指數移動平均基線

v4 使用**指數移動平均（Exponential Moving Average, EMA）**作為基線：

```python
baseline = 0.1  # 初始值
for step in range(max_iters):
    # ... 訓練 ...
    avg_reward = sum(rewards) / len(rewards)
    baseline = 0.95 * baseline + 0.05 * avg_reward
```

`α = 0.95` 的 EMA 基線本質上是**獎勵的加權歷史平均**。它動態追蹤當前策略的平均表現水準：

- 如果獎勵總是 1.0：基線趨近 1.0，優勢（reward - baseline）趨近 0，停止更新（已經很好了）
- 如果獎勵從 0.3 提升到 0.8：基線緩慢追趕，優勢維持正值，持續推動改進
- 如果獎勵下降：優勢變負，引導模型遠離當前行為

### 基線的動態行為

以下是一次實際訓練中基線的演變（v4 運行記錄）：

```
Step   0 | Reward: 1.000 | Baseline: 0.145  (初始階段)
Step  30 | Reward: 0.912 | Baseline: 0.803  (基線快速上升)
Step 120 | Reward: 1.000 | Baseline: 0.992  (基線趨近最大)
Step 180 | Reward: 1.000 | Baseline: 0.991  (穩定在高點)
```

初始 `baseline = 0.1` 是保守估計。第一步就得到獎勵 1.0，優勢 `1.0 - 0.1 = 0.9`，梯度更新強烈。隨著訓練進行，基線追上實際獎勵水準，優勢變小，梯度更新變溫和 — 這正是我們想要的：模型已經學得很好時，避免過大的更新破壞已學到的知識。

## 探索與利用（Exploration vs Exploitation）

RL 中的核心權衡：

- **利用（Exploitation）** — 使用當前已知最好的策略來獲得高獎勵
- **探索（Exploration）** — 嘗試新的行為來發現可能更好的策略

若模型始終貪婪地選擇機率最高的 token（argmax），就永遠不會發現更好的策略 — 這是「固定策略陷阱」。

v4 在兩個層面實現探索：

### 1. 訓練時的取樣生成

```python
token = torch.multinomial(probs, 1)  # 從機率分布取樣
```

即使模型對 token `8` 的機率為 0.9，其他 token（如 `7`、`9`）仍有機會被選中。如果 `7` 偶然得到了獎勵 1.0（回答了不同的問題），這條經驗會讓模型對 `7` 的機率略微提升。

### 2. 評估時的取樣生成

評估時也使用 `multinomial`（而非 argmax），這使得評估結果反映了**真實的生成品質**，而不是最佳情況的表現。評估準確率在 96–100% 之間波動，反映了取樣隨機性。

### 探索的溫度控制

更進階的探索控制是**溫度參數（Temperature）**：

```
P(token) = exp(logit / T) / Σ exp(logit / T)
```

- T → 0：分布趨近 argmax（純利用）
- T = 1：標準 softmax
- T > 1：分布趨近均勻（純探索）

v4 使用 T = 1（標準 softmax），沒有明確的溫度調度。在實際應用中，可以在訓練初期使用較高的溫度鼓勵探索，後期降低溫度專注利用。

## On-Policy vs Off-Policy

| 特性 | On-Policy（REINFORCE） | Off-Policy（DQN） |
|------|----------------------|-------------------|
| 資料來源 | 當前策略取樣的資料 | 任意策略產生的資料都可使用 |
| 樣本效率 | 低（每個樣本只能用一次） | 高（可重複使用歷史資料） |
| 穩定性 | 較低（梯度變異大） | 較高（經驗回放緩衝區） |
| 實作難度 | 簡單 | 複雜 |

REINFORCE 是**On-Policy**演算法：每次更新後，策略改變了，之前取樣的資料就不能再用了。這意味著：

```
錯誤做法：
for step in 100:
    data = sample(pi_old)   # 收集一批資料
    for _ in 1000:
        update(data)        # 重複使用同一批資料（REINFORCE 不允許！）

正確做法：
for step in 100:
    data = sample(pi)       # 用當前策略取樣
    update(data)            # 只更新一次，立即丟棄資料
```

v4 每次迭代都重新取樣（`random.choice(qa_pairs)` + 重新生成），確保梯度永遠基於當前策略。這導致**樣本效率低**但**演算法正確**。

### PPO 如何解決這個問題？

PPO（Proximal Policy Optimization）通過**重要性取樣（Importance Sampling）**和**裁剪（Clipping）**允許重複使用舊資料：

```
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clipped_ratio * A)
```

裁剪防止新策略與舊策略差異過大，從而允許在舊資料上進行多次更新。這是現代 RLHF（如 ChatGPT 的訓練）的標準做法。

## 訓練動態分析

### 損失與獎勵的關係

REINFORCE 的「損失」與監督學習的損失意義不同：

- 監督學習：損失越低越好（越低表示擬合越好）
- REINFORCE：損失**可能為負**！因為 `loss = -log_prob * advantage`

當 `advantage > 0`（表現比基線好）時，損失為負，梯度下降會進一步增加 log prob（提高好行為的機率）。當 `advantage < 0` 時，損失為正，梯度下降會降低 log prob。

v4 實際運行記錄：

```
Step   0 | Loss:  0.0243 | Reward: 1.000 | Advantage:  0.855
Step  30 | Loss: -0.5354 | Reward: 0.912 | Advantage:  0.109
Step  60 | Loss:  0.0006 | Reward: 1.000 | Advantage:  0.042
Step 120 | Loss:  0.0001 | Reward: 1.000 | Advantage:  0.008
```

隨著訓練進行，loss 從正變負再趨近零 — 這反映了 advantage 逐漸縮小（基線追上獎勵），參數更新越來越小，最終收斂。

### 準確率提升軌跡

v4 的一次實際運行：

```
初始 Eval Accuracy: 約 98%
Step  50 | Eval Accuracy: 98%
Step 100 | Eval Accuracy: 98%
Step 150 | Eval Accuracy: 100%
Step 200 | Eval Accuracy: 96%
Step 250 | Eval Accuracy: 100%
FINAL   | Eval Accuracy: 99.0%
```

觀察：

1. **微調後的模型已經很好** — 因為 SFT 階段的準確率已經達到了 ~98%，RL 階段的改進空間有限（從 98% 到 99%）。

2. **波動是正常的** — RL 訓練的準確率會波動（96% → 100% → 96% → 100%），因為 `multinomial` 取樣引入了隨機性，而且評估樣本只有 50 題，統計誤差較大。

3. **100% 準確率可達成** — 50 題評估中達到 100% 說明模型確實學會了正確回答模式。最終 200 題評估的 99% 更穩健。

4. **RL 的作用不是大幅提升準確率** — 當 SFT 已經很好時，RL 的主要貢獻是讓模型對正確答案更有信心（增加正確 token 的機率）、削減錯誤 token 的機率，使生成更穩定。

## 獎勵駭取（Reward Hacking）

RL 中的一個經典問題：模型可能會找到獎勵函數的漏洞，最大化獎勵但沒有學到預期行為。

假設我們設計了一個獎勵函數，錯誤地獎勵了「輸出更長的回應」：

```python
def bad_reward(response, expected):
    return 1.0 if expected in response else 0.5  # 即使錯誤也給 0.5
```

模型會學到：無論正確與否，只要輸出內容就有獎勵。最終模型可能選擇永遠輸出同一個數字（如 `111111...`），因為 `1` 在許多答案中出現。

v4 的獎勵函數設計避免了這個陷阱：

- 正確匹配（1.0）與部分匹配（0.3）的差距夠大（3.3 倍），強烈鼓勵正確行為
- 部分匹配（0.3）只在輸出數字時給出，不會獎勵胡亂輸出
- 如果完全沒有數字，獎勵為 0 — 不獎勵任何形式的無意義輸出

## GRPO：比 REINFORCE 更現代的替代方案

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek-R1 等現代模型使用的 RL 演算法，與 REINFORCE 的主要差異：

| 層面 | REINFORCE（v4） | GRPO |
|------|----------------|------|
| 基線 | 歷史 EMA，單一標量 | 同 prompt 的 group 平均獎勵 |
| 群體 | 無 | 每條 prompt 生成 G 個回覆 |
| 優勢計算 | R - baseline | (R_i - mean(R_group)) / std(R_group) |
| 裁剪 | 無 | 有（防止過大更新） |

GRPO 的優勢定義：
```
advantage_i = (reward_i - mean(reward_group)) / std(reward_group)
```

這等於在同一個 prompt 的多個取樣之間做「相對比較」。如果群體回覆分數為 [0.9, 0.3, 0.1, 0.1]：
- 0.9 → 正優勢（比其他好）
- 0.3 → 接近零（普通）
- 0.1 → 負優勢（比其他差）

GRPO 的優點：
- 不需要額外的基線模型或變數
- 群體內比較自動消除了 prompt 難度的影響（容易的 prompt 所有人的獎勵都高，但誰更好還是取決於相對差異）
- 標準化讓優勢的尺度在不同 prompt 之間保持一致

v4 可以簡單擴展為 GRPO — 在 `batch_size` 內對同一個 prompt 生成 G 個回覆，計算群體內標準化優勢。

## 從 REINFORCE 到 RLHF

ChatGPT 的訓練管道包含 RLHF（Reinforcement Learning from Human Feedback）：

```
預訓練 → SFT（監督微調）→ RLHF（RL + 人類偏好）
```

RLHF 在 REINFORCE 的基礎上增加了兩層複雜度：

### 1. 獎勵模型（Reward Model）

不是用規則函數計算獎勵，而是訓練一個神經網路來**預測人類偏好**：

```
人類標註員對 SFT 模型的多個輸出排序
    ↓
訓練獎勵模型 RM（通常與 SFT 模型大小相同或更小）
    ↓
RM 對任何給定的 (prompt, response) 輸出一個標量分數
```

### 2. KL 散度懲罰

RL 訓練容易讓模型**偏離原始語言能力**（只追求高分但語言崩壞）。RLHF 加入 KL 散度懲罰：

```
reward_total = RM_score - β · KL(πRL || πSFT)
```

- `RM_score` — 獎勵模型的評分（人類偏好）
- `KL(πRL || πSFT)` — 目前模型與 SFT 模型的分布差異
- `β` — 權重係數

這防止了模型在追求高分時生成語法錯誤或重複的內容。REINFORCE 對語言崩壞沒有自然防護，但 v4 的任務（短數字問答）相對安全，因為輸出空間很小（幾個數字符號），不容易偏離。

### 3. PPO 的裁剪機制

RLHF 通常使用 PPO 而非原始 REINFORCE，因為 PPO 的裁剪（clipping）提供了一層保護：每次更新不會讓策略變化太大，進一步防止語言崩壞。

## 進一步的 RL 應用：GRPO 與 R1 風格訓練

DeepSeek-R1 展示了 **Group Relative Policy Optimization (GRPO)** 在推理任務上的強大效果。與 RLHF 不同：

1. **無需獎勵模型**：GRPO 的獎勵來自規則（如數學答案的正確性）或程式碼執行結果
2. **群體比較**：一個 prompt 生成多個回覆，在群體內做相對比較
3. **推理鏈擴展**：模型學會在回答問題時生成更長的推理鏈（Chain-of-Thought）

v4 的獎勵函數（規則式、可自動計算）已經具備了 GRPO 風格的獎勵條件。如果要擴展：

- 一個 prompt 生成 G=8 個回覆
- 計算群體內的相對優勢
- 不需要維護 EMA 基線

這對於明確有正確答案的任務（數學、程式碼）特別有效。

## mini-llm v4 的完整 RL 流程

```
          gen_data.py
              ↓
          pretrain.py
              ↓
          finetune.py
              ↓
    ┌───── reinforce.py ─────┐
    │   load finetune.pt      │
    │   for step in 300:      │
    │     for _ in batch(8):  │
    │       sample prompt     │
    │       generate response │  ← 可微分（torch.multinomial）
    │       compute reward    │  ← 規則式獎勵函數
    │       compute advantage │  ← R - EMA baseline
    │       policy gradient   │  ← -log_prob * advantage
    │     update model        │
    │   save reinforce.pt     │
    └─────────────────────────┘
```

與監督式微調的流程對比：

```
監督式微調（v2/v3）:
  固定資料集 → forward → cross-entropy(labels, logits) → backward

RL 微調（v4）:
  動態取樣 → forward → multinomial 取樣 → 計算獎勵 → policy gradient → backward
```

根本差異：SFT 直接告訴模型「正確答案是什麼」，而 RL 只告訴模型「這個回答好不好」，模型需要自己想出什麼樣的回答能得到高分。

## 為什麼 REINFORCE 對這個任務有效？

v4 選擇 REINFORCE 而非更複雜的 RL 演算法的考量：

| 因素 | REINFORCE 的優勢 |
|------|-----------------|
| 輸出空間 | 短（30 個 token），episode 變異數低 |
| 獎勵信號 | 明確（數字正確性），不需要獎勵模型 |
| 計算資源 | 不需要價值網路或 critic，參數少 |
| 實作複雜度 | 約 50 行程式碼，易於理解和修改 |
| 收斂穩定性 | 已經過 SFT，初始策略品質高，不需要複雜約束 |

對於需要長篇生成或主觀評分的任務（如創意寫作、對話），REINFORCE 的變異數過大，需要 PPO 或 GRPO 的裁剪機制。但對於 v4 的目標（短數學問答），REINFORCE 已經足夠。

## 限制與注意事項

### 1. 獎勵函數的覆蓋範圍

v4 的獎勵函數只檢查**單個數字**的正確性。對於多步推理或需要多個數字的任務（如 `(3+5)×2=？`），需要更複雜的獎勵設計 — 可能逐步計算每個中間結果的正確性，或使用程式執行結果作為獎勵訊號（programmatic reward）。

### 2. 冷啟動問題

如果「從零開始 RL」（沒有 SFT 階段），模型產生正確數字的機率接近於 0（在 42 字元的詞表中隨機取樣到正確答案的機率約為 1/42），REINFORCE 幾乎無法啟動。解決方案：

- 先用 SFT 讓模型具備基礎能力（v4 的做法）
- 使用**行為克隆（Behavioral Cloning）**初始化策略
- 使用**獎勵形狀（Reward Shaping）**提供中間獎勵信號

### 3. 過度最佳化

REINFORCE 在長期訓練下可能導致「過度最佳化」：模型學會完美回答訓練中出現過的 prompt，但對未見過的 prompt 泛化能力下降。v4 的 300 步訓練在小型模型上尚可接受，但更長時間的訓練需要引入正則化機制。

### 4. On-Policy 的樣本效率

v4 的 batch_size=8、max_iters=300，總共只產生了 8×300 = 2,400 條回應 — 以 RL 標準來說非常少（ChatGPT 的 RLHF 使用數百萬條人類偏好標註）。增加訓練步數和 batch_size 可能進一步提升效果，但受限於玩具專案的計算預算。

## 延伸閱讀

- REINFORCE 原始論文: Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (Machine Learning, 1992)
- 策略梯度定理: Sutton et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation" (NeurIPS 2000)
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv 2017)
- GRPO: DeepSeek-R1: "Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv 2025)
- InstructGPT: Ouyang et al., "Training language models to follow instructions with human feedback" (NeurIPS 2022)
- 獎勵最佳化與 RLHF 綜述: Lambert et al., "A Survey of Reinforcement Learning from Human Feedback" (arXiv 2024)
