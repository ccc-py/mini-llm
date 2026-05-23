# Reward Model — 獎勵模型

## 概述

**獎勵模型（Reward Model, RM）** 是一個神經網路，用於將 (prompt, response) 對映射到一個標量分數，代表該回應的「品質」。在強化學習（RL）的上下文中，獎勵模型取代了手工設計的規則式獎勵函數，使得 RL 可以應用於**主觀或複雜的評分任務**（如對話品質、文章相關性、程式碼可讀性），這些任務無法用簡單的規則自動評分。

mini-llm v4 實作了一個基於 Transformer 的獎勵模型（在 `train_reward_model.py` 和 `reinforce_rm.py` 中），用於數學問答任務的 RL 訓練。與規則式獎勵相比，獎勵模型的結果顯著較差（UNSEEN 準確率從 93.5% 降至 9.7%），這是一個有教育意義的失敗案例，說明了獎勵模型設計中的關鍵陷阱。

## 為什麼需要獎勵模型？

### 規則式獎勵的局限性

規則式獎勵函數在以下情況無法使用：

1. **主觀評分** — 對話的「禮貌程度」、摘要的「資訊完整性」、創意寫作的「流暢度」無法用程式碼自動評分
2. **複雜推理** — 數學證明題的正確性無法用字串匹配驗證（需要理解完整的推理過程）
3. **人類偏好** — 人類對「有用的回答」的定義是上下文相關的，無法用固定規則捕捉

### 獎勵模型的角色

獎勵模型的核心思想是：**用資料驅動的方法學習獎勵函數**。

```
人類標註 → (prompt, response_1, response_2, 偏好) 資料
     ↓
訓練獎勵模型 RM（回歸或配對學習）
     ↓
RM 對任何新回應輸出分數 → 作為 RL 的獎勵信號
```

這使得 RL 可以應用於：
- RLHF（Reinforcement Learning from Human Feedback）— ChatGPT 等模型的標準訓練管道
- 複雜任務的自動評分 — 如程式碼執行結果、數學推理鏈正確性
- 多維度評分 — 同時考慮安全性、有用性、真實性等多個方面

## 獎勵模型的架構

### 標準架構

獎勵模型通常與語言模型共享架構，只修改輸出層：

```
輸入: (prompt_tokens + response_tokens)
     ↓
共享 Transformer 主幹（與語言模型相同的權重初始化）
     ↓
[CLS] token 或最後 token 的隱藏狀態
     ↓
線性投影層 d_model → 1
     ↓
輸出: 標量分數（越高越好）
```

### v4 的 RM 實作

v4 的 RM 架構與策略模型共用同一組 Transformer 層，但替換了輸出頭：

```python
class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, seq_len=64):
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.score_head = nn.Linear(d_model, 1, bias=False)  # 取代 LM head
        self.freqs_cis = precompute_freqs_cis(...)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.norm(x)
        return self.score_head(x[:, -1, :]).squeeze(-1)  # 使用最後 token
```

**關鍵設計決策**：

1. **從 pretrain.pt 初始化** — RM 的 embed 層和 Transformer 層從預訓練模型載入權重，而不是從零開始訓練。這利用了預訓練模型已經學到的語言知識。

2. **最後 token 的隱藏狀態作為聚合表示** — RM 將整個序列的最後一個 token 位置（倒數第二層歸一化後）的 d_model 維隱藏狀態投影到 1 維標量。這與 GPT 系列 reward model 的常見做法一致。

3. **與 LM 的對比**：

| 層面 | 語言模型（策略） | 獎勵模型 |
|------|----------------|---------|
| 輸出層 | Linear(d_model, vocab_size) | Linear(d_model, 1) |
| 輸出含義 | 詞表上的機率分布 | 標量分數 |
| 損失函數 | Cross-entropy | MSE（回歸） |
| 訓練資料 | (prompt, response) pairs | (full_text, score) pairs |

## RM 的訓練資料

### 資料生成策略

RM 需要 `(text, score)` 配對資料進行回歸訓練。由於 v4 的任務（加法問答）有明確的正確答案，可以用規則自動生成訓練資料：

```python
training_data = []
for a, b, correct in problems:
    prompt = f"<Q>{a}+{b}=？<A>"

    # 正確答案 → 分數 1.0
    training_data.append((prompt + str(correct), 1.0))

    # 附近答案 → 部分分數（距離越遠分數越低）
    for delta in [1, 2, -1, -2]:
        wrong = correct + delta
        if wrong >= 2:
            score = max(0.0, 0.5 - 0.1 * abs(delta))
            training_data.append((prompt + str(wrong), score))

    # 明顯錯誤 → 分數 0.0
    for wrong in [correct // 2, correct * 2, 99]:
        training_data.append((prompt + str(wrong), 0.0))
```

這個策略的**分數設計邏輯**：

- **正確答案（1.0）** — 與規則式獎勵一致
- **附近錯誤（0.4~0.3）** — 接近正確答案的錯誤（如 3+5 輸出 7 或 9）得到部分分數，因為這些錯誤「接近正確」
- **完全錯誤（0.0）** — 明顯的隨機答案沒有分數

RM 的訓練資料量為：81 題 × (1 正確 + 4 附近 + 3 明顯錯誤) = 648 條訓練樣本。

### 與偏好資料的關係

在標準 RLHF 中，RM 使用**偏好資料**訓練：

```
人類比較 (response_A, response_B) → 偏好選擇
    ↓
Bradley-Terry 模型: P(A > B) = σ(RM(A) - RM(B))
    ↓
最大化人類偏好排名的一致性
```

v4 使用**回歸式評分**而非偏好比較，因為：
- 數學問答的分數可以明確量化（不像「哪個回答更有用」是主觀的）
- 標籤資料可以完全自動生成（不需要人類標註員）
- 回歸損失（MSE）訓練比偏好損失更穩定

## RM 訓練流程

### 初始化與訓練

```python
# 從 pretrain.pt 載入權重
rm = RewardModel(vocab_size=vocab_size)
pt = torch.load('pretrain.pt')
rm.tok_emb.weight.copy_(pt['tok_emb.weight'])
rm.layers.load_state_dict({k[7:]: v for k, v in pt.items() if k.startswith('layers.')})
rm.norm.weight.copy_(pt['norm.weight'])

# MSE 回歸訓練
optimizer = torch.optim.AdamW(rm.parameters(), lr=1e-4)
for epoch in range(15):
    for text, score in training_data:
        ids = torch.tensor(encode(text)).unsqueeze(0)
        pred = rm(ids)
        loss = F.mse_loss(pred.squeeze(), torch.tensor(score))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### RM 的測試結果

訓練完成後，RM 在測試樣本上的表現：

```
<Q>3+5=？<A>8     expected=1.00  predicted=0.607  (低估)
<Q>3+5=？<A>7     expected=0.40  predicted=0.535  (高估)
<Q>3+5=？<A>3     expected=0.00  predicted=-0.065 (合理)
<Q>3+5=？<A>99    expected=0.00  predicted=-0.028 (合理)
```

RM 的預測與期望分數大致一致，但正確答案（得分 0.607）與附近錯誤（得分 0.535）之間的區分度不足。這對後續 RL 訓練有重大影響。

## RM 在 RL 中的使用

v4 的 `reinforce_rm.py` 使用訓練好的 RM 代替規則式獎勵函數：

```python
# 載入 RM
rm = RewardModel(vocab_size=vocab_size)
rm.load_state_dict(torch.load('reward_model.pt'))
rm.eval()

# 在 REINFORCE 循環中使用 RM 計算獎勵
for _ in range(batch_size):
    # ... 生成回覆 ...
    full_text = decode(gen)
    full_ids = torch.tensor(encode(full_text)).unsqueeze(0)
    with torch.no_grad():
        reward = rm(full_ids).item()   # RM 給出分數
    reward = max(0.0, reward)          # 截斷負分數
    advantage = reward - baseline
    loss = -torch.stack(log_probs).sum() * advantage
```

### RM 與規則式獎勵的動態對比

在實際訓練中，RM 產生的獎勵分布與規則式獎勵有顯著差異：

| 特徵 | 規則式獎勵 | RM 獎勵 |
|------|-----------|---------|
| 獎勵範圍 | {0.0, 1.0}（二元） | 連續值（如 0.2~0.6） |
| 正確答案 | 1.0 | ~0.6 |
| 錯誤答案 | 0.0 | ~0.3~0.4 |
| 正確 vs 錯誤差距 | 1.0（巨大） | < 0.3（微小） |

RM 的壓縮效應（正確和錯誤的分數差距很小）是後續 RL 訓練失敗的根本原因。

## 為什麼 RM 在 v4 中失敗？

### 現象

v4 使用 RM 做 REINFORCE 的結果（三次運行的平均）：

```
指標         Before    After
SEEN 準確率   97.0%    38.0%  ← 大幅下降
UNSEEN 準確率  67.7%     9.7%  ← 幾乎崩潰
```

這比不使用任何 RL 的結果（隨機策略）還要差。對比規則式獎勵的 REINFORCE(EMA)：

```
SEEN 準確率    98.0%    95.0%  ← 幾乎無損
UNSEEN 準確率   67.7%    93.5%  ← 大幅提升
```

### 根本原因分析

#### 1. 獎勵解析度不足（Resolution Collapse）

RM 的輸出被壓縮到一個狹窄的範圍內（約 0.2~0.6），將正確答案（目標 1.0）和錯誤答案（目標 0.0）的區分度壓縮到大約 0.3 以內。

這導致了**優勢函數的信噪比極低**：

```
規則式獎勵:
  correct: reward=1.0, advantage=1.0-baseline ≈ 0.9  (強烈正信號)
  wrong:   reward=0.0, advantage=0.0-baseline ≈ -0.4  (強烈負信號)
  信噪比 ≈ 2.3 (正負信號差異大，方向明確)

RM 獎勵:
  correct: reward≈0.60, advantage≈0.60-0.42=0.18  (微弱正信號)
  wrong:   reward≈0.35, advantage≈0.35-0.42=-0.07  (微弱負信號)
  信噪比 ≈ 0.25 (正負信號接近零，梯度幾乎為隨機雜訊)
```

#### 2. 基線漂移（Baseline Drift）

RM 獎勵的 EMA 基線隨著訓練進行而變化：

```
Step   0 | Reward: 0.394 | Baseline: 0.115 | Advantage: 0.279
Step 100 | Reward: 0.399 | Baseline: 0.401 | Advantage: -0.002
Step 200 | Reward: 0.411 | Baseline: 0.393 | Advantage: 0.018
Step 300 | Reward: 0.393 | Baseline: 0.421 | Advantage: -0.028
Step 400 | Reward: 0.454 | Baseline: 0.416 | Advantage: 0.038
Step 499 | Reward: 0.396 | Baseline: 0.429 | Advantage: -0.033
```

RM 獎勵在整個訓練過程中不斷在 0.39~0.45 之間隨機波動，但 **獎勵本身的波動幅度大於正確與錯誤之間的差距**。換句話說：

- RM 對同一 prompt 的不同取樣，分數波動可能是 ±0.1
- 但正確與錯誤回答之間的平均差距只有 <0.3
- 隨機雜訊占信號的 30% 以上

這使得優勢函數的符號（正負）主要由雜訊決定，而非真正的好壞。

#### 3. 策略崩潰（Policy Collapse）

隨著訓練進行，優勢函數逐漸退化到接近零的隨機雜訊，梯度更新失去了方向：

```
Step   0: Adv: 0.279  (還有些信號)
Step 100: Adv:-0.002  (完全雜訊)
Step 200: Adv: 0.018  (繼續雜訊)
...
Step 499: Adv:-0.033  (雜訊到最後)
```

在缺乏有效梯度信號的情況下，模型開始退化：

1. 梯度幾乎隨機，每次更新都在無目的地改變策略
2. 由於策略本身（SFT 後）已經相當好（~97%），任何隨機擾動都只會降低表現
3. 隨著步數增加，模型逐漸偏離了 SFT 的優良初始點
4. SEEN 和 UNSEEN 準確率同步下降

### 與業界 RLHF 的對比

為什麼 ChatGPT 的 RM 有效而 v4 的 RM 無效？

| 因素 | ChatGPT RLHF | v4 RM |
|------|-------------|-------|
| 資料量 | 數十萬人類偏好標註 | 648 條自動生成 |
| 評分粒度 | 比較 2 個回應的相對偏好 | 絕對分數回歸 |
| RM 與策略關係 | RM 訓練資料包含 SFT 模型的輸出分布 | RM 訓練資料全為正確格式的回應 |
| 訓練分布匹配 | RM 在 RL 過程中使用時，輸入分布與訓練時不同（distribution shift） | 同左，但更嚴重 |
| 獎勵範圍 | 經過 KL 正則化控制的範圍 | 無正則化，範圍收縮 |

其中**訓練分布不匹配（Distribution Shift）** 是最嚴重的問題：

1. RM 的訓練資料全部是「正確格式」的回應（完整的 `<Q>...<A>...` 格式）
2. 在 RL 訓練中，模型會生成各種各樣的輸出（部分生成、亂碼、重複）
3. RM 從未見過這些分布外的輸入，其預測完全不可靠

## 改進方向

### 1. 擴大獎勵差距

提高 RM 對正確與錯誤回答的區分度：

```python
# 更積極的訓練資料設計
training_data.append((prompt + str(correct), 1.0))  # 正確 → 1.0
training_data.append((prompt + str(wrong), 0.0))    # 錯誤 → 0.0（不加中間分數）
```

取消中間分數（0.3~0.4）可以迫使 RM 產生更二元的輸出。但這可能導致 RM 過度自信，且失去對「接近正確」的區分能力。

### 2. 分布匹配訓練

在 RL 訓練過程中，定期使用當前策略的輸出重新訓練 RM：

```
for iteration in RL_steps:
    # 1. 使用當前策略收集資料
    current_outputs = generate(policy)
    # 2. 為這些輸出標註分數（用規則）
    labels = score(current_outputs)
    # 3. 更新 RM
    rm.train_on(current_outputs, labels)
    # 4. 使用 RM 做 RL
    rl_step(policy, rm)
```

這類似於**對抗式訓練（Adversarial Training）** — RM 不斷適應策略的分布變化。

### 3. 改用 Pairwise Loss

使用偏好學習（Bradley-Terry loss）而非回歸損失，可以讓 RM 更專注於相對排序而非絕對分數：

```python
def pairwise_loss(scores_A, scores_B, preference):
    # preference = 1 表示 A > B
    logits = scores_A - scores_B
    return -F.logsigmoid(preference * logits)
```

這種方法通常對分布偏移更穩健，因為 RM 只需要知道相對順序而非精確分數。

### 4. 標尺（Calibration）

對 RM 的輸出進行後處理，拉伸分數範圍：

```python
# 訓練完成後，在驗證集上對 RM 分數進行線性縮放
scores = [rm(text) for text in val_set]
min_s, max_s = min(scores), max(scores)
rm_score_normalized = lambda x: (rm(x) - min_s) / (max_s - min_s)
```

但這只是一個治標不治本的解決方案，無法解決分布偏移的根本問題。

## 當使用 RM 而非規則獎勵

### RM 適合的場景

1. **人類偏好學習** — 當獎勵來自人類評分且無法自動化時，RM 是不可或缺的橋樑
2. **多維度品質評估** — 同時考慮多個指標（安全性、相關性、流暢性）並整合為單一分數
3. **長文本生成** — 摘要、翻譯、創意寫作等任務無法用簡單規則評分
4. **對話品質** — 多輪對話的「有用性」需要理解上下文來評估

### 規則獎勵適合的場景

1. **明確答案的任務** — 數學、程式碼、選擇題、事實性問答
2. **可程式化驗證的任務** — 程式執行結果、資料庫查詢結果、棋盤遊戲勝負
3. **二元結果的任務** — 分類正確性、格式合規性、關鍵字包含性

### 混合方案

對於複雜任務，最常見的實務做法是結合兩者：

```
reward_total = α · RM_score + (1-α) · rule_score
```

- 規則獎勵提供穩定的基本信號（避免 RM 的崩潰風險）
- RM 獎勵捕捉主觀品質（超越規則能表達的部分）
- α 控制兩者的權重

## 延伸閱讀

- InstructGPT: Ouyang et al., 'Training language models to follow instructions with human feedback' (NeurIPS, 2022)
- 獎勵模型訓練詳解: Lambert et al., 'A Survey of Reinforcement Learning from Human Feedback' (arXiv, 2024)
- Distribution Shift in RM: Gao et al., 'Scaling Laws for Reward Model Overoptimization' (ICML, 2023)
- Bradley-Terry 偏好模型: 'Bradley-Terry model for paired comparisons' (Biometrika, 1952)
- 對抗式 RM 訓練: Cheng et al., 'Adversarial Reward Model Training' (arXiv, 2024)
- v4 失敗案例分析: mini-llm v4 reinforce_rm.py + train_reward_model.py
