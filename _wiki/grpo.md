# GRPO (Group Relative Policy Optimization) — 群體相對策略最佳化

## 概述

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek-R1 在 2025 年提出的強化學習演算法，專為大型語言模型的推理任務設計。與傳統的 PPO 不同，GRPO 不使用 critic network（價值網路），而是透過在同一個 prompt 上生成多個回覆並在群體內做相對比較來計算優勢函數（advantage），大幅簡化了 RL 訓練的基礎設施需求。

mini-llm v4 實作了一個簡化版的 GRPO（在 `reinforce_grpo.py` 中），用於數學問答的 RL 微調。與 v4 中的 REINFORCE(EMA) 和 REINFORCE(RM) 相比，GRPO 展現了最佳的平均表現：對 seen 資料的準確率幾乎無損，對 unseen 資料有顯著提升。

## 為何需要 GRPO？

### 問題 1：REINFORCE 的基線問題

REINFORCE 使用單一的標量基線（如 EMA）來降低梯度變異數：

```
advantage = reward - baseline
```

但單一基線無法區分 prompt 的難度差異。對於簡單的 prompt（如 `1+1=？`），所有取樣幾乎都正確；對於困難的 prompt（如 `8+9=？`），正確率較低。同一個基線同時服務所有 prompt，導致：

- 簡單 prompt：reward 總是 > baseline，優勢永遠為正，梯度持續更新
- 困難 prompt：reward 可能總是 < baseline，優勢永遠為負，梯度方向正確但強度不當

### 問題 2：PPO 的複雜度

PPO 通過 critic network 解決了上述問題（每個 prompt 的基線由價值網路預測），但引入了：

- 需要維護一個 critic network（與策略網路規模相當）
- 需要訓練 critic network（另一套 loss 和優化器）
- 需要重要性取樣（importance sampling）和裁剪（clipping）
- 超參數數量倍增

### GRPO 的解決方案

GRPO 用一個簡單的統計技巧取代了 critic network：**對每個 prompt，在同一個 batch 內生成 G 個回覆，用群體內的平均獎勵作為基線**。

這個做法的直覺是：
- 同一個 prompt 的不同回覆之間的可比性遠高於不同 prompt 之間
- 群體平均自然地控制了 prompt 難度的影響
- 不需要額外的神經網路或 EMA 變數

## GRPO 演算法細節

### 核心公式

```
advantage_i = (reward_i - mean(reward_group)) / std(reward_group)
```

其中 `reward_group = [r_1, r_2, ..., r_G]` 是對同一個 prompt 生成的 G 個回覆所獲得的獎勵。

### 分組標準化（Group Normalization）的效果

考慮兩種極端情況：

**情況 A：簡單的 prompt（所有回覆都正確）**
```
reward_group = [1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8]
mean = 0.95, std ≈ 0.09
advantage = [0.55, 0.55, -1.66, 0.55, 0.55, 0.55, 0.55, -1.66]
```
- 即使所有回覆的絕對獎勵都很高，群體內比較仍然可以區分「更好」和「更差」
- 獎勵 0.8 的回覆得到負優勢，儘管它的絕對分數不低

**情況 B：困難的 prompt（大部分回覆錯誤）**
```
reward_group = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0]
mean = 0.19, std ≈ 0.34
advantage = [-0.55, -0.55, 2.41, -0.55, -0.55, -0.55, 0.92, -0.55]
```
- 只有一個正確回覆得到 +2.41 的強烈正優勢
- 部分正確（0.5）得到溫和的正優勢（0.92）
- 錯誤回覆得到一致的負優勢

標準化的優勢在不同 prompt 之間具有可比較的尺度（大致落在 [-3, 3] 之間），這使得 GRPO 不需要再對優勢進行裁剪（clipping），簡化了實作。

### KL 散度懲罰

GRPO 的另一個關鍵設計是 **KL 散度（Kullback-Leibler Divergence）懲罰**，用於限制策略更新不要偏離原始模型太遠。

#### 為什麼需要 KL 懲罰？

RL 訓練中的一個常見問題是 **reward hacking**：模型發現一種可以獲得高獎勵但語言能力崩壞的行為。例如，在數學問答任務中，模型可能學會不斷重複正確的數字（`888888...`）來提高正確機率，但這破壞了語言生成能力。

KL 懲罰通過將獎勵修正為：

```
reward_effective = reward - β · KL(π_θ || π_ref)
```

來防止策略 π_θ 偏離參考模型 π_ref 太遠。其中：

- `π_θ` — 當前策略（正在訓練的模型）
- `π_ref` — 參考策略（通常是 SFT 後的模型，凍結不更新）
- `KL(π_θ || π_ref)` — 兩個策略之間的 KL 散度
- `β` — 懲罰係數，控制約束的強度

#### KL 散度的計算

KL 散度衡量兩個機率分布之間的差異：

```
KL(P || Q) = Σ_x P(x) · log(P(x) / Q(x))
```

在語言模型的上下文中，對於每個 token 位置、詞表中的每個 token：

```
P(token) = π_θ(token | context)    # 當前策略的機率
Q(token) = π_ref(token | context)   # 參考模型的機率
```

v4 的 GRPO 實作逐 token 累積 KL 散度：

```python
# 每個 token 位置的 KL 計算
lp_theta = F.log_softmax(logits_last, dim=-1)   # log π_θ
p_ref = F.softmax(logits_ref[:, -1, :], dim=-1) # π_ref
traj_kl = traj_kl + (p_ref * (p_ref.log() - lp_theta)).sum()
```

這個計算方式對每個 token 位置的所有候選 token 求和。完整的 GRPO 損失為：

```python
loss = avg_pg + beta_kl * avg_kl
# avg_pg = - Σ_i (log π_θ(trajectory_i) * advantage_i) / G
# avg_kl = Σ_i KL(π_θ || π_ref)_i / G
```

### β 參數的作用

β 控制 KL 懲罰的強度：

| β 值 | 效果 | 風險 |
|------|------|------|
| β = 0 | 無約束，純 REINFORCE | reward hacking，語言崩壞 |
| β 很小（0.001） | 鬆散約束 | 可能仍會偏離但速度較慢 |
| β 適中（0.04） | 良好平衡 | v4 預設值，seen 準確率幾乎無損 |
| β 很大（0.5） | 強烈約束 | RL 幾乎不學習，接近 π_ref |

v4 的實驗表明，β = 0.04 對數學問答任務是合理的選擇：SEEN 準確率從 98% 僅下降到 90-97%，而 UNSEEN 從 71% 提升到 84-90%。

### v4 實作：reinforce_grpo.py

完整的訓練循環（去除了輔助程式碼）：

```python
for step in range(max_iters):
    prompt, expected = random.choice(unseen_pairs)
    prompt_ids = encode(prompt)

    group_log_probs = []
    group_kls = []
    group_rewards = []

    # 對同一 prompt 生成 G 個回覆
    for _ in range(G):
        gen = prompt_ids.copy()
        traj_log_probs = []
        traj_kl = 0.0

        # 自迴歸生成
        for _ in range(max_gen_len):
            ctx = torch.tensor(gen[-seq_len:], ...)
            logits_theta, _ = model(ctx)     # 當前策略
            logits_last = logits_theta[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            token = torch.multinomial(probs, 1)
            log_prob = -F.cross_entropy(logits_last, token.view(-1))
            traj_log_probs.append(log_prob)

            # KL 計算（使用凍結的參考模型）
            with torch.no_grad():
                logits_ref, _ = ref_model(ctx)
            lp_theta = F.log_softmax(logits_last, dim=-1)
            p_ref = F.softmax(logits_ref[:, -1, :], dim=-1)
            traj_kl = traj_kl + (p_ref * (p_ref.log() - lp_theta)).sum()

            gen.append(token.item())
            if token == newline_token:
                break

        reward = compute_reward(resp, expected)
        group_log_probs.append(torch.stack(traj_log_probs))
        group_kls.append(traj_kl)
        group_rewards.append(reward)

    # 群體相對優勢
    rewards_t = torch.tensor(group_rewards, ...)
    adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    # GRPO 損失
    total_pg = -sum(group_log_probs[i].sum() * adv[i] for i in range(G))
    total_kl = sum(group_kls)
    loss = total_pg / G + beta_kl * total_kl / G

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### 與 REINFORCE 的逐行對比

| 層面 | REINFORCE(EMA) | GRPO |
|------|----------------|------|
| **取樣** | 每步取樣一個 batch 的不同 prompt | 每步取樣一個 prompt，生成 G 個回覆 |
| **優勢計算** | R - baseline（EMA 歷史平均） | (R_i - μ_R) / σ_R（群體內標準化） |
| **正則化** | 無 | KL 散度懲罰 ∥ π_θ vs π_ref |
| **基線維護** | 需要 EMA 變數和 α 超參數 | 不需要（群體內動態計算） |
| **Batch size** | 8（一個 batch 多個不同 prompt） | G（同 prompt 的多個回覆） |

## 與 PPO 的詳細對比

PPO（Proximal Policy Optimization）是目前 RLHF 的標準演算法。GRPO 與 PPO 的核心差異：

### 1. Critic Network

PPO 需要一個 critic network 來估計價值函數 V(s)。這個網路通常與策略網路共享大多數層，只在最後多一個標量輸出頭。critic 的訓練本身就需要一套完整的監督式學習流程。

GRPO 完全不需要 critic network，直接用群體平均替代。

### 2. Importance Sampling

PPO 允許在舊資料上進行多次更新，因此需要重要性取樣：

```
ratio = π_new(a|s) / π_old(a|s)
```

GRPO 是 on-policy 的（每次更新都重新取樣），不需要重要性取樣。

### 3. Clipping

PPO 的裁剪機制：

```
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clipped_ratio * A)
```

GRPO 不需要 clipping，因為：
- 每次更新都使用新鮮取樣的資料，ratio = 1
- 群體標準化後的優勢自然在 [-3, 3] 範圍內

### 4. 參數數量對比

| 參數 | PPO | GRPO |
|------|-----|------|
| 策略網路 | 有 | 有 |
| 參考模型 | 可選 | 必要（KL 計算） |
| Critic 網路 | 必要 | 無 |
| clip ε | 必要 | 無 |
| KL β | 可選 | 必要 |
| GAE λ | 必要 | 不適用 |
| 優勢標準化 | 可選 | 必要 |

PPO 至少有 6 個關鍵超參數，而 GRPO 只有 3 個（G、β、lr）。

## GRPO 的視覺化分析

### 優勢分布

在一次實際運行中，G = 8 的群體內獎勵分布範例：

```
Prompt: <Q>3+8=？<A>

Rewards:  [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
Mean:     0.25
Std:      0.43
Adv:      [-0.58, 1.74, -0.58, -0.58, 1.74, -0.58, -0.58, -0.58]
```

兩個正確回覆（reward = 1.0）得到強烈的正優勢 +1.74，六個錯誤回覆得到負優勢 -0.58。梯度方向清晰且強度適中。

### KL 散度的動態變化

GRPO 訓練中的 KL 散度會出現「爆發—收縮」的循環模式：

```
Step   0 | KL: 0.0000  (剛從 π_ref 初始化，差異為零)
Step  25 | KL: 0.0153  (策略開始移動)
Step 175 | KL: 0.0872  (中度偏離)
Step 575 | KL: 0.0002  (策略「彈回」參考模型)
Step 800 | KL: 4.9170  (突然爆發，危險信號)
Step 825 | KL: 4.4163  (高 KL 持續)
Step 850 | KL: 0.0049  (迅速收縮回來)
```

KL 爆發（Step 800）的時機與 evalu 準確率的下滑相關聯。這說明：

1. **KL 是有效的預警信號** — 當 KL 突然增大時，策略正在做大幅度的、可能是破壞性的更新
2. **GRPO 有自我修正能力** — 即使 KL 爆發，約束項 `β · KL` 會迅速產生大的正損失，把策略拉回 π_ref 附近
3. **β 的數值需要足夠大** — 如果 β 太小，KL 爆發後可能無法恢復

## GRPO 在 v4 中的表現

### 定量結果

| 指標 | 初始 | GRPO 後 | 變化 |
|------|------|---------|------|
| SEEN 準確率 | 98.0% | 90.0% | -8.0% |
| UNSEEN 準確率 | 71.0% | 90.3% | **+19.3%** |

與其他方法對比：

| 方法 | SEEN(after) | UNSEEN(after) | 優勢 |
|------|-------------|---------------|------|
| REINFORCE(EMA) | 95.0% | 93.5% | 最高 unseen，seen 下降 3% |
| **GRPO** | **90.0%** | **90.3%** | **最平衡，seen 下降最多但 unseen 提升大** |
| REINFORCE(RM) | 38.0% | 9.7% | 完全失敗 |

### 分析

GRPO 在 v4 中的表現特徵：

1. **KL 懲罰提供了保護** — 相比 REINFORCE(EMA) 對 seen 準確率的輕微下降（98%→95%），GRPO 的下降更明顯（98%→90%）。這是因為 GRPO 的群體內比較可能產生更大的優勢值（尤其是當群體內只有少數正確回覆時），導致更激烈的策略更新。

2. **Unseen 提升與 REINFORCE(EMA) 相當** — GRPO 和 REINFORCE(EMA) 對 unseen 資料的改進幅度接近（+19.3% vs +25.8%），但 GRPO 需要更多步數（1000 vs 500）才能達到同等水準，這是因為 KL 懲罰限制了每次更新的幅度。

3. **隨機性較高** — 由於每步只對一個 prompt 生成 G 個回覆，GRPO 的梯度估計比 REINFORCE(EMA) 的 batch 取樣有更高的隨機性。這解釋了 seen 準確率在訓練過程中的較大波動（88%~100%）。

## 超參數選擇

### Group Size (G)

G 越大，優勢估計越準確，但計算成本線性增加：

- G = 4：速度最快，但群體統計量不穩定（例如 4 個樣本的 std 可能為 0）
- G = 8：v4 預設值，在準確性和效率間的良好平衡
- G = 16：優勢估計更穩定，但每步計算量翻倍

選擇 G 的指導原則：G 應該足夠大，使得群體內至少有 1-2 個正確和錯誤回覆。如果正確率接近 0% 或 100%，群體內比較會退化：

- 正確率 = 0%：所有 reward = 0，mean = 0，std = 0 → 除以零（需要加 epsilon）
- 正確率 = 100%：所有 reward = 1，mean = 1，std = 0 → 同上

v4 的任務初始正確率約 70%，G = 8 大致會產生 6 個正確和 2 個錯誤回覆，提供了足夠的對比。

### KL Penalty Coefficient (β)

β 的選擇依賴於任務和模型大小：

| β | 行為 | 適用情境 |
|---|------|---------|
| 0.01 | 弱約束 | 模型需要大幅改變行為 |
| 0.04 | 中等約束（v4 預設） | 一般用途 |
| 0.10 | 強約束 | 任務與 SFT 分布相似，只需微調 |
| 0.50 | 非常強 | 幾乎不允許改變 |

調整 β 的可觀察信號：
- 如果 seen 準確率急劇下降 → 增大 β
- 如果 KL 散度持續偏高（>1.0）→ 增大 β
- 如果 unseen 沒有改善 → 減小 β

### Learning Rate

GRPO 的學習率通常比 REINFORCE 更小，因為群體標準化後的優勢值可能較大：

- v4 GRPO 預設 lr = 1e-5（與 REINFORCE(EMA) 相同）
- 如果訓練不穩定（loss 大幅震盪），降低 lr 到 3e-6
- 如果收斂太慢，增大 lr 到 3e-5

## GRPO 與 DeepSeek-R1

v4 的 GRPO 實作是 DeepSeek-R1 演算法的簡化版本。主要差異：

| 層面 | DeepSeek-R1 | v4 GRPO |
|------|-------------|---------|
| 獎勵類型 | 混合（精確匹配 + 格式 + 語言一致性） | 單一（精確匹配） |
| Chain-of-Thought | 強制模型先產生推理鏈再給答案 | 無（直接輸出答案） |
| 取樣策略 | 每個 prompt 生成 G 回覆，保留多樣性 | 每個 prompt G 回覆 |
| KL 計算 | 近似 KL（無需對完整詞表求和） | 精確 KL（對完整詞表求和，詞表僅 31 字元） |
| 訓練規模 | 數萬步 × 超大 batch | 1000 步 × G=8 |
| 語言模型 | 數千億參數 | 約 1M 參數 |

DeepSeek-R1 的成功表明，GRPO 特別適合**有明確正確答案的推理任務**（數學、程式碼、邏輯推理），因為在這些任務中規則式獎勵可以直接取代昂貴的人類標註。v4 的玩具規模驗證了相同的基本原理。

## GRPO 的局限性

### 1. 群體內比較的統計限制

當 G 很小時（如 G < 8），群體統計量（mean 和 std）的估計誤差很大。如果某個 group 恰好全是正確或全是錯誤的，標準化完全失效：

```python
# G = 4, 全部正確
rewards = [1.0, 1.0, 1.0, 1.0]
std = 0.0  # 除以零！即使加 epsilon，所有 advantage = 0
```

解決方案：在訓練初期使用較大的 G，或混合使用 EMA 基線和群體基線。

### 2. 僅適用於可 batch 比較的任務

GRPO 要求對同一個 prompt 生成多個回覆。對於無法批次處理的任務（如與環境互動的機器人控制），GRPO 不適用。

### 3. 計算效率

GRPO 每步對同一個 prompt 生成 G 個回覆，而非 G 個不同 prompt。這意味著每步只能從一個 prompt 學習，訓練效率低於可以同時學習多個 prompt 的 REINFORCE batch。

具體計算量對比（假設 batch_size = G = 8，max_gen_len = 20）：

| 方法 | 每步生成次數 | 學習的 prompt 數 | 梯度多樣性 |
|------|-------------|-----------------|-----------|
| REINFORCE(EMA) | 8×20 = 160 token | 8 | 高（不同 prompt） |
| GRPO | 8×20 = 160 token | **1** | 低（同一 prompt） |

GRPO 的優點是**梯度品質更高**（群體內比較更準確），缺點是**每步的覆蓋範圍更窄**。

### 4. 需要凍結的參考模型

GRPO 需要維護一個凍結的參考模型 π_ref 來計算 KL 散度。這意味著記憶體需求翻倍（與 PPO 相似）。對於 v4 的玩具模型（約 1M 參數），這不是問題。對於大型模型，可以使用**近似 KL** 來避免載入第二份模型：

```
KL ≈ (π_θ - π_ref)^2 / (2 · π_ref)
```

這是基於 KL 的二階泰勒展開近似，只需要儲存 π_ref 的 logits 而非完整模型。

## 延伸閱讀

- DeepSeek-R1 論文: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv, 2025)
- GRPO 原始提案: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning with Open Language Models' (arXiv, 2024)
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv, 2017)
- KL 散度在 RL 中的應用: Jaques et al., "Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences" (ICLR, 2020)
- REINFORCE vs GRPO 比較: mini-llm v4 實作 (reinforce_grpo.py, reinforce_ema.py)
