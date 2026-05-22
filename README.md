# mini-llm — 微型語言模型訓練流程

一個從零實作、純 PyTorch 的玩具級語言模型（toy LLM），展示**預訓練（Pre-training）+ 微調（Fine-tuning）+ 知識蒸餾（Knowledge Distillation）** 的完整流程。全部程式碼不加依賴框架，可在 CPU 上執行。

## 目錄

- [專案背景](#專案背景)
- [三階段架構](#三階段架構)
  - [v1-pretrain：單階段字元級語言模型](#v1-pretrain單階段字元級語言模型)
  - [v2-finetune：兩階段預訓練 + 微調](#v2-finetune兩階段預訓練--微調)
  - [v3-distill：大模型知識蒸餾](#v3-distill大模型知識蒸餾)
- [模型架構](#模型架構)
- [執行方式](#執行方式)
- [資料流](#資料流)
- [合成資料生成](#合成資料生成)
- [目錄結構](#目錄結構)
- [無依賴設計](#無依賴設計)
- [Wiki](#wiki)

## 專案背景

`mini-llm` 是一個教育性質的深度學習專案，目標是以最精簡、最可讀的程式碼，完整呈現大型語言模型的核心訓練流程。

現代 LLM（如 GPT-4、LLaMA、Claude）的訓練通常分為多個階段：大規模預訓練 → 指令微調 → 強化學習對齊。`mini-llm` 將這個流程濃縮到單一目錄即可執行完畢的程度，適合用於學習 Transformer 訓練的原理與實作。

核心技術特點：
- **純 PyTorch 實作** — 不依賴 HuggingFace Transformers、DeepSpeed 等框架，每一行程式碼都可讀
- **CPU 可執行** — 不需要 GPU，一般筆記型電腦即可運行
- **架構現代化** — 使用 RoPE、RMSNorm、SwiGLU 等現代 LLM 標準組件
- **三種資料生成方式** — 規則生成、模板填充、大模型蒸餾，對比不同方法的差異

## 三階段架構

本專案分為三個獨立階段，每個階段都可在自己的目錄下獨立執行。階段的複雜度依次遞增。

### v1-pretrain：單階段字元級語言模型

**最簡版本，只做預訓練。**

讀取一個 `.txt` 檔案（內含約 1500 字的手寫中文句子），訓練一個三層 Transformer 來預測下一個字元：

```bash
cd v1-pretrain
python mini-llm.py --file input.txt
```

訓練完成後自動用 prompt 「小貓坐」生成接續文字。支援自訂參數：
- `--iters`：訓練步數（預設 2000）
- `--seq_len`：上下文長度（預設 32）
- `--batch_size`：批次大小（預設 16）
- `--gen_len`：生成字元數（預設 100）

此版本的模型約 0.8M 參數，2000 步即可收斂（~0.09 Loss），能在 CPU 上 1 分鐘內完成訓練。

### v2-finetune：兩階段預訓練 + 微調

**加入微調階段，展示遷移學習範式。**

第一階段：用合成語料進行預訓練（500 步），讓模型學習語言結構與事實知識。
第二階段：載入預訓練權重，用問答格式的資料進行微調（300 步），讓模型學會回答問題。

執行方式：

```bash
cd v2-finetune
./run.sh
```

`run.sh` 的內容（可自行編輯選擇資料生成器）：

```bash
# python prepare_data.py          # 從 HuggingFace 下載真實語料
# python gen_data_rule.py         # 數學與規律資料
python gen_data_wuxia.py          # 武林人物知識（預設啟用）
# python gen_data_robot.py        # 智慧家庭助理
python pretrain.py                # 預訓練
python finetune.py                # 微調 + 自動測試
```

微調完成後，`finetune.py` 會自動從 `finetune.txt` 中擷取第一題進行測試，展示模型的問答能力：

```
📝 抽取到的題目: <Q>令狐沖的武功是什麼？<A>
🎯 預期的解答: 獨孤九劍
🤖 AI 實際輸出:
<Q>令狐沖的武功是什麼？<A>獨孤九劍
<Q>段譽在哪裡練武？<A>大理
...
```

### v3-distill：大模型知識蒸餾

**將大模型的知識濃縮到小模型中。**

使用 NVIDIA API 呼叫 Minimax M2.7 模型，生成關於太陽系行星的結構化訓練資料（50 條 FACT + 50 條 QA），再用這些資料訓練小模型。

```bash
cd v3-distill
export NVIDIA_API_KEY='你的金鑰'
./run.sh
```

`run.sh` 的內容：

```bash
python gen_data_distill.py        # 呼叫大模型生成訓練資料（需要 API key）
python pretrain.py                # 預訓練
python finetune.py                # 微調
```

知識蒸餾的關鍵是**Prompt 工程**：要求大模型以 `[FACT]` 和 `[QA]` 的嚴格格式輸出，方便後續解析成標準訓練資料。

## 模型架構

所有版本共用相同的核心架構（`model.py` 中的 `ModernLanguageModel`），採用現代 LLM 的標準設計：

```
輸入 Token IDs
    ↓
Token Embedding (vocab_size → d_model)
    ↓
    TransformerBlock × N 層
    ├── RMSNorm → 因果自注意力 (RoPE)
    ├── 殘差連接
    ├── RMSNorm → SwiGLU 前饋網路
    └── 殘差連接
    ↓
RMSNorm
    ↓
輸出投影 (d_model → vocab_size) ← 權重共享
    ↓
Softmax → 預測機率
```

### 核心組件

| 組件 | 說明 | 程式碼位置 |
|------|------|-----------|
| **RoPE（旋轉位置編碼）** | 以旋轉矩陣將相對位置資訊注入注意力計算，無需額外參數 | `precompute_freqs_cis()` + `apply_rotary_emb()` |
| **RMSNorm** | 精簡版 LayerNorm，僅對活化值的均方根進行正規化，計算效率更高 | `RMSNorm` 類別 |
| **SwiGLU** | Swish 活化 + 門控線性單元，提供比 ReLU 更豐富的表達能力 | `FeedForward` 類別中 `F.silu(w1(x)) * w3(x)` |
| **權重共享** | 嵌入層與輸出層共用同一組權重，減少參數量 | `self.tok_emb.weight = self.output.weight` |
| **因果遮罩** | 下三角矩陣強制每個位置只能看到歷史 token | `torch.tril()` + `masked_fill()` |

### 預設超參數

| 參數 | v1 | v2/v3 |
|------|-----|-------|
| d_model | 128 | 128 |
| n_heads | 4 | 4 |
| n_layers | 3 | 4 |
| seq_len | 32 | 64 |
| 總參數量 | ~0.8M | ~0.8M |
| 預訓練步數 | 2000 | 500 |
| 微調步數 | — | 300 |
| 預訓練學習率 | 3e-4 | 5e-4 |
| 微調學習率 | — | 1e-4 |
| 批次大小 | 16 | 32 |

### 為什麼這些超參數？

- **d_model = 128**：維持參數量在 1M 以下，確保 CPU 可在數分鐘內完成訓練
- **n_layers = 4**（v2/v3）：足夠形成有意義的表示層次，又不會過深導致訓練時間過長
- **seq_len = 64**：滿足問答格式的上下文需求（問題+答案的總長度通常不超過 64 字元）
- **微調學習率較低**：避免災難性遺忘 (catastrophic forgetting)，在保留預訓練知識的同時適應任務

## 執行方式

### 環境需求

- Python 3.8+
- PyTorch 2.0+（CPU 版本即可）
- （v3-distill 需要）`openai` Python 套件 + NVIDIA API Key
- （可選）`datasets` 套件 — 僅 `prepare_data.py` 和 `tool/build_corpus.py` 需要

```bash
pip install torch
# 可選依賴
pip install openai          # v3-distill 需要
pip install datasets tqdm   # tool/build_corpus.py 需要
```

### 執行流程

```
                    ┌──────────────────────────────────────────────────┐
                    │              mini-llm 執行流程                        │
                    └──────────────────────────────────────────────────┘

v1-pretrain:
    python mini-llm.py --file input.txt
        │
        ├── 讀取 input.txt（~1500 字）
        ├── 建立字元級詞表（vocab_size ≈ 96）
        ├── 訓練 2000 步
        └── 生成測試文字

v2-finetune:
    ./run.sh 等同於：
        ├── gen_data_wuxia.py   (或其他資料生成器)
        │   ├── 產生 vocab.pkl
        │   ├── 產生 pretrain_data.pt
        │   └── 產生 finetune_data.pt
        ├── pretrain.py
        │   ├── 載入 vocab.pkl 與 pretrain_data.pt
        │   ├── 訓練 500 步
        │   └── 儲存 pretrain.pt
        └── finetune.py
            ├── 載入 vocab.pkl、finetune_data.pt、pretrain.pt
            ├── 訓練 300 步
            ├── 儲存 finetune.pt
            └── 自動測試問答能力

v3-distill:
    ./run.sh 等同於：
        ├── gen_data_distill.py
        │   ├── 呼叫 NVIDIA API
        │   ├── 解析 FACT/QA 格式
        │   └── 產生 vocab.pkl + .pt 檔案
        ├── pretrain.py (同 v2)
        └── finetune.py (同 v2)
```

## 資料流

本專案使用**字元級（character-level）** 分詞器，這是目前最簡的 tokenization 方式，適合教育示範。

```
原始文字             字元級詞表               Tensor 資料
「郭靖在桃花島        {'郭':0, '靖':1,        [0, 1, 2, 3, 4, 5,
  苦練降龍十八掌。」     '在':2, '桃':3,         6, 7, 8, 9, 10, 11]
                       '花':4, '島':5, ...}
                           ↓
詞表儲存為 vocab.pkl   ┌── stoi: 字元→索引
（Pickle 格式）          ├── itos: 索引→字元
                       └── vocab_size: 不重複字元數
```

### 資料生成腳本的比較

| 腳本 | 領域 | 種子句子數 | 目標字數 | 外部依賴 |
|------|------|-----------|---------|---------|
| `gen_data_wuxia.py` | 武俠人物知識 | ~30 句 | 200K (pretrain) + 50K (finetune) | 無 |
| `gen_data_rule.py` | 數學與規律 | ~100 句 | 200K + 50K | 無 |
| `gen_data_robot.py` | 智慧家庭助理 | ~60 句 | 200K + 50K | 無 |
| `gen_data_distill.py` | 太陽系行星 | ~100 句 | 100K + 30K | NVIDIA API |
| `prepare_data.py` | 通用（中文維基） | 1000 篇 wiki | 100 萬字 | `datasets` |

所有生成器都輸出相同的檔案格式（`vocab.pkl`、`pretrain_data.pt`、`finetune_data.pt`），因此可以互換使用。

### 產生的檔案

所有自動產生的檔案都已加入 `.gitignore`：

| 檔案 | 說明 |
|------|------|
| `vocab.pkl` | 字元級詞表（`stoi`, `itos`, `vocab_size`） |
| `pretrain_data.pt` | 預訓練語料的 PyTorch Tensor（int64 序列） |
| `finetune_data.pt` | 微調語料的 PyTorch Tensor |
| `pretrain.pt` | 預訓練完成的模型權重（state_dict） |
| `finetune.pt` | 微調完成的模型權重 |
| `pretrain.txt` | 預訓練語料的純文字版本（方便檢視） |
| `finetune.txt` | 微調語料的純文字版本（`finetune.py` 用於測試） |

## 合成資料生成

由於本專案不使用真實的網路語料（為了可重現性和獨立性），所有訓練資料都是合成產生的。有三種策略：

### 1. 規則生成（Rule-Based）

`gen_data_rule.py` 透過數學規則和邏輯關係生成資料。例如：

```
一加一等於二。
二加三等於五。
星期一的明天是星期二。
鯨魚比大象大。
```

規律性極強，適合展示模型學習模式的能力。

### 2. 模板填充（Template Filling）

`gen_data_wuxia.py` 和 `gen_data_robot.py` 使用預定義的角色/意圖配對模板。例如：

```
郭靖在桃花島苦練降龍十八掌。
<Q>令狐沖的武功是什麼？<A>獨孤九劍
```

### 3. 大模型蒸餾（LLM Distillation）

`gen_data_distill.py` 呼叫 NVIDIA API，使用 Minimax M2.7 模型產生訓練資料。優點是內容更多樣化，缺點是需要 API 金鑰和網路連線。

### 資料擴增技術

所有生成器都使用**重複打亂拼接（Shuffle-and-Repeat）** 技術將少量種子句子擴增到目標資料量：

```python
pretrain_text = ""
while len(pretrain_text) < TARGET_LENGTH:
    pretrain_text += "".join(random.sample(facts, len(facts)))
```

這種方法讓種子資料被反覆以不同的順序排列，增加模型對句子的曝光次數。

## 目錄結構

```
mini-llm/
├── AGENTS.md             # OpenCode 設定檔（AI agent 使用）
├── README.md             # 本檔案
├── run.sh                # 根目錄執行腳本（指向 v1）
├── _ai.md                # 參考連結
├── _wiki/                # 技術名詞 wiki
│   ├── index.md
│   ├── rmsnorm.md
│   ├── rotary-position-embedding.md
│   ├── swiglu.md
│   ├── weight-tying.md
│   ├── causal-self-attention.md
│   ├── character-level-tokenizer.md
│   ├── pretraining-finetuning.md
│   ├── knowledge-distillation.md
│   ├── cross-entropy-loss.md
│   ├── adamw-optimizer.md
│   ├── gradient-clipping.md
│   ├── synthetic-data-augmentation.md
│   ├── autoregressive-language-model.md
│   └── gradient-descent.md
│
├── v1-pretrain/          # 第一階段：單階段字元級語言模型
│   ├── mini-llm.py           # 完整的語言模型訓練程式（含架構與訓練）
│   ├── mini-llm.md           # 執行記錄
│   └── input.txt          # 訓練語料（手寫 147 行中文句子）
│
├── v2-finetune/          # 第二階段：預訓練 + 微調
│   ├── model.py           # Transformer 模型定義（ModernLanguageModel）
│   ├── pretrain.py        # 預訓練腳本
│   ├── finetune.py        # 微調腳本 + 測試
│   ├── prepare_data.py    # 從 HuggingFace 下載真實語料（需要 datasets）
│   ├── gen_data_wuxia.py  # 合成資料生成：武俠人物
│   ├── gen_data_rule.py   # 合成資料生成：數學與規律
│   ├── gen_data_robot.py  # 合成資料生成：智慧家庭助理
│   ├── run.sh             # 一鍵執行腳本
│   └── run.md             # 執行記錄
│
├── v3-distill/           # 第三階段：大模型知識蒸餾
│   ├── model.py           # 同 v2 的模型定義
│   ├── pretrain.py        # 同 v2 的預訓練腳本
│   ├── finetune.py        # 同 v2 的微調腳本
│   ├── gen_data_distill.py# 知識蒸餾：呼叫 NVIDIA API 生成資料
│   ├── gen_data_wuxia.py  # 重複（與 v2 相同）
│   ├── gen_data_rule.py   # 重複（與 v2 相同）
│   ├── gen_data_robot.py  # 重複（與 v2 相同）
│   ├── prepare_data.py    # 重複（與 v2 相同）
│   ├── run.sh             # 一鍵執行腳本（預設使用 distill）
│   └── run.md             # 執行記錄
│
├── tool/                 # 輔助工具
│   ├── build_corpus.py    # HuggingFace 語料建構工具（需要 datasets）
│   └── processed_corpus/  # 處理後的語料庫
│       ├── train.jsonl
│       └── eval.jsonl
│
├── .gitignore
└── LICENSE
```

## 無依賴設計

本專案刻意不使用 `requirements.txt` 或 `pyproject.toml`，以保持最小依賴。唯一強制需要的是 `torch`。

| 工具/腳本 | 需要安裝 |
|-----------|---------|
| v1-pretrain/mini-llm.py | `torch` |
| v2-finetune/pretrain.py | `torch` |
| v2-finetune/finetune.py | `torch` |
| v2-finetune/\*gen\_data\_\*.py | 無（只用標準函式庫） |
| v2-finetune/prepare_data.py | `torch`, `datasets` |
| v3-distill/pretrain.py | `torch` |
| v3-distill/finetune.py | `torch` |
| v3-distill/gen\_data\_distill.py | `openai` |
| tool/build\_corpus.py | `datasets`, `tqdm` |

## Wiki

本專案的 [`_wiki/`](_wiki/index.md) 目錄包含 15 篇詳細的技術文件，涵蓋專案中使用到的所有核心概念，每篇約 300 行：

| 文件 | 主題 |
|------|------|
| [Autoregressive Language Model](_wiki/autoregressive-language-model.md) | 自迴歸語言模型原理 |
| [RMSNorm](_wiki/rmsnorm.md) | 均方根正規化 |
| [RoPE](_wiki/rotary-position-embedding.md) | 旋轉位置編碼 |
| [SwiGLU](_wiki/swiglu.md) | 門控活化函數 |
| [Weight Tying](_wiki/weight-tying.md) | 權重共享 |
| [Causal Self-Attention](_wiki/causal-self-attention.md) | 因果自注意力 |
| [Character-Level Tokenizer](_wiki/character-level-tokenizer.md) | 字元級分詞器 |
| [Pretraining + Fine-tuning](_wiki/pretraining-finetuning.md) | 預訓練與微調範式 |
| [Knowledge Distillation](_wiki/knowledge-distillation.md) | 知識蒸餾 |
| [Cross-Entropy Loss](_wiki/cross-entropy-loss.md) | 交叉熵損失函數 |
| [AdamW Optimizer](_wiki/adamw-optimizer.md) | AdamW 最佳化器 |
| [Gradient Clipping](_wiki/gradient-clipping.md) | 梯度裁切 |
| [Synthetic Data Augmentation](_wiki/synthetic-data-augmentation.md) | 合成資料擴增 |
| [Gradient Descent](_wiki/gradient-descent.md) | 梯度下降 |

---

**License**: MIT
