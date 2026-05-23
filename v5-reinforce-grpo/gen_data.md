# Synthetic Data Generation for Mini-LLM: Theoretical Background

## Overview

`gen_data.py` produces three corpora from 81 addition problems (1+1 through 9+9):

| Corpus | Target Size | Content | Problems Used |
|--------|-------------|---------|---------------|
| `pretrain.txt` | 100,000 chars | Shuffled fact statements | All 81 |
| `finetune.txt` | 30,000 chars | Q&A pairs (seen problems) | 50 seen |
| `rl_unseen.txt` | ~500 chars | Q&A pairs (held out) | 31 unseen |

All problems are randomly shuffled once (seed 42), then the first 50 become the SFT pool and the remaining 31 the RL pool. The single seed ensures reproducibility of the split across runs.

---

## 1. Why Synthetic Data Works for Small Language Models

### The Manifold Hypothesis in Language

Language, like natural data, lives on a low-dimensional manifold embedded in a high-dimensional token space. A small model (~1M parameters, d_model=128, 4 layers) cannot possibly cover the full distribution of natural text. Synthetic data carves out a *toy manifold* — a small, regular region of token space that the model can actually learn. The addition problems 1+1 through 9+9 define a finite combinatorial space of exactly 81 input-output pairs. The model's capacity is well-matched to this space: it has enough parameters to memorize and generalize within it, but not so many that it overfits noise.

### Structured Regularity as Inductive Bias

Synthetic data is not "fake data" — it is *structured data with known ground truth*. Every addition problem has a deterministic answer. This means the reward signal in RL is perfect: no human annotator noise, no ambiguous labeling. The gradient signal is therefore as clean as possible, which is critical when debugging RL algorithms like REINFORCE, GRPO, or reward-model-based RL. If the model fails to learn, the fault lies in the optimization, not the data.

### The 81-Problem Design

The choice of 1+1 through 9+9 — all 81 pairwise combinations — is deliberate: it is the smallest non-trivial arithmetic domain that requires both memorization of facts and generalization of the addition concept. The model cannot simply learn "always output 2" (that would only cover 1+1); it must distinguish all 81 cases. Yet the space is small enough that training converges in minutes on CPU, enabling rapid iteration.

---

## 2. Data Distribution and Curriculum Learning

### Uniform vs. Frequency-Based Sampling

Natural language follows Zipf's law: a few tokens appear very frequently, while most are rare. Synthetic pretraining data in this project uses *uniform repetition* — every fact appears the same number of times. This is intentional: for a toy domain, we want the model to learn all facts equally well, not to develop a frequency bias. If 1+1 appeared 10x more often than 9+9, the model would learn 1+1 first and might generalize poorly to rare facts.

### Implicit Curriculum via Repeated Shuffling

The pretrain loop repeatedly shuffles the 81 facts and concatenates them, building up 100K characters. This produces an implicit curriculum in two senses:

1. **Interleaved exposure**: Each training epoch (one pass through all 81 facts) exposes the model to the full distribution. No fact is seen again before all others have been seen once.
2. **Random boundary placement**: The `get_batch` function samples random 64-token sequences from the concatenated text. Because shuffling randomizes the order of facts, the model sees every possible bigram boundary between adjacent facts across different training steps.

This differs from curriculum learning (easy facts first), which is unnecessary here because all 81 problems have identical complexity at the character level.

### The 100K Target: Why Not Less, Why Not More

Pretraining continues until the concatenated text reaches ~100K characters. With seq_len=64 and batch_size=32, this yields roughly `100000 / (64 * 32) ≈ 49` gradient steps per pass through the data. The total of 500 training steps means the model sees the full set of facts roughly 10 times. This repetition count is calibrated to the model's capacity: 4 layers, d_model=128, 4 heads. More data would be wasted; less would leave the model underfit.

The 30,000-character SFT target follows the same logic but accounts for the longer format of Q&A pairs (`<Q>1+1=？<A>2` is 12 characters vs. `1+1=2。` at 5 characters). The 50 SFT problems repeated to fill 30K chars give roughly 30,000 / (12 * 50) ≈ 50 passes per SFT problem.

---

## 3. Pretrain vs. SFT vs. RL Data: Divergent Requirements

### Pretrain Data: Density and Coverage

Pretraining optimizes for two things:

- **Token-level next-token prediction (NTP)**. The model sees `1+1=2。` and must learn to predict `2` after `=`, `。` after `2`, and `+` after `1`. The dense, fact-only format maximizes the number of prediction opportunities per character.
- **Distributional coverage**. All 81 facts are present. The model forms a joint distribution over the entire problem space.

The pretrain data contains one non-fact sentence: `加法是把兩個數字合併成一個數。` ("Addition is the operation of combining two numbers into one."). This single general statement serves as a minimal grounding of the *concept* of addition in natural language, distinguishing the operation from mere string pattern matching. The model learns that `+` means something beyond a trigger character.

### SFT/Finetune Data: Instruction Following

Finetune data shifts from dense facts to a **dialog format**. The key differences:

| Aspect | Pretrain | SFT |
|--------|----------|-----|
| Format | `1+1=2。` | `<Q>1+1=？<A>2` |
| Tokens per fact | 5 | 12 |
| Seen problems | 81 | 50 |
| Train objective | NTP | NTP (same loss, different distribution) |

The Q&A wrapper changes the conditional distribution the model learns. During pretrain, the model predicts answer *after* seeing the full fact. During SFT, the model must learn to *wait for the `<A>` token* before generating the answer. This is a form of **instruction tuning at character scale**: the model learns a conditional generation protocol.

### RL Data: Held-Out Generalization

The 31 RL-unseen problems are structurally identical to the 50 SFT problems — same format, same difficulty. The split is purely about **distribution shift**: the model has never seen the correct answer for these problems during supervised training. This tests whether the model has learned the *underlying addition function* or merely memorized 50 input-output pairs. If the model generalizes to the 31 unseen problems during RL (where rewards are computed from ground-truth answers), it demonstrates that the pretrain+SFT stages taught the model the addition concept, not just a lookup table.

---

## 4. Train/Test Split Methodology

### Why 50/31?

81 is 3 * 3 * 3 * 3 (3^4). The split of 50 seen / 31 unseen is not round, and that is deliberate:

- **50/31 is not an even fraction**, so there is no trivial pattern (like "first half / second half" by operand) that the model could exploit. The random shuffle (seed 42) ensures the split is arbitrary with respect to the natural ordering (1+1, 1+2, ..., 9+9).
- **50 > 31**: The SFT set is larger than the RL set to ensure the model fits the seen distribution well before RL attempts generalization. Underspecifying the seen set would make it hard to distinguish "model failed to learn SFT" from "model failed to generalize in RL."
- **31 is still statistically meaningful**: With 31 held-out problems, a model that generalizes perfectly would achieve 100% on the RL set; a model that memorizes only the SFT set would score 0/31. This creates a clear signal.

### Why Not Cross-Validation?

For a toy pipeline, a single fixed split is superior to cross-validation:

1. **Reproducibility**: Every run uses the same seen/unseen split, so RL algorithm comparisons (REINFORCE vs. GRPO vs. reward-model RL) are apples-to-apples.
2. **Simplicity**: The codebase has no test harness; the split is the test harness.
3. **The seed is part of the experiment definition**: Changing the seed changes the difficulty of the split (some 50-problem subsets may be easier to generalize from than others). By fixing seed=42, the split becomes a controlled experimental constant.

### Stratification Considerations

The random shuffle does not stratify by operand. This is acceptable because:
- Each operand value (1-9) appears in 9 problems (e.g., `a=1` has `1+1` through `1+9`).
- With 50/81 ≈ 62% of problems in SFT, each operand is approximately 62% likely to appear in SFT. No operand is completely absent from SFT.
- Stratification would require tracking operand frequencies, adding complexity without clear benefit for a 9×9 domain.

---

## 5. Character-Level Tokenization Implications

### Vocabulary Construction

The tokenizer is a simple character-level encoder built from the union of all characters appearing in `pretrain.txt` and `finetune.txt`:

```
vocab_size = len(set(pretrain_text + finetune_text))
```

For this dataset, the vocabulary is approximately 20 characters:

```
123456789+ = ？。 \n < Q A 加 法 是 把 兩 個 數 字 合 併 成 一 。 <— note: 法 and 數 may be included
```

### Design Implications

**1. Every character is equally expensive.** The model pays the same prediction cost for `<`, `Q`, `>`, `1`, `+`, `=`, `？`, `A`, `2` as it does for semantic content. This incentivizes compact formats. The Q&A format `<Q>1+1=？<A>2` is 12 characters vs. `1+1=2。` at 5 characters — 2.4× the sequence length for the same information. This is a deliberate cost: the model must learn to process the extra tokens as protocol, not noise.

**2. The `<` and `>` tokens serve as structural markers.** In a subword tokenizer (BPE, WordPiece), `<Q>` might be a single token. In character-level tokenization, the model must learn that ` < Q > ` is a *chunk* — a multi-character pattern that signals "question begins." This is harder for the model but makes the learning more transparent to debug.

**3. No out-of-vocabulary (OOV) issues.** Character-level tokenization guarantees every possible string in the domain is representable. The model can theoretically generate any character sequence, including malformed ones like `<X>1+1=？<B>`, which the RL reward function can penalize.

**4. The Chinese characters add representational load.** Characters like `加`, `法`, `把`, `兩`, `個`, `數`, `字`, `合`, `併`, `成` appear only in the single grounding sentence. The model must allocate a small amount of capacity to these characters despite their near-zero frequency outside that sentence. This is a deliberate stress test: can the model remember rare characters after only one exposure per training epoch?

### Sequence Length and Context Window

With seq_len=64 and a Q&A format averaging ~12 chars, roughly 5 full Q&A pairs fit in one context window. The model therefore sees multiple "conversations" in a single training example, learning to transition between them. The newline character `\n` serves as a segment separator that the model must learn to predict, effectively performing a rudimentary form of document boundary detection.

---

## 6. Data Quality and Downstream RL Performance

### The Quality Chain

In this pipeline, data quality propagates forward:

```
Data quality (gen_data.py)
  → Pretrain quality (pretrain.py)
    → SFT quality (finetune.py)
      → RL initiation quality (reinforce_*.py)
        → Final RL policy quality
```

At each stage, errors compound. The synthetic data is deliberately *perfect* to isolate the effects of RL algorithm choice from data issues.

### What Constitutes "High Quality" for Each Stage

| Stage | Quality Dimension | Why It Matters for RL |
|-------|------------------|----------------------|
| Pretrain | Fact correctness | Wrong facts → wrong beliefs → RL cannot unlearn easily |
| Pretrain | Coverage of all answer tokens | Missing an answer token (e.g., `18` for `9+9`) → model cannot generate that string → RL reward is unachievable |
| SFT | Format consistency | Inconsistent `<Q>/<A>` usage → model generates malformed responses → RL reward function penalizes format → sparse reward signal |
| SFT | Seen/Unseen boundary leak | If any RL-unseen answer appears in SFT → RL generalization test is contaminated |

### The Contamination Safeguard

The script explicitly separates `sft_pool` and `rl_pool` before generating any data. No RL-unseen answer appears in `finetune.txt`. This is verified by the print statement at the end, which serves as a manual audit log. In a production setting, one would add an assertion:

```python
assert all(f"{r}" not in finetune_text for _, _, r in rl_pool)
```

The absence of this assertion in the script is a deliberate simplicity choice — for a toy pipeline, manual verification suffices — but it represents a quality gap that a production system would need to fill.

### Format Noise and Reward Signal

The RL reward function checks for the exact answer string. If the model generates `2` instead of `2` (trailing space), or `2。` (Chinese period), the reward is zero. The Q&A format with exact `<A>` prefix primes the model to generate clean answers. This is a form of **reward shaping through data design**: by making the output format highly constrained, we increase the probability that the model's generated text is parseable by the reward function.

---

## 7. Repetition in Pretraining Data

### The Case for Repetition

Modern scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) suggest that for optimal compute efficiency, models should not see data multiple times (i.e., training should be one epoch). However, these laws apply to:

- Large models (billions of parameters)
- Massive, diverse datasets (trillions of tokens)
- Compute-optimal training regimes

For small models on small, structured domains, the dynamics are different:

1. **The data manifold is fully covered in one epoch.** There are only 81 facts. A single pass gives the model at most 81 * 5 = 405 character-level observations. With d_model=128 and 4 layers, the model has ~1M parameters to fit from 405 characters — extreme overparameterization. Multiple epochs are necessary for the optimizer to converge.

2. **Repetition compensates for small batch size.** With batch_size=32 and seq_len=64, each gradient step observes at most 2048 characters. In one epoch (~49 steps), the model sees each fact only a few times. Multiple epochs (500 steps / ~10 passes) allow the gradient to average over multiple presentations of the same fact.

3. **Interleaved repetition prevents catastrophic forgetting.** Because facts are shuffled each pass, the model continues to see old facts as it learns new ones. This contrasts with block repetition (AAAA BBBB CCCC), where the model would forget A when learning B.

### The 100K Character Target

The target of 100,000 characters is derived from `(average chars per fact) * (number of facts) * (desired passes)`. With ~5 chars per fact and 81 facts:

- 1 pass = 405 chars
- ~247 passes to reach 100K chars

In practice, the while loop fills slightly more than 100K chars and then truncates. This means the last chunk of pretrain data is a partial pass, which is fine — the random offset in `get_batch` means the model sees all boundary conditions.

### Why Not Add More Variety?

An alternative design would use more diverse pretrain data: subtraction, multiplication, word problems, etc. The theory against this for a toy pipeline:

1. **Debugging clarity**: If the model fails, we want the cause to be in the RL algorithm, not the data diversity.
2. **Controlled variables**: The RL reward function only checks single addition. Additional operations would require additional reward functions, complicating the comparison between RL methods.
3. **Character budget**: With a fixed 100K target, adding variety means each addition fact appears fewer times, potentially reducing the model's confidence in its addition knowledge before RL begins.

---

## 8. Format Design Theory

### The Q&A Protocol

The format `<Q>{question}？<A>{answer}` is a minimal instantiation of the **instruction-response protocol** used in modern LLM fine-tuning (e.g., `<|im_start|>user`, `<|im_start|>assistant` in ChatML, or `[INST]` in Llama). The design choices:

**`<Q>` and `<A>` as special tokens.** In subword-tokenized models, these are typically added to the vocabulary as special tokens. In character-level tokenization, they are regular characters. This tests whether a small model can learn to treat multi-character sequences as structural markers through pure next-token prediction — a non-trivial test of in-context learning.

**The Chinese question mark `？`.** This character is visually distinct from `=` and `.` (used in the pretrain format). It signals "this is a question, not a statement." The model must learn that `？` triggers the expectation of `<A>`. This is a form of **pragmatic marker learning**: the model learns a discourse convention from distributional evidence alone.

**The period `。` in pretrain vs. nothing in SFT.** In the pretrain format, `1+1=2。` terminates with a Chinese period. In the SFT format, the answer has no trailing period. This prevents the model from learning to always emit `。` after the answer, which would interfere with RL reward parsing.

### Why `<` and `>` Instead of `[` and `]` or `{` and `}`

Angle brackets are chosen because:
- They are visually distinct from mathematical operators (`+`, `=`).
- In many markup languages (HTML, XML), angle brackets denote structure.
- They are unlikely to appear in the answer strings themselves (no addition problem produces `<` or `>`).

### The Single Grounding Sentence

The inclusion of `加法是把兩個數字合併成一個數。` serves a theoretical purpose beyond mere variety. In the framework of **Gricean maxims**, this sentence provides the model with a *definitional* statement about addition. The model can use this sentence to ground the symbol `+` in a linguistic concept. Without it, the model could learn a purely syntactic mapping: `+` triggers a lookup in a memorized table. With it, the model has a natural language description that might aid generalization — though whether this helps or distracts in a character-level model is an empirical question.

### Newline as Segment Boundary

Newlines separate examples in all three corpora. During training, the model learns to predict `\n` as a *reset signal*: after `\n`, the next character starts a new fact or Q&A pair. This implicitly teaches the model that contexts do not carry across newlines, which is essential for batched training where random sequence boundaries fall mid-example.

---

## 9. Summary of Design Principles

| Principle | Manifestation in gen_data.py |
|-----------|------------------------------|
| Match data complexity to model capacity | 81 problems, 4-layer model, d_model=128 |
| Perfect supervision for RL debugging | Deterministic arithmetic, rule-based reward |
| Controlled generalization test | 50/31 seed-fixed split, no contamination |
| Token-economy awareness | Character-level tokenization → compact formats |
| Reproducibility as a design goal | Fixed seed 42, no randomness in split |
| Minimalism before complexity | Single operation (addition), single format per stage |
| Signal-to-noise ratio | No distractors, no irrelevant tokens, no incorrect labels |

These principles make the mini-llm pipeline an effective testbed for studying how data design choices at the earliest stage (data generation) propagate through to RL training outcomes. The synthetic data is not a simplification of reality — it is an **idealization** that isolates the variables of interest.
