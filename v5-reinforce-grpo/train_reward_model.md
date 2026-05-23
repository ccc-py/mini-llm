# Reward Model Training: Theory and Background

## Overview

A Reward Model (RM) is a learned scoring function that maps a generated
sequence to a scalar reward signal. In reinforcement learning from human
feedback (RLHF), the RM substitutes for an expensive human evaluator,
providing dense, continuous feedback at each step of policy optimization.
`train_reward_model.py` implements a simple neural RM trained with
regression on synthetic math data — a minimal instantiation of this idea.

## Architecture: Shared Backbone, Minimal Head

The RM reuses the same Transformer backbone as the policy model:

- **Embedding layer**: `vocab_size → d_model` (weight-tied with policy
  embedding space).
- **4 Transformer layers**: identical to the policy model — RoPE rotary
  embeddings, RMSNorm, SwiGLU feed-forward, pre-norm residual design.
- **Score head**: a single linear layer `d_model → 1`, applied to the
  last token's hidden state.

The design choice is deliberate: sharing the backbone forces the RM to
operate on the same representation space as the policy, which means the
reward signal is grounded in what the policy "understands." The score
head is deliberately minimal — a single linear projection — ensuring
that reward prediction is a lightweight readout of the Transformer's
learned features, not a complex function that could overfit to spurious
patterns in the training data.

This mirrors the standard RLHF setup (InstructGPT, Llama-2) where the
RM is initialized from the same pretrained checkpoint and the reward
head is a shallow addition.

## Pretrained Initialization

The RM loads its backbone weights from `pretrain.pt` — the same
checkpoint used to initialize the policy model:

- `tok_emb.weight` — copied directly.
- `layers.*` — all 4 Transformer blocks loaded from the pretrained
  state dict.
- `norm.weight` — final RMSNorm parameters copied.
- `score_head` — randomly initialized (no pretrained correspondence).

This transfer has several theoretical advantages:

1. **Feature reuse**: Pretraining on the language modeling objective
   (next-token prediction on arithmetic fact strings) builds a
   representation that captures syntactic structure, numerical patterns,
   and positional relationships. The RM can reuse these features rather
   than learning them from scratch with only 644 samples.

2. **Accelerated convergence**: The optimization landscape starts near
   a good local minimum in the backbone parameter space. Only the score
   head and minor backbone adjustments are needed to specialize for
   reward prediction.

3. **Alignment with policy**: Since the policy also starts from the same
   `pretrain.pt`, the RM evaluates sequences in the same latent space
   that the policy generates from. This reduces distribution mismatch
   between training-time and inference-time inputs to the RM.

## Synthetic Label Generation

The training data is constructed automatically from 81 addition problems
(a, b ∈ [1,9], a+b):

- **Correct answers**: `prompt + correct_answer → label = 1.0`
- **Near-wrong answers**: `prompt + (correct ± delta) → label = max(0, 0.5 - 0.1 * |delta|)`
  for delta ∈ {±1, ±2} (subject to non-negative result). This gives
  scores like 0.4 (|delta|=1) and 0.3 (|delta|=2).
- **Far-wrong answers**: `prompt + (correct//2, correct*2, 99) → label = 0.0`
  (excluding duplicates).

Total: ~644 training samples from 81 problems.

This automated labeling strategy encodes a **distance-based reward
prior**: answers numerically close to the ground truth receive higher
scores, while obviously wrong answers score zero. It is a dense
proxy for correctness that the RM must learn to interpolate.

The approach has a key limitation: the label function is hand-designed
and monotonic by construction. The RM is trained to approximate this
specific function, not to learn human preferences or nuanced notions
of correctness. It is a **supervised regression** problem, not a
preference-learning problem (Bradley-Terry model, pairwise comparisons).

## MSE Regression Objective

The RM is trained with mean squared error (MSE) loss:

```
L = E_{(x, y)} [ (f_θ(x) - y)^2 ]
```

where x is the tokenized sequence (prompt + response), f_θ(x) is the RM
score, and y ∈ {0.0, 0.3, 0.4, 1.0} is the synthetic label.

MSE is the natural choice for scalar regression. It penalizes large
errors quadratically, incentivizing the RM to match both the ranking
(correct > near-wrong > wrong) and the exact score magnitudes.

In the standard RLHF framework, RMs are trained with pairwise ranking
loss (preference-based), not MSE. The ranking objective only requires
that the RM assign higher scores to preferred completions, ignoring the
absolute scale. MSE is stricter: it fixes the scale and forces the RM
to reproduce the specific label values. This is feasible here because
the synthetic labels are deterministic and known, but it means the RM's
score distribution is artificially constrained to [0, 1] by the
training data, which may not reflect the true range of response quality.

## Training Procedure

- **Optimizer**: AdamW, lr = 1e-4 (lower than the pretrain lr of 5e-4,
  reflecting that we are fine-tuning, not training from scratch).
- **Epochs**: 15 full passes over the 644 samples.
- **Batch size**: effectively 1 (sample-by-sample gradient updates).
- **Scheduler**: none (constant learning rate).

The 15 epochs are enough for the model to memorize the label function
on 644 points, given the low complexity of the task. In larger-scale
RLHF, RM training typically uses 1-2 epochs on much larger datasets
(100K+ preference pairs) to avoid overfitting. The long training here
reflects the toy-scale nature of the project — overfitting to the
training distribution is acceptable because the training data
exhaustively covers the (a,b) input space? Not exactly — see below.

## Why This RM Might Fail

### 1. Limited Training Data

644 samples from 81 arithmetic problems cover only the 1-digit addition
domain. The RM never sees multi-digit addition, subtraction, or any
text beyond the `<Q>a+b=?<A>answer` format. When the policy generates
out-of-distribution sequences during RL, the RM's predictions become
unreliable — it may assign high scores to gibberish or low scores to
valid arithmetic.

### 2. Narrow Score Distribution

The synthetic labels are confined to {0.0, 0.3, 0.4, 1.0} — just four
discrete values. The RM is trained to output in this narrow range.
During RL, the policy may generate responses that deserve scores
outside this range (e.g., a partially correct multi-step answer).
The RM has no basis for producing such values, leading to compression
artifacts.

### 3. Distribution Mismatch at RL Time

This is the most critical failure mode. The RM is trained on
deterministically constructed sequences (prompt + known answer). During
RL, it evaluates on-policy generations — sequences produced by the
current policy, which may contain novel tokens, repeated patterns,
formatting errors, or non-arithmetic text. These on-policy sequences
differ systematically from the training data, and the RM backbone
(initialized from language modeling) was not fine-tuned on such
generations. The score head may extrapolate wildly.

### 4. Single-Token Readout

The RM uses only the last token's hidden state for scoring. For
longer generations, information about the full response must be
compressed into a single vector. This is a representational bottleneck
that limits the RM's ability to assess multi-token reasoning chains.

### 5. No Calibration

MSE loss minimizes average squared error but provides no guarantees
about score calibration. A score of 0.6 from the RM does not mean "60%
chance the answer is correct" — it is an uncalibrated linear readout.
Different RL methods treat these scores differently: REINFORCE(RM) uses
them as raw advantage signals, while GRPO normalizes within a group
(removing scale bias).

## Summary

`train_reward_model.py` builds a minimal neural RM by transferring a
pretrained language model backbone and adding a linear score head. The
MSE regression objective, automated label generation, and shared
architecture mirror the mechanics of larger-scale RLHF systems. However,
the limited data, narrow score distribution, and distribution mismatch
with on-policy generations make this RM a potential source of
reward hacking and performance collapse — which is precisely what the
`compare.py` evaluation table is designed to diagnose.
