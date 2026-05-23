# Comparing RL Methods: Evaluation Theory

## Overview

`compare.py` reads three `metrics_*.json` files produced by the
REINFORCE(EMA), REINFORCE(RM), and GRPO training scripts and prints a
side-by-side table of seen and unseen accuracy before and after RL
training. Despite its simplicity, this table encodes the core evaluation
challenges of reinforcement learning for language models.

## Data Split: SFT Seen vs. RL Unseen

The 81 addition problems (a, b ∈ [1,9], a+b) are split into:

- **50 seen problems** — used for pretrain data, SFT (finetune.txt), and
  evaluation of "in-distribution" accuracy. The model has seen these
  exact arithmetic facts and QA pairs during training.
- **31 unseen problems** — held out from SFT (rl_unseen.txt). The model
  has only seen these during pretraining (as raw fact strings, not as
  QA pairs) and during RL training. They test generalization.

This split mirrors the standard train/test paradigm in supervised
learning, with an important twist: the RL algorithms train *on the
unseen problems*. The "unseen" evaluation measures both (a) whether RL
improves performance on the problems it was optimized for, and (b)
whether this improvement transfers (or regresses) on the original SFT
distribution.

## Why Measure Both Seen and Unseen Accuracy

### Seen Accuracy (In-Distribution)

Measures retention of the SFT-finetuned behavior. A drop in seen
accuracy indicates **catastrophic forgetting** — the RL update
overwrites the supervised knowledge. This is a well-known failure mode
in RL fine-tuning: the policy may collapse to a narrow set of rewarded
behaviors at the expense of broader capability.

### Unseen Accuracy (Generalization)

Measures whether the RL signal actually teaches the policy to solve
new problems. Because unseen problems share the same structure (addition
of two single digits), a good RL method should transfer the underlying
skill rather than memorizing specific answers.

The ideal outcome: high unseen accuracy **and** maintained (or minimally
degraded) seen accuracy.

## The Seen/Unseen Gap as a Generalization Metric

The difference between seen and unseen accuracy — or more precisely, the
ratio of improvement on unseen vs. seen — is a proxy for generalization
quality:

- **EMA baseline** (exponential moving average of past rewards) provides
  a learnable, stable advantage signal. It tends to produce smooth
  improvements that generalize, because the advantage is computed
  per-sample and the baseline adapts to the reward distribution.
- **GRPO** (group relative policy optimization) normalizes rewards
  within a group of G samples from the same prompt. This removes
  absolute scale and focuses on relative ranking, which can lead to
  more robust updates. The KL penalty (β * KL divergence against frozen
  reference) further constrains the policy from drifting too far.
- **RM-based reward** replaces the sparse correctness signal with a
  learned dense score. If the RM is poorly calibrated or out-of-distribution,
  the policy can **reward-hack**: find sequences that score high under
  the RM but are not actually correct, destroying both seen and unseen
  accuracy.

## Statistical Considerations

The evaluation uses 50 seen and 31 unseen pairs. At this sample size,
accuracy estimates have wide confidence intervals. For a binomial
proportion with N observations, the standard error is approximately
√(p(1-p)/N). At p ≈ 0.9 and N = 31, SE ≈ 0.054, giving a 95% CI of
roughly ±10.6 percentage points. Observed differences between methods
should be interpreted with this uncertainty in mind.

The small evaluation set also means that a single lucky or unlucky
generation (due to sampling temperature in the policy's multinomial
decoding) can shift reported accuracy by several points. The scripts
mitigate this by using a fixed evaluation set rather than random
sampling, but the inherent variance of autoregressive generation
remains.

## What the Comparison Table Reveals

The table reports four numbers per method:

| Method | Seen(before) | Seen(after) | Unseen(before) | Unseen(after) |
|--------|-------------|-------------|----------------|----------------|

Before RL, all methods start from the same `finetune.pt` checkpoint, so
initial seen accuracy should be identical (or nearly so, since
evaluation uses sampling). The "before" column establishes the baseline.

The "after" columns show the effect of each RL method. The pattern tells
a story:

- **If seen ↓ and unseen ↑**: the policy is trading off breadth for
  specialization on the RL task. Some forgetting is tolerable if the
  unseen gain is large.
- **If both ↓**: the RL signal is destructive. This is classic reward
  hacking or training instability.
- **If both ↑**: near-ideal — the RL signal is extracting a genuinely
  more capable policy.
- **If seen → and unseen ↑**: the RL improves generalization without
  hurting retention — the gold standard.

Together, these four numbers provide a compact diagnostic of each RL
method's tradeoffs between retention, generalization, and robustness
to reward signal quality.
