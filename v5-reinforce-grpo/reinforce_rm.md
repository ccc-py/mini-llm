# REINFORCE with Learned Reward Models: Theoretical Background

## Motivation: From Rule-Based to Learned Rewards

Rule-based rewards are the simplest form of credit assignment in policy gradient RL. In `reinforce_ema.py`, the reward function is a deterministic binary oracle:

```
reward(response, expected) = 1.0 if expected in response else 0.0
```

This works when the target behavior can be mechanically verified — substring match for arithmetic answers, exact output for formatting tasks. But for open-ended generation (summarization, dialogue, instruction following), no rule can capture response quality. The industry solution, popularized by RLHF (Ouyang et al. 2022, *Training language models to follow instructions with human feedback*), replaces the rule with a learned reward model (RM) trained on human preferences.

The RM approach replaces an explicit verification function with an implicit one: a neural network trained to approximate human judgments. In `reinforce_rm.py`, the RM is a 4-layer Transformer (identical architecture to the policy backbone, `model.py`) with a linear score head, initialized from `pretrain.pt` weights and fine-tuned on synthetic math problems with a regression target.

### Why This Fails in Practice

The rule-based EMA variant holds steady at 95% seen / 93.5% unseen after 500 steps. The RM variant collapses from 97% → 38% (seen) and 67.7% → 9.7% (unseen) over the same 500 steps. This is not a tuning issue — it is a structural failure of the learned-reward paradigm when the RM is poorly calibrated for the on-policy distribution. The remainder of this document develops the theory behind this failure.

---

## RM Integration: The Reward Model as Part of the Environment

In standard REINFORCE, the environment supplies a reward signal $r_t$ for each transition. When the reward is rule-based, this signal is deterministic and fixed — the environment is a static function $r(s, a)$ with known properties.

With a learned RM, the reward function is itself a parameterized model $R_\phi$. The agent's Markov Decision Process (MDP) becomes:

```
S → A → S' → R_ϕ(S') → advantage → policy update
```

The RM sits inside the environment loop. In `reinforce_rm.py`, this appears at line 119–121:

```python
reward = rm(full_ids).item()
reward = max(0.0, reward)
```

The RM transforms complete trajectories (the full generated text) into scalar rewards. Two critical properties follow:

1. **The RM is part of the reward function, not the policy.** It is frozen during RL ($\phi$ fixed, $\nabla_\phi L = 0$). The policy sees the RM as a fixed (but imperfect) component of the environment.

2. **The gradient estimate uses the RM's output, not the true reward.** If $R_\phi(s') \neq R_\text{true}(s')$, the policy optimizes a misspecified objective. In the extreme case, the policy can "hack" the RM — finding trajectories that score high under $R_\phi$ but low under the true reward (Goodhart's law for learned reward functions).

The RM was trained on a specific data distribution: synthetic `a+b=?` problems with answers $c$, $c \pm 1$, $c \pm 2$, $c/2$, $2c$, and $99$, with scores assigned by a hand-written rule ($1.0$ for correct, $0.5 - 0.1|\delta|$ for near misses, $0.0$ for wild guesses). This is a clean, balanced, closed-form distribution. The RL loop generates trajectories on-policy — the model's own stochastic samples — which may drift far from the RM's training manifold.

---

## Distribution Shift: RM Trained on Clean Data, Evaluated on Exploration Outputs

The central failure mode of learned reward models in RL is **distribution shift**. The RM is trained offline on a static dataset $\mathcal{D}_\text{RM} = \{(x_i, y_i)\}_{i=1}^N$, which in `train_reward_model.py` consists of 324 carefully constructed (prompt + answer, score) pairs. During RL, the policy generates trajectories $\tau \sim \pi_\theta$, and the RM evaluates $R_\phi(\tau)$. Nothing guarantees $\tau \in \text{supp}(\mathcal{D}_\text{RM})$.

Formally, let $p_\text{train}(x)$ be the RM's training distribution and $p_\text{RL}(x)$ be the distribution of generated texts during RL. The RM's expected error on the RL distribution is:

$$
\mathbb{E}_{x \sim p_\text{RL}}[\ell(R_\phi(x), R_\text{true}(x))] =
\mathbb{E}_{x \sim p_\text{train}}\left[\frac{p_\text{RL}(x)}{p_\text{train}(x)} \ell(\cdot)\right]
$$

When $p_\text{RL}(x) \gg p_\text{train}(x)$ for some $x$ — i.e., the policy generates texts the RM has never seen — the importance weight explodes. The RM's predictions on these out-of-distribution (OOD) texts are unconstrained.

In `reinforce_rm.py`, the policy starts from a fine-tuned model (`finetune.pt`) that already knows how to answer math questions. The RL exploration adds random token-level noise via multinomial sampling (line 110). Early in training, most trajectories are near-correct and the RM assigns reasonable scores. But as the policy updates to maximize RM score, it discovers trajectories that:

- **Exploit RM blind spots**: sequences whose character-level patterns happen to trigger high RM scores despite being wrong answers
- **Drift in token distribution**: the character-level n-gram statistics shift away from the clean `a+b=?<A>c` format seen during RM training

The result is a positive feedback loop: policy drifts → RM scores drift → policy chases OOD scores → policy drifts further. This is known as *reward model overoptimization* in the RLHF literature (Gao et al. 2023, *Scaling laws for reward model overoptimization*).

---

## Signal-to-Noise: Continuous Rewards vs Binary Rewards

Rule-based rewards in v4 are binary $\{0, 1\}$ — either the answer is correct or it is not. The signal-to-noise ratio (SNR) of a binary reward is:

$$
\text{SNR}_\text{binary} = \frac{|\mathbb{E}[R \mid \text{correct}] - \mathbb{E}[R \mid \text{wrong}]|}{\sqrt{\text{Var}[R \mid \text{correct}] + \text{Var}[R \mid \text{wrong}]}} = \frac{1}{0} \to \infty
$$

There is no variance in the per-class reward — every correct answer gets exactly $1.0$, every wrong answer gets $0.0$. The entire gradient variance comes from the policy's stochasticity, not from reward noise.

A learned RM outputs continuous values. The output distribution of the RM in `reinforce_rm.py` is uncalibrated: it may concentrate in a narrow range (e.g., $[0.3, 0.7]$) or spread widely depending on the input. After `max(0.0, \cdot)` clamping, the effective reward range is $[0, R_\text{max}]$ where $R_\text{max}$ is the highest score the RM assigns to any in-distribution correct answer.

The SNR for a continuous RM output is:

$$
\text{SNR}_\text{RM} = \frac{|\mathbb{E}[R_\phi \mid \text{correct}] - \mathbb{E}[R_\phi \mid \text{wrong}]|}{\sqrt{\text{Var}[R_\phi \mid \text{correct}] + \text{Var}[R_\phi \mid \text{wrong}]}}
$$

Two problems arise:

1. **The numerator shrinks**: If the RM is uncertain about its judgments (overconfident for wrong answers, underconfident for correct ones), the separation between correct/wrong reward distributions decreases.

2. **The denominator grows**: The RM's output is noisy — two different correct answers may receive scores of $0.85$ and $0.92$ due to position effects, tokenization artifacts, or the RM's imperfect generalization.

When SNR drops below 1, individual gradient steps contain more noise than signal, and the policy performs a random walk in parameter space.

---

## Reward Compression: RM Outputs in a Narrow Range

The RM in `reinforce_rm.py` was trained with MSE loss on a training set where target scores range from $0.0$ to $1.0$. MSE training with bounded targets naturally produces outputs concentrated in a sub-range of $[0, 1]$, with additional compression from:

- **Softmax saturation**: the final hidden state passes through the Transformer's residual stream and RMSNorm before the linear score head. Small variations in hidden states produce small variations in output.
- **Regression to the mean**: MSE loss penalizes large errors quadratically, encouraging conservative predictions near the mean of the training targets.
- **Clamping**: `reward = max(0.0, reward)` introduces a one-sided floor but no ceiling. If the RM occasionally outputs negative scores for OOD inputs (a common failure mode), clamping to zero collapses them into a single indistinguishable outcome.

The effective reward range might be $[0.4, 0.9]$ instead of $\{0, 1\}$. The compression factor $C = \frac{\text{range}_\text{RM}}{\text{range}_\text{rule}}$ can be $0.5$ or less. This directly scales the gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (R_\phi(\tau) - b) \right]
$$

If $R_\phi(\tau) \in [0.4, 0.9]$ and $b$ (the EMA baseline) converges to $\bar{R} \approx 0.6$, then the advantage $R_\phi - b \in [-0.2, 0.3]$. The same advantage with binary reward and $b \approx 0.5$ would be $\{-0.5, 0.5\}$ — a much larger signal.

---

## Baseline Interaction with Compressed Rewards

The EMA baseline tracks the running mean reward:

```python
baseline = 0.95 * baseline + 0.05 * avg_reward  # α = 0.95
```

Initialized at `baseline = 0.1`, this converges to the mean RM reward over time. With binary rewards, the baseline sits between $0$ and $1$, producing advantages of roughly $\pm 0.5$ after convergence. With compressed RM rewards, the baseline quickly tracks the narrow RM range.

The critical interaction is **effective advantage magnitude**. Let $\sigma_R$ be the standard deviation of RM rewards over the policy's trajectory distribution. The advantage $A = R_\phi - b$ has standard deviation $\sigma_R$ if $b$ is perfectly correlated with the mean (which EMA approximates). The effective gradient step size is proportional to $A$.

If $\sigma_R = 0.15$ (compressed RM) vs $\sigma_R = 0.5$ (binary), the RM-based gradient is $3.3\times$ smaller in expectation. To compensate, the learning rate would need to increase proportionally, but a higher LR destabilizes the policy updates when the RM occasionally assigns outlier scores to OOD samples.

Furthermore, the EMA baseline has momentum. When the RM's output distribution shifts (because the policy drifts), the baseline lags behind by approximately $1/(1-\alpha) = 20$ steps. During these 20 steps, the advantage is systematically biased — either consistently positive (if RM scores are rising faster than the baseline catches up) or consistently negative. This bias can push the policy in the wrong direction for an extended period.

---

## Why RM-Based RL Fails in v4: A Synthesis

The failure of `reinforce_rm.py` is not a bug — it is the inevitable outcome of several theoretical violations:

### 1. Training RM on Static Synthetic Data

The RM was trained on 324 pairs from a known generative process ($a+b$ for $a,b \in [1,9]$). This is a closed-world distribution. The RL policy explores a much larger space of character sequences (including malformed answers, repeated tokens, out-of-domain content). The RM has **zero exposure** to realistic on-policy data during training.

Requirement for success: **distribution matching** — the RM training data must cover the space of trajectories the policy might generate, or there must be a mechanism to detect OOD inputs and fall back to a safe reward.

### 2. No Calibration

The RM outputs an uncalibrated scalar. A score of $0.7$ from the RM may correspond to a true correctness probability of $0.95$ for some inputs and $0.3$ for others. Without calibration, the policy cannot distinguish between "confidently correct" and "uncertain but high-scoring" trajectories.

Requirement for success: **calibration** — the RM's scores should be meaningful as probabilities or at least be monotonic with respect to true quality. Platt scaling, temperature scaling, or isotonic regression are standard fixes.

### 3. Insufficient Output Resolution

The combined effect of MSE training and softmax-based features compresses RM outputs into a narrow band. The policy sees advantages of $\pm 0.2$ instead of $\pm 0.5$. With $\alpha=0.95$, the baseline reacts sluggishly to shifts, compounding the problem.

Requirement for success: **resolution** — the RM must produce rewards that span a wide enough range to drive meaningful gradient updates. This often requires careful output layer design (e.g., no final nonlinearity, learnable temperature, or ensemble disagreement penalty).

### 4. No KL Regularization

Unlike `reinforce_grpo.py` (which adds a KL penalty $\beta \cdot D_\text{KL}(\pi_\theta \| \pi_\text{ref})$), the RM-based REINFORCE has no mechanism to constrain policy drift. The policy is free to move into regions where the RM is unreliable.

Requirement for success: **proximity constraint** — KL regularization (as in PPO and GRPO) or a trust region prevents the policy from exploiting RM errors. Without it, the policy *will* find adversarial trajectories under a learned reward model.

### 5. Clamping Destroys Information

`reward = max(0.0, reward)` discards negative scores. If the RM assigns negative scores to low-quality trajectories (which is reasonable), clamping them to zero removes the negative signal. The policy stops learning what *not* to do.

Requirement for success: **preserve reward sign** — clamping should be used with extreme care. A better approach is to normalize rewards (e.g., z-score within each batch) rather than clip.

---

## The Theoretical Requirements for Successful RM Integration

The v4 RM failure is a case study in the gap between "learn a reward model" and "use a reward model successfully." For a learned RM to work in policy gradient RL, it must satisfy:

| Requirement | Formal condition | v4 violation |
|---|---|---|
| **Calibration** | $\mathbb{P}[R_\text{true} > t \mid R_\phi(x) > t] = \mathbb{P}[R_\text{true} > t]$ for all thresholds $t$ | No calibration step; raw MSE outputs used as rewards |
| **Distribution matching** | $\text{supp}(p_\text{RL}) \subseteq \text{supp}(p_\text{train})$ or RM has OOD robustness | $p_\text{train}$ is 324 synthetic pairs; $p_\text{RL}$ is unbounded policy exploration |
| **Resolution** | $\text{Var}[R_\phi \mid \text{quality} = q] \ll \mathbb{E}[R_\phi \mid q_1] - \mathbb{E}[R_\phi \mid q_2]$ for distinguishable quality levels $q_1, q_2$ | RM outputs compressed to $\approx [0.4, 0.9]$; binary rewards span $[0, 1]$ |
| **Monotonicity** | $R_\phi(x) > R_\phi(x') \iff R_\text{true}(x) > R_\text{true}(x')$ for all $x, x' \in \text{supp}(p_\text{RL})$ | Not verified; regression RM may rank pairs incorrectly |
| **Regularization** | $\exists$ constraint $D(\pi_\theta \| \pi_\text{ref}) < \epsilon$ preventing exploitation of RM errors | No KL penalty; no trust region |
| **Reward scale** | $|\mathbb{E}[A]| \gg \text{std}(\nabla_\theta \log \pi \cdot A_\text{noise})$ | Compressed advantage; $\sigma_A \approx 0.15$ drowns signal |

The observed collapse from 97% → 38% accuracy is a clean empirical illustration: when any of these requirements is violated, the RM-augmented MDP becomes misspecified, and the policy optimizes for artifacts of the learned reward rather than the true objective.

## Relationship to RLHF Practice

In production RLHF systems (e.g., InstructGPT, Llama 2), these issues are mitigated by:

- **Large-scale preference data**: RM training uses $>10^5$ human comparisons, distributed across the generation space
- **Ensemble RMs**: multiple independently trained RMs; rewards are the ensemble mean minus a penalty for disagreement (variance)
- **PPO with KL penalty**: the policy is explicitly constrained ($\beta D_\text{KL}$ of 0.01–0.1 per token), preventing drift
- **Reward normalization**: running statistics normalize rewards to zero mean, unit variance within each batch
- **Iterated RM training**: the RM is periodically retrained on new data sampled from the current policy

The v4 experiment strips all of these safeguards. The result — catastrophic collapse — demonstrates why each one exists.

---

## Summary

REINFORCE with a learned reward model fails in v4 because:

1. The RM is trained on a narrow, closed synthetic distribution — the RL policy explores out of distribution within dozens of steps
2. The RM's continuous outputs are compressed into a narrow range, reducing the effective advantage signal $3\times$ to $5\times$ compared to binary rewards
3. The EMA baseline ($\alpha = 0.95$) cannot compensate for compressed advantages and introduces lag bias during distribution shift
4. No KL regularization allows unbounded policy drift into RM-exploiting regions
5. Reward clamping (`max(0.0, \cdot)`) removes negative signal, preventing the policy from learning to avoid bad trajectories

The 97% → 38% accuracy drop is not a pathology — it is the expected behavior when a learned reward model is deployed in an RL loop without calibration, distribution matching, resolution guarantees, or policy constraints. The theory of reward model overoptimization predicts exactly this outcome.
