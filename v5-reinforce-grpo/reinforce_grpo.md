# Group Relative Policy Optimization (GRPO) — Theoretical Background

## 1. The Policy Gradient Problem

Reinforcement learning from verifiable rewards (RLVR) fine-tunes a language model policy $\pi_\theta$ to maximize expected reward on a distribution of prompts. The fundamental challenge is **credit assignment**: given a binary reward (correct/incorrect) that arrives only at the end of a generated sequence, which token-level decisions contributed to the outcome?

The REINFORCE algorithm addresses this by computing the gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]$$

where $R(\tau)$ is the total (undiscounted) return for trajectory $\tau$. The critical weakness is that $R(\tau)$ is used directly — a reward of 1.0 is applied uniformly to every token, including those that were irrelevant or detrimental. This produces high-variance gradient estimates and slow convergence.

GRPO attacks this variance problem at its root: **group-relative advantage estimation**.

---

## 2. Why Group-Relative Advantage Replaces the Critic

PPO and A2C reduce variance by learning a **critic network** $V_\phi(s)$ that estimates the expected value of each state, producing an advantage $A(s,a) = R(s,a) - V_\phi(s)$. The critic acts as a **baseline**: it subtracts out the expected reward, so tokens are only reinforced when they outperform expectations.

GRPO eliminates the critic entirely by using a **per-prompt Monte Carlo baseline**. For a single prompt $p$, we sample $G$ complete responses $r_1, \dots, r_G$ from $\pi_\theta$ and compute rewards $R_1, \dots, R_G$. The advantage of response $i$ is:

$$A_i = \frac{R_i - \mu_R}{\sigma_R + \varepsilon}, \quad \mu_R = \frac{1}{G}\sum_{j=1}^G R_j, \quad \sigma_R^2 = \frac{1}{G}\sum_{j=1}^G (R_j - \mu_R)^2$$

The `reinforce_grpo.py` implementation uses `G=8` and computes exactly this:

```python
rewards_t = torch.tensor(group_rewards, device=device, dtype=torch.float)
adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
```

**Why this works**: For any fixed prompt, the group mean $\mu_R$ is a stochastic estimate of the same quantity a critic would learn — $\mathbb{E}[R | p]$. But unlike a critic, which requires a separate optimization loop and can lag behind the policy, the group mean is computed from current on-policy samples. It is always up-to-date and requires no learned parameters.

The group standard deviation $\sigma_R$ provides **automatic scaling**: if all $G$ responses achieve similar rewards (e.g., all 1.0 or all 0.0), advantages collapse toward zero and the policy update becomes small. When rewards vary, well-performing responses are pushed up and poor responses are pushed down. This self-normalizing property is particularly valuable for binary rewards (1.0/0.0), where the group advantage acts as a **per-batch ranking signal**.

---

## 3. Statistical Theory: The Per-Prompt Baseline

Consider the classic REINFORCE gradient with an arbitrary baseline $b$:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R(\tau) - b) \right]$$

The baseline $b$ does not bias the gradient as long as it does not depend on the action $a_t$:

$$\mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b \right] = b \cdot \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \right] = 0$$

The group mean $\mu_R$ is a valid baseline because it is computed from the entire set of $G$ responses — conditioned on the prompt $p$, it is independent of any single response's action choices. The optimal baseline (minimizing gradient variance) is the expected reward conditioned on the state, $\mathbb{E}[R | s]$. The group mean approximates this conditional expectation by Monte Carlo, providing near-optimal variance reduction without learning.

The standard deviation term $\sigma_R$ adds an additional benefit: it **normalizes advantage magnitudes** across different prompts. A prompt where all $G$ responses score 1.0 or all score 0.0 produces $\sigma_R \approx 0$, and the advantage collapses to zero (or is clamped by $\varepsilon$). The policy is not updated for already-solved or hopeless prompts — training effort concentrates on ambiguous cases.

---

## 4. Mathematics of Group Advantage

Let $\mathcal{R}_p = \{R_1, \dots, R_G\}$ be the reward set for prompt $p$. The standardized advantage is:

$$A_i = \frac{R_i - \bar{R}}{s_R}, \quad \bar{R} = \frac{1}{G}\sum_{i=1}^G R_i, \quad s_R = \sqrt{\frac{1}{G}\sum_{i=1}^G (R_i - \bar{R})^2}$$

Key properties:

1. **Zero mean**: $\sum_i A_i = 0$ — exactly half the responses get positive advantage, half negative (or zero). This creates a **constant-sum competition** within each group: improving one response's relative standing necessarily reduces another's.

2. **Unit variance**: $\text{Var}(A) \approx 1$ — the advantage signal has fixed scale regardless of the absolute reward range. This makes hyperparameter transfer across tasks more reliable.

3. **Information preservation**: For binary rewards (1.0/0.0), $A_i$ can take at most $G$ distinct values. With $G=8$, advantages are quantized into 9 levels (−1.51, −1.13, −0.76, −0.38, 0, 0.38, 0.76, 1.13, 1.51 for a 5/3 split, etc.). The quantization resolution increases with $G$.

4. **Variance of the advantage estimate**: The standard error of $\bar{R}$ as an estimate of $\mathbb{E}[R|p]$ scales as $\sigma_R / \sqrt{G}$. Larger $G$ gives a more reliable baseline, which reduces gradient variance. However, returns diminish: going from $G=4$ to $G=8$ halves the variance of the baseline, but going from $G=8$ to $G=16$ only reduces it by ~30%. The choice $G=8$ in `reinforce_grpo.py` strikes a practical balance between variance reduction and computational cost — each response requires a full forward pass of $max\_gen\_len = 20$ tokens.

---

## 5. KL Divergence as Trust Region

The GRPO loss in `reinforce_grpo.py` has two terms:

$$\mathcal{L}(\theta) = \underbrace{-\frac{1}{G} \sum_{i=1}^G \left( \sum_{t} \log \pi_\theta(a_{i,t} | s_{i,t}) \right) \cdot A_i}_{\text{policy gradient}} + \beta \cdot \underbrace{\frac{1}{G} \sum_{i=1}^G \text{KL}(\pi_\text{ref} \parallel \pi_\theta)_{\text{traj } i}}_{\text{KL penalty}}$$

The KL divergence is computed **per-token** and summed across each trajectory:

$$\text{KL}(\pi_\text{ref} \parallel \pi_\theta) = \sum_{t} \sum_{x} \pi_\text{ref}(x | s_t) \cdot \left( \log \pi_\text{ref}(x | s_t) - \log \pi_\theta(x | s_t) \right)$$

This is the **forward KL divergence**: it averages the log-ratio weighted by the reference probabilities, which penalizes $\pi_\theta$ for assigning high probability where $\pi_\text{ref}$ assigns low probability (mode-covering behavior) and is conservative about exploring new regions.

**Why KL, not clipping**: PPO clips the importance-sampled probability ratio to $[1-\epsilon, 1+\epsilon]$, creating a hard trust region. GRPO replaces the hard clip with a **soft KL penalty**. The KL penalty is more principled: it directly measures the information loss from deviating from the reference, and the coefficient $\beta$ controls the trade-off between reward optimization and policy conservatism.

---

## 6. The Role of $\beta$ and the Frozen Reference Model

The reference model $\pi_\text{ref}$ is frozen at the initial fine-tuned weights (loaded from `finetune.pt`). It serves as an **anchor**: the KL penalty prevents the policy from diverging too far from the pre-RL solution.

The coefficient $\beta = 0.04$ controls the **temperature of the trust region**:

- **Large $\beta$** (e.g., 0.1): The KL penalty dominates. The policy barely moves, learning slowly but safely. Useful when rewards are noisy or unreliable.
- **Small $\beta$** (e.g., 0.01): The policy is free to chase reward, potentially overfitting to spurious correlations or collapsing to a single high-reward pattern.
- **$\beta = 0.04$** : A moderate value. At this setting, a typical KL divergence of ~0.1–0.5 per token contributes 0.004–0.02 to the loss, roughly balancing the policy gradient term which typically ranges from −0.01 to +0.05 per step.

The KL penalty acts as **entropy regularization with a target distribution**: it prevents the policy from collapsing into a deterministic degenerate solution (which would have zero entropy but high KL to the reference). This is theoretically cleaner than simply adding an entropy bonus, which penalizes all sharp distributions equally regardless of whether they are sensible.

**Frozen vs. trainable reference**: Freezing $\pi_\text{ref}$ makes the KL penalty an **absolute anchor** — the policy is pulled back toward its initialization. If $\pi_\text{ref}$ were updated (e.g., as an exponential moving average of $\pi_\theta$), the KL penalty would only prevent rapid changes, allowing unbounded drift over time. The frozen reference prevents catastrophic forgetting of the supervised fine-tuning solution.

---

## 7. Why No Importance Sampling or Clipping

PPO's clipping and importance sampling are responses to **off-policy data**: PPO reuses old trajectories for multiple gradient steps, requiring the importance ratio $\pi_\theta(a|s) / \pi_{\theta_\text{old}}(a|s)$ to correct for the distribution mismatch. This ratio is clipped to $[1-\epsilon, 1+\epsilon]$ to prevent large updates from a single batch.

GRPO is **on-policy**: in `reinforce_grpo.py`, each training step samples $G=8$ new responses from the current policy, computes advantages, performs one gradient update, then discards the responses. There is no data reuse across steps. Consequently:

- The importance ratio is always 1 (the data is from the current policy), so importance sampling is unnecessary.
- Clipping is redundant because the KL penalty already prevents overly large updates within a single step.
- The $max\_gen\_len = 20$ constraint per response keeps per-step compute manageable despite on-policy sampling.

The practical trade-off: PPO's off-policy reuse is sample-efficient (one batch of responses feeds $K$ epochs of updates); GRPO's on-policy approach is **compute-efficient** (no repeated gradients on stale data, no ratio clipping to tune, no critic to train). For tasks where generating responses is cheap (short sequences, small models), on-policy is simpler and often equally effective.

---

## 8. GRPO vs. REINFORCE: Same Root, Different Branch

GRPO and REINFORCE share the same essential structure:

| Component | REINFORCE | GRPO |
|-----------|-----------|------|
| Gradient | $\nabla\log\pi \cdot R$ | $\nabla\log\pi \cdot A$ |
| Baseline | None | Group mean $\mu_R$ |
| Signal scaling | Raw $R$ | Z-score $(R - \mu_R)/\sigma_R$ |
| Regularization | None | KL penalty $\beta \cdot \text{KL}$ |

Both compute the policy gradient as $\mathbb{E}[\nabla\log\pi \cdot \text{(advantage signal)}]$. The critical differences are:

1. **REINFORCE uses $R$ directly**: For binary rewards, every correct answer gets gradient push of magnitude 1.0; every wrong answer gets push of magnitude 0.0. The lack of baseline means variance is proportional to the absolute reward scale.

2. **GRPO centers and scales**: By subtracting $\mu_R$, only the relative ranking within the group matters — a response that scores 1.0 in a group where all 8 responses score 1.0 gets zero advantage (no room for improvement). A response that scores 0.0 in a group where 7 of 8 score 0.0 also gets zero advantage (the policy is uniformly wrong — this may be a hard prompt; don't push).

3. **GRPO adds KL regularization**: REINFORCE with no regularization can prematurely converge to a low-entropy policy that exploits reward function glitches. The KL penalty provides a principled stopping mechanism.

In the limit $G \to \infty$, the group advantage converges to $A_i = (R_i - \mathbb{E}[R|p]) / \sqrt{\text{Var}(R|p)}$, which is exactly the **normalized advantage** that an optimal critic with perfect knowledge would provide. GRPO with finite $G$ is a Monte Carlo approximation to this ideal.

---

## 9. Group Size $G$ and Variance

The group size $G$ is the most important hyperparameter in GRPO. Its effect on gradient variance can be analyzed through the law of total variance.

The advantage $A_i$ depends on the random set $\mathcal{R}_p$. For a fixed prompt $p$ with true reward distribution $R \sim D_p$ (mean $\mu_p$, variance $\sigma_p^2$), the group advantage has approximate variance:

$$\text{Var}(A_i) \approx 1 + \frac{2}{G} \quad \text{(for large } G, \text{ normal approximation)}$$

The factor $2/G$ is the **overdispersion** from estimating the baseline from finite samples. As $G$ grows, $\text{Var}(A_i) \to 1$, and the advantage estimate converges to the true normalized advantage.

For binary rewards and small $G$, the advantage distribution is discrete. With $G=8$, the advantage depends on the number of correct responses $k$ in the group:

- $k=0$ or $k=8$: all advantages are 0 (no learning signal)
- $k=1$: $A_\text{correct} = \sqrt{7} \approx 2.65$, $A_\text{wrong} \approx -0.38$
- $k=4$: $A_\text{correct} \approx 1.0$, $A_\text{wrong} \approx -1.0$

When $k=0$ (all wrong), the advantage is zero — GRPO naturally skips prompts where the policy is uniformly incapable. When $k=8$ (all correct), advantage is also zero — the policy is already optimal for this prompt. This **automatic gating** is elegant: training effort concentrates on prompts where the policy produces **mixed results**.

$G$ also determines the **minimum signal-to-noise ratio** for a gradient update. With $G=8$, a prompt needs at least one correct and one incorrect response to generate non-zero advantages. In `reinforce_grpo.py` with binary rewards, roughly half the training steps may produce all-correct or all-wrong groups (depending on policy accuracy), effectively skipping some updates — this is by design, not an inefficiency.

---

## 10. Preventing Reward Hacking and Catastrophic Forgetting

Two failure modes plague RL fine-tuning of language models:

**Reward hacking**: The policy finds spurious patterns that maximize reward without solving the intended task. For example, a model might learn to repeat the answer multiple times, increasing the chance of a substring match in the `reinforce_grpo.py` reward function (`if expected.strip() in response.strip(): return 1.0`).

**Catastrophic forgetting**: The policy over-optimizes for the RL domain and loses general language capabilities.

GRPO addresses both through the KL penalty:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{PG} + \beta \cdot \text{KL}(\pi_\text{ref} \parallel \pi_\theta)$$

The KL penalty creates a **Lagrangian relaxation** of the constrained optimization problem:

$$\max_\theta \mathbb{E}[R] \quad \text{s.t.} \quad \text{KL}(\pi_\text{ref} \parallel \pi_\theta) \leq \delta$$

For any $\beta$, the solution is Pareto-optimal: you cannot improve expected reward without increasing KL divergence beyond what $\beta$ allows. The ratio $\beta = 0.04$ defines the marginal rate of substitution between reward and information loss.

For reward hacking: if the policy tries to exploit the substring-matching reward by repeating tokens, the KL divergence spikes (the reference model does not repeat), and the KL penalty term counteracts the reward gradient.

For catastrophic forgetting: $\pi_\text{ref}$ represents the full distribution of supervised fine-tuning. The KL penalty ensures the policy maintains support for all tokens and patterns the reference knows, even those unrelated to the RL task.

---

## 11. Why GRPO Works for Verifiable Rewards

GRPO is particularly well-suited for tasks with **deterministic, verifiable rewards** (math problems with known answers, code with unit tests, factual QA):

1. **Binary rewards are naturally handled**: With rewards in $\{0, 1\}$, the group advantage reduces to a function of the proportion correct within the group. The 0/1 scale means $\mu_R \in [0, 1]$ and $\sigma_R$ is maximized at $\mu_R = 0.5$, producing the strongest gradient signal for prompts at the policy's decision boundary.

2. **No reward model needed**: Unlike RLHF, which learns a reward model from human preferences (adding another source of bias and variance), GRPO computes rewards directly from the ground-truth verifier. The `compute_reward` function in `reinforce_grpo.py` is a simple substring check — deterministic, cheap, and perfectly consistent.

3. **Group advantage handles deterministic rewards**: When the reward function is deterministic given the response, the only source of variance is the sampling process itself. The group advantage isolates this sampling variance, providing clean gradients without the noise of learned reward models.

4. **Short generations**: Verifiable tasks often have short answers. With `max_gen_len = 20`, trajectories are brief, making on-policy sampling computationally feasible and keeping the variance of the per-token log-probability sum manageable.

5. **No mode collapse**: In generative tasks with open-ended rewards (e.g., summarization, creative writing), the KL penalty fights an uphill battle against reward pressure toward a single high-reward style. In verifiable tasks, many responses can achieve maximum reward through different correct reasoning paths — the KL penalty is not fighting the reward, and the policy can converge to a high-entropy correct solution.

---

## 12. Summary: The GRPO Design Philosophy

| Aspect | GRPO Choice | Rationale |
|--------|-------------|-----------|
| Advantage | Group-relative z-score | Eliminates critic, self-normalizing |
| Baseline | Group mean | Unbiased, on-policy, no learning |
| Trust region | KL penalty ($\beta$) | Soft constraint, principled |
| Reference | Frozen at SFT weights | Prevents catastrophic forgetting |
| Data usage | On-policy, single step | No importance sampling needed |
| Rewards | Rule-based verifier | Deterministic, cheap, aligned |
| Regularization | KL + frozen ref | Pareto-optimal reward/KL trade-off |

GRPO sits between vanilla REINFORCE (no baseline, no regularization) and PPO (critic, clipping, importance sampling, off-policy epochs). It strips away complexity by exploiting two properties of the RLVR setting: **verifiable rewards** (allowing group-based advantage without learned value functions) and **fast on-policy generation** (making importance sampling unnecessary). The result is an algorithm with minimal moving parts — $G=8$, $\beta=0.04$, $lr=10^{-5}$, no clipping, no critic — that nonetheless provides stable, principled policy optimization grounded in classical statistical theory.
