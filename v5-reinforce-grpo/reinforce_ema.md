# REINFORCE with EMA Baseline — Theoretical Background

## 1. The Policy Gradient Theorem

Let a policy $\pi_\theta(a|s)$ parameterize a probability distribution over actions $a$ given state $s$. The objective is expected cumulative reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} r_t \right]$$

where $\tau = (s_0, a_0, r_0, s_1, \dots)$ is a trajectory sampled under $\pi_\theta$. The gradient of $J$ with respect to $\theta$ is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t \right]$$

where $R_t = \sum_{k=t}^{T} r_k$ is the *return* (cumulative future reward). This result, the **Policy Gradient Theorem** (Sutton et al., 1999), follows from the log-derivative trick:

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

Substituting into $\nabla_\theta J(\theta)$ and expanding the expectation over trajectories yields the expression above. The key insight: the gradient scales the log-probability of each action by the return that followed it — actions that led to good returns are made more probable; actions that led to poor returns are suppressed.

## 2. REINFORCE: The Simplest Policy Gradient

REINFORCE (Williams, 1992) is the Monte Carlo instantiation of the policy gradient. Rather than learning a value function or bootstrapping from estimates, it uses complete trajectory returns as the signal:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot R_{i,t} \right]$$

In the language-model setting of `reinforce_ema.py`, "states" are token histories, "actions" are next-token predictions, and the return $R$ is the episode reward (1.0 if the generated answer contains the expected answer, 0.0 otherwise). Since the reward is received only at the end of generation, every token in the sequence shares the same $R$. The gradient for a single prompt-response pair becomes:

$$\nabla_\theta J(\theta) \approx \nabla_\theta \left( \sum_{t} \log \pi_\theta(a_t|\text{context}_t) \right) \cdot R$$

which is exactly `loss = -torch.stack(log_probs).sum() * advantage` — the negative sign converts gradient ascent into minimization.

## 3. Why Baseline Subtraction Is Unbiased

A baseline $b$ can be subtracted from the return without introducing bias, provided $b$ does not depend on the action $a_t$:

$$\mathbb{E}_{a_t \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b \right] = b \cdot \sum_a \pi_\theta(a|s_t) \nabla_\theta \log \pi_\theta(a|s_t)$$

But $\sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = \sum_a \nabla_\theta \pi_\theta(a|s) = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0$.

Therefore:

$$\mathbb{E}_{a_t \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R_t - b) \right] = \mathbb{E}_{a_t \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t \right]$$

The expected gradient is unchanged *for any baseline $b$ that is constant with respect to the action*. The variance, however, changes dramatically. This is the core insight behind all advantage-based methods: we can freely subtract a baseline to lower variance without touching the expected gradient.

## 4. EMA Baseline Theory

The exponential moving average baseline in `reinforce_ema.py` is updated after each step:

```python
baseline = 0.95 * baseline + 0.05 * avg_reward
```

This is the update rule for an exponentially weighted moving average:

$$b_{k+1} = \alpha \, b_k + (1-\alpha) \, \bar{R}_k$$

where $\alpha = 0.95$, $b_0 = 0.1$, and $\bar{R}_k$ is the mean reward of the batch at step $k$. Expanding the recurrence:

$$b_k = (1-\alpha) \sum_{i=0}^{k-1} \alpha^{k-1-i} \bar{R}_i + \alpha^k b_0$$

The baseline is a convex combination of all past batch-average rewards, with exponentially decaying weights — recent steps matter most, distant steps are forgotten. This is distinct from a simple running average: $\alpha=0.95$ means that a reward from $n$ steps ago contributes a factor of $0.95^n$, halving after approximately $\ln(0.5)/\ln(0.95) \approx 13.5$ steps.

### Why $\alpha=0.95$?

The choice reflects a trade-off between responsiveness and stability. A higher $\alpha$ (e.g., 0.99) would produce a smoother baseline that adapts slowly — useful when rewards are noisy but stationary. A lower $\alpha$ (e.g., 0.8) would react quickly but jitter. With 500 steps and batch size 8, $\alpha=0.95$ provides an effective window of roughly 10-20 steps, matching the timescale over which the policy's performance meaningfully changes under $10^{-5}$ learning rate.

## 5. Advantage = Reward − Baseline: Making the Signal Meaningful

The advantage $A = R - b$ measures whether a given rollout outperformed the baseline expectation. When:

- $R > b$: positive advantage → all token log-probabilities in the sequence are *increased* (the loss becomes more negative → gradient ascent pushes prob up).
- $R < b$: negative advantage → all token log-probabilities are *decreased*.
- $R \approx b$: near-zero gradient → no update, since the policy already performs at baseline level.

This last property is crucial. Without a baseline, every correct answer (reward = 1.0) produces a positive gradient, even if the policy already answers correctly 90% of the time. The baseline dynamically adjusts so that the gradient shrinks as performance improves, preventing over-updating on common successes.

## 6. Rule-Based Reward: When Binary Is Sufficient

The reward function is minimal:

```python
def compute_reward(response, expected):
    if expected.strip() in response.strip():
        return 1.0
    return 0.0
```

This is a sparse, binary, rule-based reward. In the RL literature, such rewards are common when the task has an unambiguous ground truth (e.g., math answer matches, code compiles, game outcome). The binary scheme is justified when:

1. **Correctness is all-or-nothing**: partial credit is misleading. If the model generates the correct answer embedded in extra text, substring matching suffices.
2. **No reward shaping needed**: intermediate tokens do not need per-token credit because the final outcome cleanly determines quality.
3. **No reward model required**: unlike `reinforce_rm.py` which trains a neural reward model, the rule-based approach is zero-cost and perfectly consistent.

The limitation is that all incorrect answers are treated identically (reward 0.0), regardless of how close they are. This provides no learning signal to distinguish "almost right" from "nonsense." The binary reward works well here because the 31 unseen problems are structurally similar to the seen ones — the policy already knows the general format and simply needs to be pushed toward correctness.

## 7. Exploration-Exploitation via Multinomial Sampling

During training, tokens are sampled from the full vocabulary distribution:

```python
probs = F.softmax(logits_last, dim=-1)
token = torch.multinomial(probs, 1)
```

This is **on-policy exploration through stochastic sampling**. Unlike greedy decoding (always pick argmax), multinomial sampling preserves the possibility of choosing lower-probability tokens, which may lead to novel correct answers.

The exploration-exploitation balance is controlled implicitly by the policy distribution itself. As training progresses and certain token sequences receive positive advantage, their probabilities increase — the policy shifts from exploration toward exploitation. However, because the softmax distribution over a vocabulary of thousands of tokens rarely collapses to a delta, some exploration persists throughout training.

This contrasts with $\epsilon$-greedy (which explores uniformly at random) or entropy-bonus methods (which explicitly penalize peaked distributions). REINFORCE with multinomial sampling explores proportionally to the policy's current uncertainty, which is a natural exploration strategy for structured generation tasks.

## 8. On-Policy Nature: Why Fresh Samples Each Step

REINFORCE is an **on-policy** algorithm: the gradient estimate uses trajectories sampled from the *current* policy $\pi_\theta$. Once $\theta$ is updated, old trajectories become biased estimators because they were drawn from a different distribution $\pi_{\theta_\text{old}}$.

Concretely, the policy gradient theorem requires:

$$\mathbb{E}_{\tau \sim \pi_\theta} [ \nabla_\theta \log \pi_\theta(\tau) R(\tau) ]$$

The expectation is over $\pi_\theta$. If we reuse a trajectory sampled from $\pi_{\theta_\text{old}}$, we would need importance weighting:

$$\mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}} \left[ \frac{\pi_\theta(\tau)}{\pi_{\theta_\text{old}}(\tau)} \nabla_\theta \log \pi_\theta(\tau) R(\tau) \right]$$

which introduces additional variance and complexity if the two policies diverge. `reinforce_ema.py` avoids this entirely by regenerating `batch_size=8` full responses from the current model at every step. This is the defining characteristic of Monte Carlo policy gradient: the cost is computational (500 steps × 8 samples × up to 20 tokens per sample = ~80,000 forward passes), but the gradient estimates are unbiased for the current policy.

## 9. Dynamic Baseline Adjustment

The EMA baseline serves an additional adaptive role: as the policy improves (or degrades) during training, the baseline automatically tracks its performance level. Consider three regimes:

**Early training**: Starting from the SFT checkpoint, the policy has reasonable but not perfect accuracy on unseen problems. The initial baseline $b_0 = 0.1$ is deliberately low — probably below actual performance. This means early advantages are mostly positive, encouraging broad exploration toward correct answers.

**Mid training**: As accuracy rises to 30-50%, the baseline follows with a lag. The advantage for correct answers ($1.0 - b$, where $b \approx 0.3$ to $0.5$) remains positive but shrinks. Incorrect answers yield negative advantage ($0.0 - b \approx -0.4$), actively suppressing wrong token sequences.

**Late training / convergence**: If the policy reaches, say, 70% accuracy, the baseline hovers around 0.7. Correct answers contribute advantage $\approx 0.3$; incorrect answers contribute $\approx -0.7$. The gradient magnitude decreases overall, preventing the policy from overshooting or becoming overconfident.

This dynamic adjustment is critical: without a baseline, the gradient magnitude would be proportional to the absolute reward rather than the *relative* improvement, and the policy would never converge gracefully.

## 10. Theory of Variance Reduction in Policy Gradients

Policy gradient estimators suffer from high variance because:

1. **Credit assignment over long sequences**: the same return $R$ is attributed to every token, ignoring which tokens were actually responsible for the outcome. This is the "temporal credit assignment" problem.
2. **Sampling noise**: with batch size 8, the estimate $\frac{1}{8}\sum \nabla_\theta \log \pi_\theta \cdot A$ has high Monte Carlo variance.
3. **Compound variance**: the product of log-probability gradients (which can be large for rare tokens) and returns amplifies noise.

The EMA baseline addresses variance through **control variate** theory. A control variate is a correlated random variable with known expectation that is subtracted from the estimator. Here:

- The baseline $b$ is a scalar that estimates $\mathbb{E}[R]$.
- $R - b$ has lower variance than $R$ alone if $\text{Cov}(R, b) > 0$, which holds because $b$ is explicitly constructed to track $\mathbb{E}[R]$.
- The subtraction is unbiased because $b$ is independent of each individual action (it is computed from *past batches*, not the current action).

Quantitatively, if $\text{Var}(R) = \sigma^2$ and $\text{Cov}(R, b) = \rho \sigma \sigma_b$, then:

$$\text{Var}(R - b) = \sigma^2 + \sigma_b^2 - 2\rho\sigma\sigma_b$$

For a well-tuned baseline ($\rho \to 1$, $\sigma_b \to \sigma$), variance approaches zero. In practice, the EMA baseline achieves $\rho \approx 0.5\text{--}0.8$ depending on the stationarity of the reward distribution.

### Other variance reduction techniques not used here

- **Per-token advantage** (REINFORCE with leave-one-out, or advantage per position) would reduce variance by assigning different weights to different tokens. `reinforce_ema.py` uses the same advantage for all tokens, which is a deliberate simplification.
- **Larger batch sizes** (e.g., 32) would reduce variance but increase memory. Batch 8 is a pragmatic choice for CPU training.
- **Entropy regularization** would prevent premature collapse. Not used here, but the small learning rate ($10^{-5}$) serves a similar role by limiting how quickly the policy can change.

## 11. Summary of Design Choices in `reinforce_ema.py`

| Parameter | Value | Theoretical Rationale |
|-----------|-------|----------------------|
| $b_0$ | 0.1 | Conservative initial assumption; ensures early positive advantage |
| $\alpha$ | 0.95 | 13.5-step half-life; smooth tracking of nonstationary reward |
| Batch | 8 | Pragmatic variance/reward trade-off for CPU training |
| Steps | 500 | Sufficient for convergence with $10^{-5}$ LR on 31 problems |
| Max gen | 20 | Covers all expected answer lengths with margin |
| Reward | 1.0 / 0.0 | Sparse but unambiguous for substring-match correctness |
| Unseen only | 31 problems | Tests generalization; held out from SFT |
| Optimizer | AdamW ($10^{-5}$) | Conservative LR; AdamW handles sparse gradients well |
| Sampling | Multinomial | Natural on-policy exploration proportional to uncertainty |
| Gradient clip | 1.0 | Prevents single bad batch from destroying policy |

These choices instantiate the purest form of REINFORCE: Monte Carlo returns, EMA baseline for variance reduction, and on-policy sampling throughout training.
