# Supervised Fine-Tuning (SFT) — Theoretical Background

## Overview

Supervised Fine-Tuning (SFT) is the second stage in the modern LLM training pipeline: after a model learns broad language statistics during pretraining, SFT adapts it to follow a specific format or task. In mini-llm v4, `finetune.py` loads the pretrained checkpoint (`pretrain.pt`), trains for 300 steps on Q&A-formatted data (`finetune.txt`), and produces `finetune.pt` — the starting point for all three REINFORCE variants. The core design decisions (300 steps, lr=1e-4, batch_size=32, full-parameter update) each reflect deliberate tradeoffs between adaptation speed, forgetting prevention, and downstream RL compatibility.

## SFT as Distribution Matching

Pretraining learns the unconditional distribution of natural text: `P(token_t | token_<t)`. The model becomes good at predicting what character comes next in general Chinese text. SFT shifts this distribution toward a conditional task distribution: `P(answer | <Q> question <A>)`.

This is distribution matching under a domain shift. The pretraining distribution `D_pretrain` (narrative sentences about wuxia characters) and the finetuning distribution `D_finetune` (Q&A pairs with special tokens) are not the same — they share vocabulary and basic language structure, but differ in format, token co-occurrence patterns, and conditional dependencies. SFT moves the model's internal distribution from `D_pretrain` toward `D_finetune` by continuing gradient descent on the new domain.

From an information-theoretic perspective, SFT minimizes the **KL divergence** between the model's distribution and the empirical finetune distribution:

```
θ* = argmin_θ KL(D_finetune || P_θ) = argmin_θ E_{x~D_finetune}[-log P_θ(x)]
```

This is equivalent to maximizing the log-likelihood of the finetune data under the model. The KL divergence decomposes into two terms: `KL(D_finetune || P_θ) = H(D_finetune, P_θ) - H(D_finetune)`. Since `H(D_finetune)`, the entropy of the finetune data, is a constant (determined by the dataset), minimizing cross-entropy is the same as minimizing KL divergence. The model is not being asked to match the pretrain distribution anymore — it is being pulled toward a new target distribution, and the pretrained weights determine where this gradient descent trajectory begins.

The loss spike observed at finetuning step 0 (from ~0.22 at end of pretrain to ~6.41, well above the random baseline of ln(112) ≈ 4.72) is direct evidence of this distribution mismatch. The model's parameters encode conditional probabilities that assign near-zero likelihood to sequences starting with `<Q>`, because such sequences never appeared in pretraining. SFT must overwrite these priors, and the magnitude of the loss spike (6.41 - 4.72 = 1.69 nats above uniform) quantifies how surprised the pretrained model is by the finetune format.

## Full Fine-Tuning vs. Parameter-Efficient Methods

`finetune.py` performs **full fine-tuning**: all model parameters are updated. This contrasts with parameter-efficient methods like LoRA (Low-Rank Adaptation) or adapter layers, which freeze the base weights and insert small trainable modules.

Full fine-tuning is appropriate here for several reasons:

- **Model scale**: With ~400K parameters, the cost of storing and updating all weights is negligible. LoRA's memory advantage (no need to store full gradient states for frozen params) only matters at scales above ~100M parameters.
- **Representation shift**: The gap between pretrain and finetune distributions is large enough that freezing lower layers would limit adaptation. The model must learn that `<Q>`, `<A>`, and `？` now carry structural meaning — this requires updating embeddings and early-layer representations.
- **Downstream RL compatibility**: All three REINFORCE scripts (`reinforce_ema.py`, `reinforce_rm.py`, `reinforce_grpo.py`) load `finetune.pt` and continue training. Full fine-tuning ensures the RL stage starts from a model whose full parameter space is already aligned to the task, giving RL maximum flexibility.

If LoRA were used with rank r=8, the trainable parameters would be roughly `2 × d_model × r × n_layers × 2` (for query and value projections in each transformer block) ≈ 2 × 128 × 8 × 4 × 2 = 16,384 parameters. This is 4% of the total parameter count. At this scale, the representational capacity of the adapter might be insufficient to learn the new conditional distribution induced by `<Q>` and `<A>` tokens — the embedding layer alone (112 × 128 = 14,336 parameters) accounts for nearly all LoRA's budget.

## Weight Tying During Finetuning

The `ModernLanguageModel` uses **weight tying**: the token embedding matrix and the output projection matrix share the same parameters (`self.tok_emb.weight = self.output.weight`). During finetuning, this shared matrix receives gradients from two paths:

1. **Embedding path**: gradient flows from the input layer, through the embedding lookup, into the transformer layers.
2. **Output path**: gradient flows from the cross-entropy loss, through the output projection, into the shared matrix.

The effective gradient on the shared weight matrix at each step is the sum of the embedding gradient and the output gradient. This coupling has implications for finetuning dynamics:

- **Regularization through parameter sharing**: The same parameters must serve both as input representations and as output classifiers. Changes that improve output logits must also preserve useful input embeddings, creating a natural constraint that limits overfitting.
- **Accelerated format learning**: The output projection directly sees the gradient signal for predicting `<Q>`, `<A>`, and answer tokens. This gradient simultaneously updates the embeddings of these tokens, speeding up the acquisition of format-specific representations.

Without weight tying, the embedding layer would only be updated via backpropagation through the transformer layers — a longer gradient path that dilutes the format-specific signal. Weight tying effectively short-circuits this path, giving finetuning a 2× gradient signal for the shared parameters.

## Why Small-Data Fine-Tuning Works After Pretraining

The finetune dataset in mini-llm is small — roughly 30,000–50,000 characters, equivalent to perhaps 500–800 Q&A pairs after shuffling and concatenation. Training on this from scratch (random initialization) would fail. But after pretraining, it works. The reason is **representation reuse**.

Pretraining teaches the model:

1. **Character-level statistics**: The transition probabilities between Chinese characters (e.g., `郭` → `靖`, `降` → `龍` → `十` → `八` → `掌`).
2. **Grammatical structure**: Subject-verb-object ordering, common syntactic patterns.
3. **Semantic clustering**: Characters that appear in similar contexts develop similar embeddings.
4. **Attention patterns**: The causal self-attention heads learn to track positional relationships and long-range dependencies.

When finetuning begins, these representations are already in place. The model doesn't need to relearn what characters are, how Chinese grammar works, or how to attend over a sequence. It only needs to learn a new **mapping**: given `<Q>...<A>`, the next tokens should follow answer patterns rather than narrative patterns.

This is precisely what transfer learning theory predicts: pretraining provides a good initialization in parameter space, and finetuning with a low learning rate explores a local neighborhood of that initialization to find parameters that work for both the old distribution (approximately) and the new one. The Hessian of the loss at the pretrained minimum has many near-zero eigenvalues (flat directions) that correspond to features irrelevant to the pretask; finetuning can move parameters along these directions without increasing pretrain loss significantly.

## Catastrophic Forgetting

The primary risk during finetuning is **catastrophic forgetting** — the model overwrites its pretrained knowledge while adapting to the new task. In mini-llm, this would manifest as the model learning to produce Q&A text but losing the ability to form coherent Chinese sentences.

`finetune.py` mitigates this through two design choices:

1. **Lower learning rate** (1e-4 vs 5e-4 for pretrain): Smaller parameter updates per step reduce the distance traveled in weight space, preserving pretrained representations while still allowing adaptation.
2. **Limited training steps** (300 vs 500 for pretrain): The model takes fewer total gradient steps, reducing the cumulative parameter drift.

The learning rate ratio (finetune_lr / pretrain_lr = 0.2) is consistent with the ULMFiT prescription of using a learning rate roughly 1/5 to 1/10 of the pretrain rate for discriminative fine-tuning. At 1e-4 with AdamW, typical parameter updates are on the order of `||Δθ|| ≈ lr × ||g||` where `||g||` is the gradient norm (clipped to 1.0). Over 300 steps, the total parameter movement is bounded, and the model retains its language capabilities.

If the pretrain learning rate (5e-4) were used for finetuning, each step would move parameters 5× further. The model would rapidly overfit to the Q&A format and lose general language ability — generating repetitive loops or failing on any prompt that deviates from the exact training patterns.

The **elastic weight consolidation (EWC)** perspective offers another lens: the most important parameters for pretraining are those with high Fisher information. Finetuning with a low learning rate naturally moves less in these high-importance directions (because gradient magnitude tends to be smaller in well-minimized directions), providing implicit protection against catastrophic forgetting without requiring explicit regularization.

## The Role of Gradient Clipping

Both `pretrain.py` and `finetune.py` apply gradient clipping at norm 1.0. During finetuning, gradient clipping serves a particularly important role. The initial steps of finetuning produce very high loss values (~6.41), which generate correspondingly large gradients — the model is very "surprised" by the finetune data, and the gradient magnitude scales roughly linearly with the loss magnitude. Without clipping, the first few gradient updates would be enormous, potentially destroying the pretrained representations in a single step.

The gradient norm threshold of 1.0 ensures that no single batch can cause catastrophic parameter movement. This is especially important given that `get_batch` randomly samples 32 sequences of length 64 from the finetune data. A batch that happens to contain many `<Q>` tokens at unfamiliar positions could produce an outsized gradient. Clipping limits the per-step update to `max(||g||, 1.0)`, so even in the worst case, the parameter change is controlled.

## Bridging Raw Language Modeling and Task-Specific Behavior

Pretraining optimizes for next-token prediction — a general objective that captures linguistic competence but not task compliance. A pretrained model can continue a prompt like `<Q>1+1=？<A>` with anything: it might output `2`, but it might equally output another question, a narrative about arithmetic, or random characters. All are valid continuations under the pretraining distribution because the model has never seen this format.

SFT bridges this gap by **conditioning the model on task structure**. After 300 steps, the model internalizes:

- The `<Q>` token signals "a question follows"
- The `<A>` token signals "the answer follows"
- The model should complete the answer and stop (or prepare for the next Q&A pair)

This is not semantic understanding in the human sense — the model doesn't "know" that 1+1=2. Rather, it has learned a conditional pattern: the character sequence after `<Q>1+1=？<A>` is likely `2`, because that pattern appeared repeatedly in training. The model is performing **pattern completion conditioned on format tokens**.

This distinction matters: if we tested the model on `<Q>999+1=？<A>` (a question format it has seen but with numbers never seen in combination), the model might output `1000` (if it has generalized the addition pattern) or `100` (if it has simply memorized that the answer has a `1` followed by zeros). The latter is format compliance without arithmetic understanding — SFT has successfully taught the model the Q&A format, but not necessarily the underlying computation.

## Loss Landscape: Pretraining vs. Finetuning

The loss landscape during pretraining is relatively smooth: the model moves from high loss (~4.89, near uniform distribution over 112 characters) to low loss (~0.22) over 500 steps. The landscape during finetuning is qualitatively different:

1. **Initial spike**: Step 0 of finetuning shows loss ~6.41 — higher than random. This is because the model assigns near-zero probability to the first token of the finetune sequence (`<Q>`), and cross-entropy loss = -log P(token) approaches infinity as P(token) → 0. The loss for the first batch is dominated by these extremely low-probability tokens.

2. **Rapid initial descent**: Within 50–100 steps, loss drops from ~6.41 to ~0.5. This is much faster than pretraining's initial descent (which took ~100 steps to reach 0.27), because only the output distribution needs to shift — the underlying representations are already good.

3. **Lower final loss**: Finetuning often reaches a lower final loss (~0.21) than pretraining (~0.22). This is partly because Q&A data has lower entropy (more predictable patterns) and partly because of overfitting.

The loss landscape analogy: pretraining is like sculpting a rough shape from a block of stone (many large chisel strikes), while finetuning is like adding fine details to an existing sculpture (small, precise adjustments).

A deeper mathematical perspective: the loss during pretraining can be modeled as following a relatively convex trajectory in parameter space, moving down a basin toward a minimum. The finetune loss landscape is better thought of as a **perturbed landscape** — the pretrained minimum is at `θ_pretrain`, but the finetune loss function `L_finetune(θ)` has its minimum at a different location `θ_finetune*`. The observed training trajectory moves from `θ_pretrain` toward `θ_finetune*`, but the number of steps (300) and learning rate (1e-4) prevent it from reaching `θ_finetune*` exactly. The model ends up at an intermediate point that balances low finetune loss with proximity to the pretrained minimum — a compromise that helps downstream RL.

## Optimizer Choice: AdamW in Finetuning

`finetune.py` uses AdamW with the default betas (β1=0.9, β2=0.999, ε=1e-8) and no weight decay specified (PyTorch default weight_decay=0). The absence of explicit weight decay during finetuning is notable: weight decay acts as L2 regularization that penalizes large weights and can help prevent overfitting. Not using it means the model has weaker protection against memorizing the finetune data.

This choice is defensible for two reasons:

1. **The finetune run is short** (300 steps). Weight decay's effect accumulates over many steps; in 300 steps, the implicit regularization from early stopping dominates.
2. **The model is small** (~400K params). At this scale, the gap between memorization and generalization is narrow — the model's limited capacity acts as an implicit regularizer.

AdamW's adaptive learning rates per parameter are especially beneficial during finetuning because different layers require different update magnitudes. The attention layers near the output need to adapt to the `<Q>/<A>` format quickly (their gradients are large during the initial steps), while the embedding layer and early transformer blocks should change more slowly (their gradients are smaller and their representations are more general). AdamW's per-parameter normalization automatically provides this differential learning rate behavior.

## Why SFT Alone Is Insufficient (Motivating RL)

`finetune.pt` is not the final output of v4 — it is the **input** to three REINFORCE variants. SFT alone has fundamental limitations that motivate the RL stage:

1. **Loss-function mismatch**: SFT minimizes cross-entropy on each token independently. A model that assigns 60% probability to the correct answer token and 40% to wrong tokens has fairly low cross-entropy, but it will still generate wrong answers 40% of the time during sampling. SFT doesn't directly optimize for "always output the correct answer."

2. **No exploration**: SFT trains on a fixed dataset. The model never gets to try out different responses and learn from their outcomes. If the training data always says `1+1=2`, the model learns to say `2` — but it could benefit from also learning why `3` is wrong.

3. **Exposure bias**: During SFT training, the model conditions on ground-truth previous tokens (teacher forcing). During generation, it conditions on its own predicted tokens. This mismatch (train-time vs. test-time distribution) means errors compound during autoregressive generation. RL, by training on self-generated sequences, closes this gap.

4. **No reward signal**: SFT doesn't know whether the answer is correct or not. The loss for predicting `2` after `1+1=？<A>` is identical whether `2` is the right answer or the wrong one — it only measures how well the model matches the training label. RL provides a scalar reward: correct answers get positive feedback, wrong answers get negative feedback, directly optimizing the quantity we care about.

5. **Overconfidence calibration**: SFT tends to produce models that are overconfident in their predictions, assigning high probability to the training answer even when unsure. RL with a reward signal can recalibrate: if the model outputs the wrong answer, the negative advantage reduces the probability of that token, producing a more calibrated distribution over answers.

## Format Learning Theory: Special Tokens

The special tokens `<Q>` and `<A>` serve as **structural markers** that partition the sequence into functional roles:

- `<Q>` (question delimiter): signals the start of a query
- `<A>` (answer delimiter): signals the boundary between query and response

These tokens function similarly to control tokens in instruction-tuned models (e.g., `<|im_start|>`, `<|user|>`, `<|assistant|>` in ChatML). They are not semantically meaningful characters but rather **format governors** that restructure the model's conditional generation.

From the model's perspective, these tokens are simply characters in the vocabulary with indices like any other. Their special role emerges purely from their co-occurrence statistics: `<Q>` is always followed by known question patterns and eventually `<A>`, which is always followed by known answer patterns. The attention heads learn to use the presence of `<Q>` at a given position as a query signal, attending differently to tokens before and after `<A>`.

After 300 steps of finetuning, the model's internal representations encode a **three-phase generation policy**:

1. After seeing `<Q>`, attend broadly to question content (the characters between `<Q>` and `<A>`)
2. After seeing `<A>`, switch to answer mode — generate the response token
3. After completing the answer, transition to generating the next `<Q>` (since training data is concatenated Q&A pairs)

This structured behavior is entirely learned through gradient descent on the cross-entropy loss — no explicit policy or rule is encoded. The attention patterns across the 4 layers evolve during finetuning: early layers (layers 1–2) primarily learn to route information from `<Q>` positions to `<A>` positions, while later layers (layers 3–4) learn to map the aggregated context to correct answer tokens.

The `<A>` token is particularly interesting: after finetuning, it acts as a **conditional branch point**. The hidden state at the position following `<A>` must encode enough context from the preceding question to predict the answer token. This places a strong demand on the attention mechanism at the layer just before the output projection — it must selectively attend to the relevant question tokens and ignore irrelevant ones.

## Finetune Step Count: 300 vs. Pretrain's 500

The ratio of finetune steps to pretrain steps (300/500 = 0.6) reflects empirical considerations:

- **Convergence speed**: The finetune loss typically plateaus within 150–200 steps. The final 100 steps provide marginal improvement but ensure stable convergence. The loss curve typically shows: rapid drop (steps 1–50), slow improvement (50–150), plateau (150–300).

- **Overfitting boundary**: Beyond ~300 steps, the model begins to overfit to the finetune dataset. Since the dataset is small and repetitive (the same Q&A pairs shuffled repeatedly), continued training leads to near-zero loss on training data but degraded generation quality on any variation — the model memorizes specific character sequences rather than learning the general Q&A pattern.

- **RL warm-start requirement**: The 300-step model retains enough diversity in its output distribution for RL exploration. If trained to convergence (loss → 0), the model's token probabilities would become near-deterministic, and the multinomial sampling in REINFORCE would always select the same token — eliminating exploration.

The 300-step count is also a deliberate under-training relative to the pretrain steps. Pretraining uses 500 steps with a higher learning rate to cover more parameter space; finetuning uses 60% of those steps but at 20% of the learning rate, meaning the total "learning budget" (steps × lr) is 300 × 1e-4 = 0.03 for finetuning vs. 500 × 5e-4 = 0.25 for pretraining — an 8× difference. This reflects the intuition that finetuning should make smaller total adjustments to the model.

## Batch Size and Gradient Variance

The batch size of 32 (same as pretrain) determines the gradient variance during finetuning. For a dataset of ~50K characters, a batch of 32 sequences × 64 tokens = 2,048 tokens represents ~4% of the total dataset. The gradient is a noisy estimate of the true gradient over the full finetuning distribution.

A larger batch size would reduce gradient variance, leading to smoother training but requiring more memory. A smaller batch size would increase stochasticity, potentially helping the model escape local minima but at the cost of noisier convergence. The choice of 32 balances these considerations and is consistent with the pretrain configuration.

Critically, the batch sampling is **without replacement across data positions** (uniform random indices from the character sequence), not **without replacement across Q&A pairs**. This means a single batch might sample multiple windows from the same Q&A pair, or mix question and answer boundaries. The model must learn to handle all these contexts, which provides a form of data augmentation.

## Finetune.pt as the RL Starting Point

`finetune.pt` serves as the shared initialization for three distinct REINFORCE algorithms:

| RL Script | Baseline | Reward Source | KL Control |
|-----------|----------|---------------|------------|
| `reinforce_ema.py` | EMA of past rewards | Rule-based (exact match) | None |
| `reinforce_rm.py` | EMA | Neural reward model | None |
| `reinforce_grpo.py` | Group mean/std | Rule-based | KL penalty against frozen π_ref |

Starting all three from the same `finetune.pt` ensures that observed differences in RL outcomes are attributable to the algorithm, not the initialization. The SFT checkpoint provides:

1. **A policy with nonzero entropy**: The model assigns plausible probability to both correct and incorrect tokens, giving RL room to reshape the distribution. If SFT produced a near-deterministic policy, RL would have no exploration signal.

2. **A coherent language prior**: Even if RL pushes token probabilities toward maximizing reward, the underlying language representations (learned during pretrain and refined during SFT) prevent the model from collapsing into gibberish.

3. **A reference point for KL regularization**: In `reinforce_grpo.py`, the KL penalty `β · KL(π_RL || π_ref)` uses the frozen SFT model as π_ref. This anchors the RL policy to the SFT distribution, preventing reward hacking where the model maximizes reward at the cost of linguistic coherence.

4. **A consistent baseline for comparison**: The reward model in `reinforce_rm.py` is also pretrained from a `pretrain.pt` initialization. Using `finetune.pt` as the RL policy initialization means the policy and the reward model begin from related (but not identical) parameter states, providing a stable training dynamic.

The design philosophy is clear: SFT gets the model to the right **region** of policy space (it can answer questions), and RL then **sharpens** the distribution within that region to favor correct answers and penalize incorrect ones. Without SFT, RL would face a cold-start problem — the model's outputs would be nearly random, and the sparse reward signal (correct answers are rare in 112-character token space) would yield vanishingly small gradient signal.

## The Test Mechanism as Behavioral Verification

After training, `finetune.py` performs an automatic test: it reads the first line of `finetune.txt`, splits on `<A>`, feeds the prompt portion to the model, and generates 100 tokens. This test serves as a quick behavioral verification — not a rigorous evaluation.

The test reveals an important property of the finetuned model: because the finetune data consists of concatenated Q&A pairs (e.g., `<Q>A？<A>B\n<Q>C？<A>D\n...`), the model learns to generate not just a single answer but the entire Q&A sequence pattern. The output might look like:

```
<Q>1+1=？<A>2
<Q>2+2=？<A>4
<Q>3+3=？<A>6
```

This "continuing the pattern" behavior is a direct consequence of the training data structure. The model is not answering a single question; it is reproducing the sequential Q&A format it saw during training. This is format generalization — the model has learned that after one Q&A pair comes another — but it does not imply that the model can handle arbitrary questions outside the training distribution.

## SFT in the Broader LLM Pipeline

SFT occupies a specific point in the modern LLM training pipeline. The three-stage progression — pretrain → SFT → RL — corresponds to three levels of objective alignment:

| Stage | Objective | Data | Behavior |
|-------|-----------|------|----------|
| Pretrain | Next-token prediction | Large, diverse corpus | Linguistic competence |
| SFT | Conditional next-token (task format) | Task-specific demonstrations | Format compliance |
| RL | Reward maximization | Self-generated + feedback | Outcome optimization |

In the InstructGPT/ChatGPT pipeline, SFT is the stage that transforms a raw language model into an instruction-following agent. The `finetune.pt` checkpoint in mini-llm plays the same role: it separates the "can generate text" model from the "can answer questions" model.

The difference between mini-llm's SFT and real-world SFT is one of scale and diversity, not principle. Real-world SFT uses 10K–100K instruction-response pairs spanning diverse tasks; mini-llm uses a few hundred Q&A pairs on a single topic. But the learning mechanism is identical — the model shifts its conditional distribution to match the demonstrated input-output mapping.

## Summary

SFT in mini-llm is not merely "more training on different data." It is a targeted distribution shift that repurposes general language knowledge into task-specific behavior, while carefully balancing adaptation against forgetting. The low learning rate (1e-4), limited step count (300), full-parameter update, gradient clipping, and structured format tokens each play a specific role in this transformation. Together they convert a raw language model into a Q&A system — setting the stage for the REINFORCE algorithms to refine correctness at the token level.

The checkpoint `finetune.pt` encodes both the general language knowledge from pretraining and the task-specific format knowledge from SFT. It is the bridge between "a model that generates text" and "a model that answers questions correctly," and it provides the warm-start initialization that makes RL training feasible and effective.
