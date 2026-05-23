# Theoretical Background: Causal Language Model Pretraining

## 1. The Self-Supervised Learning Paradigm

Pretraining is a form of **self-supervised learning**: the data provides its own supervision without human annotation. In causal language modeling, the supervision signal is inherent in the sequential structure of text — each token's natural position in a sequence defines the prediction target for all preceding tokens.

### 1.1 The Autoregressive Factorization

Given a sequence of tokens $(x_1, x_2, \dots, x_T)$, the joint probability of the sequence factorizes via the chain rule:

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

This factorization is always valid (by the product rule of probability) but is computationally useful only if we can model the conditional distributions efficiently. The Transformer architecture enables this through causal masking: each position `t` can only attend to positions `≤ t`, mirroring the autoregressive constraint.

### 1.2 Next-Token Prediction as Maximum Likelihood

The training objective minimizes the negative log-likelihood of each token given its context:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

This is exactly a cross-entropy loss between the model's predicted distribution over the vocabulary and the one-hot true token. In `pretrain.py`, this is computed as:

```
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
```

where `logits` has shape `(B, T, vocab_size)` and `targets` is the input shifted left by one position. The loss is averaged over all `B×T` positions in the batch.

### 1.3 Why This Works

Minimizing next-token prediction loss forces the model to internalize the statistical structure of the training distribution. Any regularity in the data — syntactic patterns, factual associations, reasoning chains — manifests as predictable next-token probabilities. The model must capture these regularities to reduce its loss. This is the information-theoretic justification: the loss lower-bounds the cross-entropy between the true data distribution and the model's distribution, and minimizing it drives the model toward the true distribution in KL divergence.

## 2. What Pretraining Actually Learns

### 2.1 Linguistic Knowledge

Even without explicit grammar instruction, next-token prediction induces implicit linguistic knowledge. To predict the next character accurately, the model must learn:

- **Phonotactics and orthography**: which character sequences are legal
- **Morphology**: how characters combine into meaningful units
- **Syntactic structure**: word order, agreement, subordination patterns
- **Semantic relationships**: which concepts co-occur and in what contexts

In our toy setting with only digits, `+`, `=`, and punctuation, the model learns a micro-language: the grammar of arithmetic expressions. The pattern `{digit}+{digit}={digit}` is a rigid syntactic structure, and the model must learn both the form (two operands, operator, equals, result) and the content (the result must equal the sum).

### 2.2 Factual Knowledge

Language models store facts implicitly in their weights. A fact like `3+5=8` appears many times in the pretraining data; the model learns to predict `8` after seeing `3+5=`. This is not memorization in the traditional sense — the model does not store a lookup table. Instead, the facts are encoded in the distributed representations of the Transformer's residual stream and attention patterns, which is why the same mechanism generalizes to unseen combinations (though in this toy setup, all 81 combinations appear in pretraining).

### 2.3 The World Model Hypothesis

A growing body of research suggests that language models, through next-token prediction, build an internal **world model** — a compressed representation of the process that generated the training data. The model must predict not just surface-level statistics but the latent causes behind the text. In our case, the latent cause is the arithmetic operation `+`. The model must discover that the symbol `+` denotes a function that maps pairs of digits to sums. This requires the model to learn a compositional representation: the representation of `3+5` must be systematically related to the representations of `3` and `5` and the operation `+`.

This is why causal LM pretraining transfers to downstream tasks: the internal representations encode features that are useful beyond next-token prediction.

## 3. The Scaling Hypothesis

### 3.1 Empirical Scaling Laws

Kaplan et al. (2020) and Hoffmann et al. (2022) demonstrated that Transformer language models follow power-law scaling with respect to three axes: **model size** (parameter count), **dataset size** (training tokens), and **compute** (FLOPs). The test loss decreases as a power law in each, provided the other resources are not bottlenecks.

The Chinchilla scaling law (Hoffmann et al., 2022) states that for compute-optimal training, the model size and data size should scale in proportion: doubling the model requires roughly doubling the training tokens, not just increasing one.

### 3.2 Implications for Toy Models

Our model has approximately 1.6M parameters (vocab_size=31, d_model=128, 4 layers, 4 heads, SwiGLU FFN with hidden_dim=512). The training data is 100K characters (~100K tokens at the character level). This gives a token-to-parameter ratio of roughly 0.06 — far below the compute-optimal ratio (which is approximately 20:1 tokens-to-parameters for Chinchilla).

Why does such a data-starved setting still produce a useful model? The arithmetic grammar is extremely low-dimensional. The true underlying distribution has only ~81 distinct facts plus a handful of meta-statements. The effective **information content** of the data is minuscule compared to natural language. A model with millions of parameters can easily memorize the training distribution — which is actually the goal here, since we want the pretrained model to serve as a warm start for finetuning.

## 4. Pretraining as Initialization for Downstream Tasks

### 4.1 The Two-Stage Paradigm

The dominant paradigm in modern NLP is **pretrain then finetune**:

1. **Pretraining**: Train on a large, diverse, unlabeled corpus using self-supervised loss.
2. **Finetuning**: Continue training on a smaller, labeled dataset for a specific task.

In `v4-reinforce`, the pipeline is:
- `pretrain.py` → produces `pretrain.pt`
- `finetune.py` → loads `pretrain.pt`, trains on `finetune.txt`
- RL stage → loads finetuned weights for reinforcement learning

### 4.2 Why Pretraining Helps

Pretraining provides a superior initialization for several reasons:

**Feature reuse**: The lower layers of the pretrained model learn general-purpose features (character embeddings, positional patterns, simple arithmetic). Finetuning only needs to adjust the higher layers to the specific task format (`<Q>{a}+{b}=？<A>{r}`).

**Regularization**: Starting from a pretrained initialization constrains the model to remain near the pretrained solution, which acts as a strong regularizer when the finetuning dataset is small.

**Optimization landscape**: Pretrained weights lie in a region of the loss landscape that is both low-loss and well-connected to good finetuning solutions. Random initialization may land in regions with poor curvature or sharp minima.

### 4.3 The Lottery Ticket Perspective

A complementary view: pretraining selects a "winning ticket" subnetworks. The pretraining objective discovers which parameters are important for modeling language structure; random initialization does not provide this information. The 500 steps of pretraining allow the optimizer to identify a useful subspace that finetuning can then refine.

## 5. Data Diversity vs. Repetition

### 5.1 The Diversity Spectrum

Natural language pretraining corpora aim for maximal diversity: different topics, genres, authors, styles. Diversity ensures the model encounters a wide range of syntactic constructions and factual associations, which improves generalization.

### 5.2 Repetition in Toy Domains

Our pretraining data consists of the same 81 arithmetic facts repeated ~1200 times (100K chars / ~80 chars per repetition). This is the opposite of diverse — it is maximally repetitive.

Repetition serves a different purpose here:
- **Overfitting is the goal**: We want the model to memorize the addition table perfectly.
- **Convergence speed**: High repetition of a small set of patterns leads to rapid convergence of the loss.
- **Representation consolidation**: Each repetition reinforces the same underlying patterns, allowing the model to compress the facts into stable internal representations.

The key insight: **diversity and repetition are on a continuum, and the optimal point depends on the true entropy of the data distribution**. For low-entropy distributions (like a closed set of arithmetic facts), high repetition is efficient. For high-entropy distributions (like natural language), diversity is critical.

### 5.3 The Risk of Repetition

Repeated data can lead to **representational collapse** if the model overfits to surface statistics rather than learning the underlying generative process. However, because our data generator creates each line independently (shuffling facts each epoch), the model cannot simply memorize fixed position-content mappings — it must learn the actual arithmetic rule.

This is analogous to the distinction between **memorization** (learning input-output pairs) and **knowledge** (learning the underlying function). The shuffling forces the latter.

## 6. Convergence Theory for Small Transformers

### 6.1 Why 500 Steps Is Enough

Training for 500 steps with batch_size=32 and seq_len=64 processes approximately 500 × 32 × 64 = 1,024,000 tokens. Since the dataset is 100K characters, the model sees roughly 10 epochs of data.

Several factors explain rapid convergence:

**Model size relative to data complexity**: The true function to be learned (addition of digits 1-9) has a Kolmogorov complexity of perhaps 50 bits. A Transformer with ~1.6M parameters has enormous capacity relative to this complexity, so gradient descent can find a solution quickly.

**Loss landscape conditioning**: The AdamW optimizer with learning rate 5e-4 provides adaptive gradient scaling. The combination of RMSNorm (which stabilizes hidden state magnitudes) and gradient clipping (norm=1.0) ensures training remains in a well-conditioned region of the loss landscape.

**Information bottleneck**: With seq_len=64, the model sees up to 64 characters of context. The relevant information for predicting the next character (e.g., the two operands and the operator) is well within this window. There is no long-range dependency problem.

### 6.2 Gradient Clipping

Gradient clipping (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`) caps the global gradient norm at 1.0. This prevents any single batch from causing a destructive update. In the early stages of training, when the model's predictions are essentially random (cross-entropy near log(31) ≈ 3.4), gradients can be large. Clipping ensures stability, especially with a small model and aggressive learning rate.

### 6.3 Learning Rate and Optimizer Choice

AdamW with lr=5e-4 is the standard choice for Transformer pretraining. AdamW decouples weight decay from the adaptive gradient updates, which improves generalization compared to L2 regularization. The learning rate is kept constant (no scheduler) because:
- 500 steps is too short for a cosine or linear decay schedule to matter
- The loss converges well within this window; a scheduler would only affect the final ~50 steps

In larger-scale pretraining, a warmup phase followed by cosine decay is essential. The absence of a scheduler here is a pragmatic simplification justified by the small scale.

## 7. Character-Level vs. Subword-Level Language Modeling

### 7.1 The Tokenization Hierarchy

Language models can operate at different granularities:

| Level | Example | Vocabulary Size |
|-------|---------|----------------|
| Byte | `0xE3 0x80 0x80` | 256 |
| Character | `3 + 5 = 8` | ~31 (our case) |
| Subword (BPE) | `3+5=8` | ~8K-50K |
| Word | `three plus five equals eight` | ~100K+ |

### 7.2 Tradeoffs

**Character-level**:
- *Pro*: No tokenization ambiguity; closed vocabulary; any string can be represented.
- *Pro*: The model learns orthographic and morphological patterns from first principles.
- *Con*: Longer sequences (each word is multiple characters) → more computation for the same semantic content.
- *Con*: The prediction task is easier (31-way classification vs. 50K-way), which may limit the model's ability to learn high-level abstractions — though this is a practical concern, not a theoretical one.

**Subword-level**:
- *Pro*: Better compression: common words are single tokens, rare words are decomposed.
- *Pro*: Aligns well with the statistics of natural language (Zipf distribution).
- *Con*: Requires a separate tokenization step; tokenizer artifacts can affect model behavior.

### 7.3 Why Character-Level Fits This Toy Domain

With only 31 characters and highly regular patterns, character-level modeling is ideal:
- The vocabulary is the complete set of symbols needed
- The patterns are short and character-boundary-aligned (digits are single characters)
- There is no morphological complexity to capture
- It simplifies code (no BPE training, no special tokens like `<unk>`)

This setup essentially reduces to a sequence modeling problem where the model must learn a finite-state machine for arithmetic.

## 8. Loss Landscape and Optimization in Small Transformers

### 8.1 The Loss Landscape

The loss landscape of a Transformer is high-dimensional, non-convex, and充满了with saddle points rather than local minima (for large models). However, for our small model, several properties simplify optimization:

**Weight tying**: `self.tok_emb.weight = self.output.weight` ties the input embedding and output projection matrices. This reduces the parameter count and constrains the optimization to a lower-dimensional manifold. It also ensures that the same representations are used for encoding and decoding, which acts as a regularizer.

**Precomputed RoPE frequencies**: Rotary Position Embeddings are fixed (not learned), which removes positional encoding parameters from the optimization problem. The sinusoidal structure ensures smooth interpolation across positions.

### 8.2 Convergence Behavior

Typical training progress in this setup:
- **Step 0**: Loss ≈ log(31) ≈ 3.4 (uniform distribution over 31 characters)
- **Step 100**: Loss drops to ~1.0-1.5 (model learns character bigram statistics and the basic `digit+digit=` pattern)
- **Step 300**: Loss below ~0.5 (model predicts most arithmetic results correctly)
- **Step 500**: Loss ~0.1-0.2 (near-perfect memorization)

The rapid drop followed by a long tail is characteristic of learning in overparameterized models: the model quickly captures the high-frequency patterns (digit distributions, the `+` and `=` tokens) and then slowly refines the low-frequency details (the specific digit-digit-result mappings).

### 8.3 The Role of Architecture

The specific architectural choices in `ModernLanguageModel` affect the loss landscape:

- **RMSNorm** (instead of LayerNorm): Removes the mean-centering step, which simplifies the gradient flow and slightly reduces compute. In small models, the difference is negligible, but RMSNorm has become standard in modern LLMs (e.g., Llama, Mistral).

- **SwiGLU FFN** (instead of ReLU or GELU): The SwiGLU activation (`F.silu(w1(x)) * w3(x)`) introduces a gating mechanism that has been shown to improve training efficiency. The gating provides a multiplicative interaction that can learn more expressive functions per parameter.

- **RoPE** (instead of learned absolute positions): RoPE encodes relative position information directly in the attention computation. This means the model can generalize to the full seq_len=64 window even if particular patterns appear at varying positions in the training data.

## 9. Pretraining and the Model's Internal World Model

### 9.1 Emergent Representations

Through pretraining, the model develops internal representations that reflect the structure of the domain. The residual stream at each layer can be interpreted as a **compressed state** of the computation so far. Attention patterns reveal which previous tokens are relevant for predicting the next token — for example, the model learns to attend to the first operand and the operator `+` when predicting the result.

These emergent representations constitute a world model because they support **counterfactual reasoning**: given partial input `3+5=`, the model's hidden states encode the expectation that the next token should be `8`. This expectation is not just a learned association; it is the result of a compositional computation over the representations of `3`, `+`, and `5`.

### 9.2 The Predictive Coding View

Pretraining through next-token prediction can be seen as a form of **predictive coding**: the model continuously predicts upcoming input, and errors (surprisal) drive learning. This framework, originating in neuroscience, posits that intelligence emerges from the drive to minimize prediction error across sensory modalities. Language model pretraining operationalizes this at the symbolic level.

### 9.3 Limitations of the World Model

Our toy model's "world model" is limited to arithmetic. It cannot generalize to operations it has not seen (e.g., subtraction) because the training data contains no examples. However, the internal representations are structured enough that a small amount of finetuning on the same operation in a different format (`<Q>3+5=？<A>8`) succeeds. This demonstrates that pretraining has extracted the relevant features (digit values, the addition function) that are reusable across formats.

## 10. Summary

Causal language model pretraining, even on a toy scale with 500 steps on synthetic arithmetic data, embodies the same theoretical principles that power modern large language models:

- **Self-supervised learning** via autoregressive next-token prediction
- **Implicit knowledge acquisition** from statistical regularities
- **Distributed representations** that encode a world model
- **Transferable features** that benefit downstream tasks

The specific design choices in `pretrain.py` — 500 steps with AdamW, gradient clipping, RMSNorm, RoPE, SwiGLU, weight tying, and character-level tokenization — are each motivated by the theory of optimization and generalization in neural language models, scaled down to a regime where the model can be trained on a laptop in seconds while still demonstrating the core principles of the pretrain-finetune paradigm.
