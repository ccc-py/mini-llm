# ModernLanguageModel — Theoretical Background

## Overview

The `ModernLanguageModel` is a compact decoder-only Transformer designed for a toy character-level language modeling pipeline. It incorporates four modern architectural innovations that have become standard in large language models: **Rotary Position Embedding (RoPE)**, **Root Mean Square Normalization (RMSNorm)**, **SwiGLU** activation in the feed-forward network, and **weight tying** between the input embedding and output projection layers. This document explains the theory behind each component, why they were chosen over classical alternatives, and how they interact in the forward pass.

---

## 1. Rotary Position Embedding (RoPE)

### The Position Encoding Problem

Transformers are permutation-invariant: the self-attention operation `softmax(QK^T / √d)V` computes pairwise affinities between tokens with no inherent notion of order. Without position information, a bag-of-words representation results. Classical Transformer (Vaswani et al., 2017) used **absolute sinusoidal embeddings** added to the input: each position was assigned a fixed vector of sinusoids at varying frequencies.

Absolute sinusoidal embeddings have two limitations: (1) they are added to the token embedding, creating potential interference between semantic and positional signals; (2) they do not naturally encode *relative* position, which is what attention actually needs (knowing that token at position `i` is two steps before token at position `j` is more useful than knowing both absolute indices).

### Rotary Formulation

RoPE (Su et al., 2021) addresses these issues by encoding position directly into the query and key vectors via a *rotation in the complex plane*, rather than adding a positional signal to the input. For a given position `t` and a pair of dimensions `(2k, 2k+1)` within the head-dimension space, RoPE applies:

```
R(t) = [[cos(tω_k), -sin(tω_k)],
        [sin(tω_k),  cos(tω_k)]]
```

where `ω_k = 1/θ^(2k/d)` with `θ = 10000.0` (the same frequency schedule as sinusoidal embeddings). This rotation matrix `R(t)` is applied to the query and key vectors at position `t`:

```
q'_t = R(t) · q_t
k'_t = R(t) · k_t
```

### Relative Position Emerges

The key theoretical insight is that the dot product between a rotated query at position `t` and a rotated key at position `s` depends *only on their relative offset* `t − s`:

```
⟨R(t)·q, R(s)·k⟩ = ⟨q, R(s−t)·k⟩
```

because rotation matrices are orthogonal and `R(t)^T R(s) = R(s−t)`. This means the attention score `q'^T_t k'_s` is a function of `(q, k, t−s)` — the model naturally learns to attend based on relative distances, while each absolute position still gets a unique encoding.

In the implementation (`model.py:16-28`), the frequencies are precomputed as complex numbers on the unit circle (`torch.polar`) and multiplied element-wise with the query and key tensors reinterpreted as complex values. This is mathematically equivalent to the 2×2 rotation but is computationally efficient: a complex multiplication simultaneously rotates both components of each pair of real dimensions.

### Design Choices in This Model

- **`theta=10000.0`**: The same base frequency used in the original RoPE and in Llama. Controls how quickly the rotation frequency decays across dimensions — low-index dimension pairs rotate fast (capture fine-grained position), high-index pairs rotate slowly (capture long-range dependencies).
- **`seq_len=64` and `precompute_freqs_cis(d_model//n_heads, seq_len*2)`**: Frequencies are computed for up to `2×seq_len` tokens, allowing generation beyond the training context window. Each head has `head_dim=32` dimensions, so 16 complex-valued rotation pairs.
- The frequencies are injected into the `Attention` module (line 56) and sliced to the current sequence length `T`, supporting variable-length inputs without recomputation.

### Why RoPE Over Learned Position Embeddings

Learned absolute embeddings require training a separate parameter per position and cannot extrapolate beyond the maximum trained length. RoPE is parameter-free (the rotation matrices are deterministic functions of position), naturally encodes relative position, and has been shown to generalize to longer sequences than seen during training (thanks to the decay of inter-token affinity with distance inherent in the rotation formulation).

---

## 2. RMSNorm (Root Mean Square Normalization)

### The Role of Normalization in Transformers

Training deep Transformers is notoriously unstable due to the interaction between residual streams and attention outputs. LayerNorm (Ba et al., 2016) was the original solution, normalizing each token's hidden state to zero mean and unit variance, then applying an affine transformation:

```
LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β
```

where `μ = mean(x)`, `σ² = var(x)`, and `γ, β` are learnable.

### RMSNorm Simplification

RMSNorm (Zhang & Sennrich, 2019) removes the mean-centering step, normalizing by the root-mean-square statistic only:

```
RMSNorm(x) = γ · x / √(RMS(x)² + ε)
where RMS(x) = √(mean(x²))
```

In `model.py:6-14`, this is implemented as:

```python
norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
return self.weight * (x * torch.rsqrt(norm_x + self.eps))
```

The re-scaling factor `torch.rsqrt(mean(x²) + ε)` ensures the output has unit RMS (approximately), and the learnable `weight` (γ) allows the model to scale each dimension independently.

### Theoretical Justification

Why is mean-centering unnecessary? The residual stream in a Transformer accumulates signals across layers. LayerNorm's centering ensures zero mean regardless of the input distribution, but Zhang & Sennrich showed that the re-scaling operation alone provides sufficient normalization for stable training. The intuition is that the **scale** of activations (their RMS) is the primary source of training instability, while the **mean** tends to be relatively stable or can be absorbed by the following linear layer's bias.

More formally: removing the mean-centering operation is equivalent to assuming the incoming activations have approximately zero mean already. In a Pre-LN Transformer (where normalization is applied *before* each sub-layer, as in `model.py:76-77`), the residual stream receives the output of the previous normalized sub-layer added back, which empirically maintains near-zero mean. Under this condition, the centering step of LayerNorm is redundant, and RMSNorm saves the computation of the mean.

### RMSNorm vs LayerNorm in Practice

| Property | LayerNorm | RMSNorm |
|----------|-----------|---------|
| Normalization statistic | Mean + variance | RMS only |
| Parameters | γ (scale) + β (shift) | γ (scale only) |
| FLOPs per token | ~3N (mean, var, scale) | ~2N (square, scale) |
| Training stability | Equivalent (empirically) | Equivalent |

The parameter savings (no β per normalizer) are minor in absolute terms — with `d_model=128` and 5 RMSNorm instances (two per layer × 4 layers + final), this saves 640 parameters — but the principle scales: in a 7B-parameter model with 64 layers and d_model=4096, RMSNorm saves millions of parameters.

### Position in the Architecture

The model uses **Pre-LN** placement (norm before each sub-layer, `model.py:76-77`):

```python
x = x + self.attention(self.norm1(x), freqs_cis)  # attention with Pre-LN
x = x + self.ffn(self.norm2(x))                    # FFN with Pre-LN
```

And a final normalization before the output projection (`model.py:95`):

```python
logits = self.output(self.norm(x))
```

Pre-LN has been shown to enable training with higher learning rates and more stable gradients compared to Post-LN, and has become the de facto standard in modern LLMs (GPT-2, Llama, Mistral).

---

## 3. SwiGLU Activation

### The Gated Linear Unit Family

The original Transformer used a ReLU-activated two-layer feed-forward network:

```
FFN_ReLU(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
```

This expands the dimension from `d_model` to `d_ff` and projects back, with a non-linear activation in between. Various improvements replace ReLU with other activations (GELU, SiLU), but **SwiGLU** (Shazeer, 2020) takes a different approach: it introduces a *gating mechanism*.

SwiGLU is defined as:

```
FFN_SwiGLU(x) = W₂ · (SiLU(W_gate · x) ⊙ W_up · x)
```

where `⊙` is element-wise multiplication, `W_gate` and `W_up` are two independent linear projections to the hidden dimension, and `SiLU` (also called Swish) is `σ(x) · x` where `σ` is the sigmoid function.

In `model.py:30-38`, the `FeedForward` class implements this:

```python
def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

`w1` is the gate projection, `w3` is the up projection, `w2` is the down projection. The `hidden_dim` is set to `4 * dim` (so 512 for d_model=128).

### Why Gating Improves Expressiveness

The key insight: the element-wise product `SiLU(W_gate · x) ⊙ (W_up · x)` allows the network to *selectively* pass or block information through each hidden dimension. The gate `SiLU(W_gate · x)` acts as a learned, input-dependent mask over the up-projected values `W_up · x`.

Consider a standard ReLU FFN: ReLU(W₁x) is an element-wise threshold, but every dimension is computed from the *full* input. In SwiGLU, the up-projection `W_up · x` and the gate `W_gate · x` are two independent linear combinations of the input — the gate learns *which combinations of input features* should be allowed through. This gives the network more flexibility to route information differently for different inputs, akin to the gating in LSTMs or mixture-of-experts.

### Parameter Count Analysis

A notable property: SwiGLU introduces a third weight matrix (`w3`). For a standard ReLU FFN with `d_model=128` and `d_ff=512`, the parameters are:

- ReLU FFN: `128×512 + 512×128 = 131,072` (two matrices)
- SwiGLU FFN: `128×512 + 128×512 + 512×128 = 196,608` (three matrices)

SwiGLU has 50% more parameters in the FFN. However, Shazeer (2020) showed that SwiGLU achieves better performance even when `d_ff` is reduced by 2/3 (i.e., `hidden_dim ≈ 2.67 × d_model` instead of 4×). Many modern LLMs (Llama, PaLM, GPT-J) use `hidden_dim = (8/3) × d_model` to roughly match the parameter count of a 4× ReLU FFN. This model uses 4× for simplicity, accepting the parameter increase at toy scale.

### The SiLU (Swish) Activation

SiLU is `x · sigmoid(x)`. Unlike ReLU (which has zero gradient for negative inputs), SiLU is smooth, non-monotonic, and has a small negative gradient for negative values. This avoids the "dying ReLU" problem and can yield better optimization, though the difference is relatively small at small scale.

---

## 4. Weight Tying

### The Embedding-Output Symmetry

In `model.py:88`, the input embedding and output projection share weights:

```python
self.tok_emb.weight = self.output.weight
```

This assigns the same `nn.Parameter` object to both matrices, so they always have identical values and receive combined gradients during backpropagation.

### Theoretical Intuition

The input embedding `E` maps a one-hot token index to a dense vector in ℝ^d_model. The output projection `W_out` maps a hidden state in ℝ^d_model to logits over the vocabulary. These two operations are, in a sense, transposes of each other: embedding looks up a row, output projects via a column.

In models with tied weights, the same vector serves dual roles: it is the "representation" of a token when it appears as input, and it is the "unembedding" vector that determines how strongly the model predicts that token as output. This creates a **symmetry constraint**: if token A is similar to token B in embedding space, the model must also predict them similarly.

This is theoretically appealing because it enforces a form of regularization — the model cannot learn arbitrary input representations that diverge from what the output layer expects. From a gradient perspective, weight tying ensures that updates to the embedding layer are influenced by both the input embedding loss and the output prediction loss, creating a richer training signal.

### Parametric Benefit

At small scale, the savings are modest. With `vocab_size=31` and `d_model=128`, the embedding matrix has `31×128=3,968` parameters, which is 31×128 = 3,968. Without tying, the output matrix adds another 3,968, totaling ~8K. Tying saves 3,968 parameters — about half the embedding/output parameters (though still a small fraction of the full model's ~230K parameters).

The benefit grows with vocabulary size. In large language models with 32K-256K token vocabularies, weight tying saves tens of millions of parameters and is standard practice (GPT-2, Transformer-XL, most BERT-style encoders).

### Practical Consideration

Weight tying requires `d_model` to be the same for the embedding and output layers, which is naturally the case here. It also means the final `nn.Linear(d_model, vocab_size, bias=False)` must have no bias, since `nn.Embedding` has no bias parameter to match. The model correctly omits bias in the output projection.

---

## 5. Interplay of Components in the Forward Pass

The full forward pass (`model.py:91-101`) proceeds as:

1. **Token Embedding** (line 92): Each input token index is mapped to a `d_model=128`-dimensional vector via `tok_emb`. No position information is added at this stage — RoPE handles it later.

2. **TransformerBlock × 4** (lines 93-94): Each block applies:

   a. **RMSNorm → Attention**: The token stream is normalized, then attention computes Q, K, V projections. RoPE is applied to Q and K (line 56). The attention scores `QK^T/√d_head` are masked with a causal triangle (line 59) and softmaxed. The weighted sum over V is projected through `wo`.

   b. **Residual add** (line 76): `x + attention_output` — the residual connection preserves the original token information and allows gradient flow.

   c. **RMSNorm → SwiGLU FFN**: The stream is normalized again, passed through the gated FFN.

   d. **Residual add** (line 77): `x + ffn_output`.

3. **Final RMSNorm → Output** (line 95): The final hidden state is normalized and projected to vocabulary logits via the (tied) output weight matrix.

### Signal Propagation

The residual connections (`x + sublayer(norm(x))`) ensure that the token embeddings can pass through all 4 layers largely unmodified. The normalization layers prevent the sub-layer outputs from dominating the residual stream. The RMSNorm before output ensures the final hidden state has consistent scale for the linear projection.

RoPE modifies Q and K but not V — the value vectors carry content information without positional distortion. This is intentional: the attention distribution should depend on position (via rotated Q, K), but the *information* retrieved from each token (V) should be position-independent.

---

## 6. Design Rationale for a Toy Model

### Why These Choices at Small Scale?

A classical "mini" Transformer might use learned position embeddings, LayerNorm, and ReLU FFN. This model deliberately chooses the modern stack to demonstrate that these innovations are not exclusive to large models:

- **RoPE** adds zero learned parameters but provides extrapolation capability. Even at `seq_len=64`, it allows generating beyond 64 tokens by sliding the context window.
- **RMSNorm** is slightly simpler to implement and marginally cheaper to compute. At toy scale the difference is negligible, but the principle generalizes.
- **SwiGLU** is arguably over-engineered for a 128-dimensional model, but it demonstrates gated architectures. The extra parameters in the FFN are absorbed by the toy scale.
- **Weight tying** is beneficial at any scale where the vocabulary is non-trivial.

### Didactic Value

Each component reflects a real design choice in production LLMs (Llama uses RMSNorm + RoPE + SwiGLU + weight tying). The toy model serves as a minimal reproducible example of the Llama architecture, making it a useful educational tool for understanding modern Transformer design.

---

## 7. Scaling Properties

### Dimension Scaling

- **d_model=128**: Each token is represented in 128 dimensions. The head dimension is `128/4=32`. This is small but sufficient for a 31-character vocabulary at sequence length 64. In large models, d_model scales to 4096-16384.
- **n_heads=4**: Each head operates on 32 dimensions. More heads allow the model to attend to different types of relationships (syntactic, semantic, positional) in parallel. In practice, the number of heads increases with d_model (typically d_model/head_dim where head_dim=64-128).

### Depth Scaling

- **n_layers=4**: Each layer refines the representation through attention (mixing information across tokens) and FFN (transforming each token independently). Deeper models (32-80 layers) can learn hierarchical patterns and more complex dependencies, though they require careful normalization (RMSNorm) and residual connections to train.

### Sequence Length

- **seq_len=64**: The causal mask restricts attention to the preceding 63 tokens. RoPE can extrapolate somewhat beyond this, but performance degrades as positions become unfamiliar to the trained attention patterns.

### Parameter Breakdown

With the given hyperparameters:

| Component | Dimensions | Parameter Count |
|-----------|-----------|-----------------|
| Token embedding | 31 × 128 | 3,968 |
| Attention (per head, per layer) | Q: 128×128, K: 128×128, V: 128×128, O: 128×128 | 65,536 |
| Attention total (4 layers) | 4 × 65,536 | 262,144 |
| FFN (per layer) | w1: 128×512, w3: 128×512, w2: 512×128 | 196,608 |
| FFN total (4 layers) | 4 × 196,608 | 786,432 |
| RMSNorm (5 instances) | 5 × 128 | 640 |
| **Total (tied)** | — | ~1,053,184 |

The FFN dominates parameter count at ~75% of total, which is typical for Transformers.

---

## 8. Role of Each Sub-Layer

### Attention (Causal Self-Attention)

**Role**: Mix information across positions. Each token's output is a weighted sum of all preceding tokens' value vectors, where weights are determined by compatibility between query (current token) and keys (preceding tokens).

**Why causal**: Language modeling is autoregressive — token at position `t` can only depend on tokens at positions `< t`. The triangular mask (`model.py:59`) enforces this by setting attention scores between future positions and current position to -∞.

### Feed-Forward Network (SwiGLU)

**Role**: Transform each token's representation independently (no cross-token communication). SwiGLU's gating acts as a learned feature selector, allowing the network to route different input patterns through different hidden dimensions.

**Why two norms + residual**: The attention and FFN sub-layers serve complementary roles — attention handles inter-token dependencies, FFN handles intra-token computation. Each gets its own residual connection (allowing gradients to bypass either sub-layer if uninformative) and its own normalization (keeping activation scales bounded).

### Output Projection (Tied Embedding)

**Role**: Map the final hidden state to vocabulary logits. Because of weight tying, the `i`-th row of the embedding matrix is both the representation of token `i` as input and the weight vector that produces the logit for token `i` at the output.

---

## References

- Vaswani et al., "Attention Is All You Need" (2017) — original Transformer with sinusoidal position encodings, LayerNorm, ReLU FFN
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — RoPE formulation
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) — RMSNorm
- Shazeer, "GLU Variants Improve Transformer" (2020) — SwiGLU activation
- Ba et al., "Layer Normalization" (2016) — original LayerNorm
- Press & Wolf, "Using the Output Embedding to Improve Language Models" (2017) — weight tying rationale
- Touvron et al., "Llama: Open and Efficient Foundation Language Models" (2023) — modern production stack combining all four techniques
