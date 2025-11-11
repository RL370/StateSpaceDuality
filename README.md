# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Presenter**: Ryan Li  
**Email**: [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)  
**Institution**: Vanderbilt University - Data Science Institute  
**Paper**: [arXiv:2405.21060](https://arxiv.org/pdf/2405.21060)  
**Authors**: Tri Dao (Princeton University) & Albert Gu (Carnegie Mellon University)

---

## 📋 15-Minute Presentation Structure

1. **Introduction** (2 min) — Current state of sequence modeling  
2. **Background Deep Dive** (5 min) — Evolution of sequence models, SSMs, and Attention  
3. **The Core Discovery** (3 min) — Mathematical equivalence (Theorem 3.4)  
4. **The SSD Algorithm** (3 min) — Block decomposition with pseudocode  
5. **Experimental Validation** (1.5–2 min) — Mamba-1 vs Mamba-2 comparison  
6. **Significance & Discussion** (1 min) — Why this unification matters

---

## 1. Introduction: The State of Sequence Modeling (2024)

In 2024, sequence modeling is dominated by two paradigms:

**Transformers**  
- Power — state-of-the-art on most benchmarks  
- Cost — Quadratic \(O(T^2)\) complexity limits context length  
- Reality — GPT-4’s 128 K context remains extremely expensive  

**State Space Models (SSMs)**  
- Efficiency — Linear \(O(T)\) complexity enables long contexts  
- Quality — Approaching Transformer performance at small-to-medium scale  
- Adoption — Growing interest but still lagging behind Transformers  

> **Central Question → Are these architectures fundamentally different, or two sides of the same coin?**

---

## 2. Background Deep Dive

### 2.0 Evolution of Sequence Models

1️⃣ **RNNs & LSTMs** — Causal recurrence, but sequential + gradient issues.  
2️⃣ **Temporal Convolutions** — Parallel but fixed receptive fields.  
3️⃣ **Transformers** — Global context, quadratic cost.

---

### 2.1 Attention Mechanisms

\[
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
\]

| Role | Meaning |
|------|----------|
| **Queries (Q)** | “What am I looking for?” |
| **Keys (K)** | “Where is that information?” |
| **Values (V)** | “What information do I return?” |

Attention = adaptive memory lookup → powerful but \(O(T^2)\).

---

### 2.2 State Space Models (SSMs)

\[
h_t=A_t h_{t-1}+B_t x_t,\qquad y_t=C_t^{\top}h_t
\]

| Parameter | Meaning |
|:--|:--|
| \(A_t\) | State transition – memory decay/persistence |
| \(B_t\) | Input projection |
| \(C_t\) | Output projection |

Modern lineage: S4 (2021) → S5/S6 → Mamba (2023) → **Mamba-2 (2024)**.

---

### 2.3 The Trade-off

| Aspect | Transformers | SSMs |
|:--|:--|:--|
| Complexity | \(O(T^2)\) | \(O(T·N)\) |
| Hardware Utilization | ≈ 90 % | ≈ 18 % |
| Parallelization | Full | Sequential |
| Context Length | ≤ 128 K | Unbounded |

Goal → linear time **and** GPU efficiency.

---

## 3. The Core Discovery (Theorem 3.4)

### Three Equivalent Forms

1️⃣ **SSM Recurrence** (sequential)  
2️⃣ **Semiseparable Matrix** (dense parallel)  
3️⃣ **Structured Masked Attention** (sparse parallel)

Thus → **SSMs ≡ Semiseparable Matrices ≡ Structured Attention**.

| Component | Standard Attention | Structured Dual (SSD) |
|:--|:--|:--|
| Mask | Uniform 1’s | Learned decay ∏ aₖ |
| Activation | Softmax | Linear |
| Complexity | O(T²) | O(T) |

---

## 4. The SSD Algorithm: Block Decomposition

**Goal:** Combine local attention (parallel) + global SSM (linear).

```text
Sequence of T tokens
┌─────────┬─────────┬─────────┬─────────┐
│ Block 1 │ Block 2 │ Block 3 │ Block 4 │
└─────────┴─────────┴─────────┴─────────┘
     ↓         ↓         ↓         ↓
 [Attention][Attention][Attention][Attention]  (within blocks)
     ↓         ↓         ↓         ↓
     └─────────┴──────────SSM──────────┘      (between blocks)
```

Within-block: small \(Q×Q\) attention → parallel.  
Between-blocks: SSM recurrence → linear.

---

### Pseudocode Sketch

```python
for j in range(num_blocks):
    G = C[block].T @ B[block]      # local Gram matrix
    L = BuildMask(A[block])        # 1-semiseparable mask
    M = L * G
    Y[block] += M @ X[block]

for j in range(1, num_blocks):
    a_block = prod(A[block])
    S[j] = a_block * S[j-1] + B[block].T @ X[block]
    Y[block] += C[block] @ S[j-1]
```

---

## 5. Experimental Highlights

| Seq Len | FlashAttn-2 | Mamba-1 | Mamba-2 | Speedup |
|:--|:--|:--|:--|:--|
| 1 K | 0.5 ms | 0.3 ms | 0.25 ms | 2× |
| 8 K | 15 ms | 2.5 ms | 1.5 ms | 6× |
| 32 K | 120 ms | 10 ms | 6 ms | >10× |

GPU utilization → 18 % → ~75 %.

Mamba-2 matches or exceeds Transformer accuracy on language benchmarks.

---

## 6. Why This Matters

**Unified View of Sequence Models**

```text
      Structured Matrices
            /     \
     Attention     SSMs
     (Quadratic)  (Linear)
            \     /
         Same Operator
```

- **Theory:** Attention and SSMs form one mathematical family.  
- **Practice:** Mamba-2 shows that hybrid algorithms can reach attention-level accuracy with linear scaling.  

---

## 7. Limitations & Future Work

- Proven for structured (semiseparable/scalar) SSMs, not general softmax.  
- SSD kernels require specialized implementation support.  
- Still emerging in vision and multimodal tasks.

---

## 8. Discussion Questions

<details>
<summary>❓ Q1 – Why is O(T·N) more favorable than O(T²)?</summary>

**Answer:**  
Because \(O(T²)\) scales quadratically with sequence length, doubling T → 4× cost.  
Linear \(O(T·N)\) means compute grows only proportionally to T, so long contexts (> 16 K tokens) remain practical.  
For large language models, this translates to orders-of-magnitude less memory and GPU time.
</details>

<details>
<summary>❓ Q2 – If SSDs are better, why do we still use Transformers and SSMs?</summary>

**Answer:**  
SSDs are new and hardware support is limited. Transformers have a massive ecosystem (optimized kernels, pretrained weights, tooling).  
SSMs still excel in some continuous and signal domains.  
SSD research is actively closing this gap but standardization and robust implementations take time.
</details>

---

**Thank you!**  
Questions? [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)
