# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Presenter**: Ryan Li  
**Email**: [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)  
**Institution**: Vanderbilt University - Data Science Institute  
**Paper**: [arXiv:2405.21060](https://arxiv.org/pdf/2405.21060)  
**Authors**: Tri Dao (Princeton University) & Albert Gu (Carnegie Mellon University)

---

## 📋 15-Minute Presentation Structure

1. **Introduction** (2 min) — Current state of sequence modeling  
2. **Background Deep Dive** (6 min) — Attention vs. SSMs: theoretical forms and efficiency  
3. **The Core Discovery** (3 min) — Structured State Space Duality (Theorem 3.4)  
4. **The SSD Algorithm** (3 min) — Block decomposition (Algorithm 1) and efficiency  
5. **Architectural Comparisons** (2 min) — Mamba-1 vs. Mamba-2  
6. **Experimental Results** (2–3 min) — Benchmarks, trade-offs, and conclusions  

---

## 1. Introduction: Sequence Modeling in 2024

**Transformers** dominate modern sequence models (GPT, Claude, Gemini).  
- **Strength:** State-of-the-art accuracy and generalization.  
- **Weakness:** Quadratic \(O(T^2)\) cost limits scalability to long sequences.

**State Space Models (SSMs)** (S4, Mamba) propose a linear \(O(T)\) alternative.  
- **Strength:** Efficient, structured recurrence.  
- **Weakness:** Historically underutilized GPUs and limited scalability.

**Key Question:**  
> Are Transformers and SSMs fundamentally different, or two algorithmic views of the same structure?

**Answer:**  
Dao & Gu (2024) prove they are **dual** — both compute the same operator using different factorizations.

---

## 2. Background: Mathematical Forms and Efficiency

### 2.1 Attention Mechanisms

\[
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
\]

Quadratic \(O(T^2)\) complexity comes from constructing \(QK^\top\), the dense similarity matrix between all tokens.  
Each token compares itself with all others — yielding expressive but expensive pairwise interactions.

---

### 2.2 Structured State Space Models (SSMs)

\[
h_t = A_t h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t
\]

Sequential \(O(T)\) computation but underutilizes parallel hardware.  
When unrolled, it becomes a **structured semiseparable matrix multiplication**:
\[
y_t = \sum_{s=0}^{t} C_t^\top \left( \prod_{k=s+1}^{t} A_k \right) B_s x_s
\]

The matrix formed by this recurrence is **lower-triangular and semiseparable**, meaning its off-diagonal blocks have low rank.

---

### 2.3 Semiseparable Matrices and 1-Semiseparable Example

The **semiseparable structure** expresses the recurrence efficiently:

\[
M_{ji} =
\begin{cases}
C_j^\top \left(\prod_{k=i+1}^{j} A_k\right) B_i, & j \ge i \\
0, & j < i
\end{cases}
\]

A **1-semiseparable matrix** (rank 1) example:
\[
M =
\begin{bmatrix}
d_1 & 0 & 0 & 0 \\
u_2 v_1 & d_2 & 0 & 0 \\
u_3 v_1 & u_3 v_2 & d_3 & 0 \\
u_4 v_1 & u_4 v_2 & u_4 v_3 & d_4
\end{bmatrix}
\]
Here each off-diagonal element is an outer product \(u_i v_j\).  
This factorization reduces parameters from \(O(T^2)\) to \(O(T)\), providing efficiency while preserving structure.

---

### 2.4 Why \(O(T)\) Parameters Instead of \(O(T^2)\)

- **Dense Attention:** Each pair of tokens interacts → \(T^2\) parameters.  
- **Structured SSMs:** Only per-timestep transitions (\(A_t, B_t, C_t\)) → \(T\) parameters.  

**Trade-offs:**
| Property | Full Attention | Structured SSM |
|-----------|----------------|----------------|
| Expressivity | Global, nonlocal | Structured, decaying memory |
| Complexity | \(O(T^2)\) | \(O(T)\) |
| Parallelism | Excellent | Sequential |
| Inductive Bias | Learned | Dynamical |
| Stability | Softmax normalization | State evolution control |

---

## 3. Duality: Linear vs. Quadratic Forms

The equivalence between:

- **Recurrent Linear Form (Sequential)**  
  \[
  h_t = A_t h_{t-1} + B_t x_t
  \]
- **Quadratic Dual Form (Explicit Kernel)**  
  \[
  y_t = \sum_{s=1}^T C_t^\top \left(\prod_{k=s+1}^t A_k\right) B_s x_s
  \]

The first is **linear in sequence length** \(O(T)\), while the second is **quadratic** \(O(T^2)\).  
Dao & Gu show that both implement the same functional transformation — one sequential, one parallel.

---

## 4. The SSD Algorithm: Block Decomposition

### 4.1 Overview

**Goal:** Combine local attention (parallel) with global SSM recurrence (linear) for both speed and accuracy.

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

Each **block** computes local attention; the SSM connects these blocks linearly.

---

### 4.2 Algorithm Steps and Factorization

**Right Factors (Compression):**
\[
S_j = B_{[jQ:(j+1)Q]}^\top X_{[jQ:(j+1)Q]}
\]
Each block summarizes input features into state embeddings.

**Center Factors (Propagation):**
\[
S_j = \alpha_j S_{j-1} + S_j, \quad \text{where } \alpha_j = \prod_{k=jQ}^{(j+1)Q} a_k
\]
State is propagated across blocks with decay controlled by \(A_t\).

**Left Factors (Projection):**
\[
Y_{[jQ:(j+1)Q]} += C_{[jQ:(j+1)Q]} S_{j-1}
\]
Reprojects the global memory state into output space.

Together:
\[
Y = (C L B^\top) X, \quad L_{ji} = \prod_{k=i+1}^{j} A_k
\]

---

### 4.3 Complexity Summary

| Metric | Attention | SSM | **SSD** |
|:--|:--|:--|:--|
| State Size | \(T\) | \(N\) | \(N\) |
| Training FLOPs | \(T^2 N\) | \(T N^2\) | \(T N^2\) |
| Inference FLOPs | \(T N\) | \(N^2\) | \(N^2\) |
| (Naive) Memory | \(T^2\) | \(T N^2\) | \(T N\) |
| Matrix Multiplication | ✓ | ✗ | ✓ |

SSD preserves matrix-multiplication parallelism while maintaining linear scaling.

---

## 5. Mamba-1 vs. Mamba-2: Architectural Differences

### 5.1 Structural Comparison

| Feature | **Mamba-1 (Sequential)** | **Mamba-2 (Parallel SSD)** |
|:--|:--|:--|
| Computation Order | Sequential recurrence | Block-parallel SSD |
| State Update | Per-timestep SSM | Batched within-block SSM |
| Input/Output Projections | \(A, B, C\) updated sequentially | \(A, B, C\) computed in parallel |
| Normalization | None | Added layer normalization |
| GPU Utilization | ~18% | **70–80%** |
| Core Mechanism | Scan kernel (1D recurrence) | SSD (structured matrix ops) |

---

### 5.2 Architectural Diagram Explanation

**Sequential Mamba Block (Mamba-1):**
- Executes \(A, B, C\) sequentially per timestep.  
- State update depends on the previous output (\(h_{t-1}\)).  
- Low GPU utilization due to recurrence.

**Parallel Mamba Block (Mamba-2):**
- Computes \(A, B, C\) in parallel.  
- Uses structured semiseparable decomposition (SSD).  
- Employs normalization for stability.  
- Significantly improved hardware parallelism.

```text
Sequential Mamba Block:             Parallel Mamba Block:
┌──────────────┐                    ┌──────────────┐
│   Linear     │                    │   Linear     │
│  Projection  │                    │  Projection  │
├──────┬───────┤                    ├──────┬───────┤
│  SSM │ Conv  │                    │  SSM │ Conv  │
│(A,B,C)      │                    │(A,B,C) in parallel│
└──────┴───────┘                    └──────┴───────┘
   ↓  nonlinear                         ↓  nonlinear
  output                               output
```

---

### 5.3 Sequence Parallelism

**Sequence Parallelism:**  
Instead of processing each timestep sequentially, the sequence is divided into blocks that can be computed **in parallel** on GPU tensor cores.  
This hybrid design — **local attention (quadratic within Q)** and **global SSM recurrence (linear across Q)** — combines both scalability and expressivity.

---

## 6. Experimental Results and Analysis

### 6.1 Speed Benchmarks

| Seq Len | FlashAttn-2 | Mamba-1 | Mamba-2 (SSD) | Speedup |
|:--|:--|:--|:--|:--|
| 1 K | 0.5 ms | 0.3 ms | 0.25 ms | 2× |
| 8 K | 15 ms | 2.5 ms | 1.5 ms | 6× |
| 32 K | 120 ms | 10 ms | 6 ms | >10× |

GPU utilization improved from 18% → 75–80%.  
SSD outperforms FlashAttention beyond ≈2K tokens.

---

### 6.2 FLOP and Memory Comparison

| Metric | Attention | SSM | **SSD** |
|:--|:--|:--|:--|
| Training FLOPs | \(T^2 N\) | \(T N^2\) | \(T N^2\) |
| Inference FLOPs | \(T N\) | \(N^2\) | \(N^2\) |
| Memory | \(T^2\) | \(T N^2\) | \(T N\) |
| Matrix Multiplication | ✓ | ✗ | ✓ |

→ SSD uses the same order of FLOPs as SSM but with **GPU-optimized operations**, yielding both linear efficiency and high utilization.

---

### 6.3 Scaling with State Dimension (N)

| N | Mamba-1 Latency | Mamba-2 Latency | GPU Utilization | Efficiency |
|:--|:--|:--|:--|:--|
| 16 | 0.3 ms | 0.5 ms | 45% | Baseline |
| 64 | 1.2 ms | 0.8 ms | 72% | 1.5× |
| 128 | 2.5 ms | 1.0 ms | 78% | 2.5× |
| 256 | 5.0 ms | 1.5 ms | 80% | 3.3× |

---

### 6.4 Language Modeling Quality (The Pile Benchmark)

| Model | Params | PPL ↓ | LAMBADA | HellaSwag | PIQA | ARC-E | WinoGrande | Avg |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Pythia-2.8B | 2.8B | 6.73 | 64.7 | 59.3 | 74.0 | 64.1 | 59.7 | 55.7 |
| Mamba-1 | 2.7B | 6.40 | 67.5 | 63.0 | 75.0 | 68.0 | 61.8 | 58.4 |
| **Mamba-2 (SSD)** | **2.7B** | **6.09** | **69.7** | **66.6** | **76.4** | **69.6** | **64.0** | **60.2** |

→ **2.5× smaller** model size for equivalent or better quality.

---

### 6.5 Associative Recall (MQAR)

| Model | N | Accuracy | Notes |
|:--|:--|:--|:--|
| Attention | — | 95% | Perfect memory |
| Mamba-1 | 16 | 20% | Weak memory |
| Mamba-2 | 64 | 90% | Strong recall |
| Mamba-2 | 256 | **98%** | Matches attention |

---

### 6.6 Theoretical vs Empirical Summary

- **Linear-time complexity** \(O(T)\)
- **Matrix-multiplication compatibility** (GPU-friendly)
- **8× larger state** \(N\) for same cost
- **Equivalent or higher accuracy** across benchmarks
- **10× faster inference** on long sequences

---

## 7. Conclusion

**Key Takeaways:**
1. Transformers and SSMs are mathematically equivalent through **Structured State Space Duality (SSD)**.  
2. SSD achieves **linear-time complexity** with **Transformer-level expressivity**.  
3. Mamba-2 realizes this duality in practice — 8× larger states, 6–10× faster inference, and 75–80% GPU efficiency.  
4. The future of sequence modeling lies in **structured parallelism** — combining recurrence, attention, and semiseparable design into unified architectures.

---

**Thank you!**  
Questions? [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)
