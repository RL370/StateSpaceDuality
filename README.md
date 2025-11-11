# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Presenter**: Ryan Li  
**Email**: [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)  
**Institution**: Vanderbilt University – Data Science Institute  
**Paper**: [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)  
**Authors**: Tri Dao & Albert Gu  

---

## 0. Talk Roadmap (13–15 minutes)

1. Motivation & Setting  
2. Mathematics of SSMs (2.1)  
3. Mathematics of Attention (2.2)  
4. Why Structured Matrices Matter (2.3)  
5. Recurrent Linear vs Dual Quadratic Forms (2.4)  
6. Matrix Form of SSMs (Eq.3), Definition 3.2 & 1-SS Matrices (3.1–3.2)  
7. Formal Algorithms  
   - Attention (Transformer)  
   - SSM (Recurrent)  
   - Mamba-1  
   - Mamba-2 (SSD)  
   - 1-SS / Left–Center–Right factorization  
8. Deeper Questions (2 collapsibles)  
9. References  

---

## 1. Motivation

Sequence modeling currently features two major paradigms:

- **Transformers (Attention)** – extremely expressive but cost grows as \(O(T^2)\).  
- **State Space Models (SSMs)** – linear-time in \(T\), efficient memory, but less exploited in large-scale sequence modeling.

**Core question:**  
> Are these two paradigms fundamentally different, or are they two algorithmic realizations of the same **linear operator**?  

**Answer (Dao & Gu, 2024):**  
Yes — they are duals. Through **Structured State Space Duality (SSD)** we show that attention mechanisms, SSM recurrences, and semiseparable matrix operators are different algorithms evaluating the same underlying transformation.  

---

## 2. Mathematics of Sequence Models

### 2.1 State Space Models (SSMs)

Define latent state \(h_t \in \mathbb R^N\), input \(x_t \in \mathbb R^d\), output \(y_t \in \mathbb R^p\):

\[
\begin{aligned}
h_t &= A_t\,h_{t-1} \;+\; B_t\,x_t, \\
y_t &= C_t^\top\,h_t.
\end{aligned}
\]

Here:
- \(A_t \in \mathbb R^{N\times N}\): state transition (how memory evolves).
- \(B_t \in \mathbb R^{N\times d}\): inject current input into state.
- \(C_t \in \mathbb R^{N\times p}\): read out the latent state into output.

**Unrolled causal form:**

\[
y_t = \sum_{s=1}^{t}
C_t^\top
\left(
\prod_{k=s+1}^{t} A_k
\right)
B_s\,x_s.
\]

This shows each past input \(x_s\) influences \(y_t\) via the product of transitions from \(s+1\) to \(t\).

**Intuitive commentary:**
- The model maintains memory via the state \(h_t\).
- The chain of \(A\)-matrices serves as a “temporal transformer” of past inputs.
- Efficiency: you update one state per timestep → \(O(T)\) operations (for structured \(A_t\)).

---

### 2.2 Attention Mechanisms

Given sequence matrix \(X \in \mathbb R^{T \times d_{\text{model}}}\):

\[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V.
\]

Define:

\[
S = \frac{Q\,K^\top}{\sqrt{d_k}},\quad
A = \mathrm{softmax}_{\text{rows}}(S).
\]

Then:

\[
Y = A\,V,\quad
y_t = \sum_{s=1}^{T} \alpha_{t,s}\,v_s,\quad
\alpha_{t,s} = \frac{\exp(q_t \cdot k_s / \sqrt{d_k})}{\sum_{r=1}^T \exp(q_t \cdot k_r / \sqrt{d_k})}.
\]

**Intuitive commentary:**
- Each position \(t\) attends to all past (and sometimes future) positions.
- Flexible dependencies without strict temporal ordering.
- But complexity: building \(QK^\top\) and softmaxing yields \(O(T^2 d)\) operations and \(O(T^2)\) memory.

---

### 2.3 Importance of Structured Matrices

Many sequence-to-sequence models can be expressed as:

\[
Y = M\,X,
\]
with \(M \in \mathbb R^{T \times T}\) whose \(M_{t,s}\) describes how input \(x_s\) affects output \(y_t\).

- In **attention**, \(M\) is dense: \(M_{t,s} = \alpha_{t,s}\).
- In **SSMs** (Eq.3 below), 

\[
M_{t,s} = 
\begin{cases}
C_t^\top \left( \prod_{k=s+1}^{t} A_k \right) B_s, & s \le t, \\
0, & s > t.
\end{cases}
\]

Thus \(M\) is **lower triangular** (causal) and often has additional rank-structure.

**Why structure matters:**

- **Parameter efficiency**: Structured matrices can have \(O(T)\) or \(O(Tr)\) parameters instead of \(O(T^2)\).  
- **Compute efficiency**: Multiplying by \(M\) can be done in \(O(T)\) or \(O(T\log T)\) if structure is exploited (via recurrences, fast transforms).  
- **Inductive bias**: Structure imposes assumptions (e.g., decaying memory, locality) beneficial for many sequence tasks.

One key class: **semiseparable matrices**, which generalize Toeplitz/Hankel structures and support efficient algorithms.

---

### 2.4 Recurrent Linear Form vs Dual Quadratic Form

#### Recurrent Linear Form

\[
\begin{aligned}
h_t &= A_t h_{t-1} + B_t x_t,\\
y_t &= C_t^\top h_t.
\end{aligned}
\]

- Algorithm: a single pass (scan) through sequence.
- Complexity: \(O(T)\) steps for structured \(A_t\).
- Memory: small state \(h_t\) stored; no entire history kept.
- Bottleneck: inherently sequential → limited hardware parallelism.

#### Dual Quadratic Form (Kernel / Attention style)

Define  
\[
K(t,s) =
\begin{cases}
C_t^\top \left( \prod_{k=s+1}^{t} A_k \right) B_s, & s \le t,\\
0, & s > t.
\end{cases}
\]
Then
\[
y_t = \sum_{s=1}^T K(t,s)\,x_s,\quad
Y = K\,X.
\]

- Conceptually like attention: a matrix multiply over full sequence length.
- Complexity: naïvely \(O(T^2 d)\) if \(K\) is dense.
- Advantage: fully parallelizable (matrix operations).
- If \(K\) has structure (e.g., semiseparable), one can get both parallelism and efficiency.

**Comparison summary:**

| Form               | Algorithmic Strategy         | Complexity     | Parallelism         |
|--------------------|-----------------------------|---------------|---------------------|
| Recurrent Linear   | Step-by-step update         | \(O(T\,N^2)\)/\(O(T\,N)\) | Low (serial)        |
| Dual Quadratic     | Full kernel matrix multiply | \(O(T^2\,d)\) | High (matrix ops)   |

SSD combines the best: **structured kernel** so we can get near-linear complexity *and* high parallelism.

---

## 3. Matrix Form, Definition & 1-Semiseparable Matrices

### 3.1 Equation (3): Matrix Transformation Form of SSMs

In the paper:

\[
Y = M\,X,\quad
M_{t,s} =
\begin{cases}
C_t^\top \Big( \prod_{k=s+1}^t A_k \Big) B_s, & s \le t,\\
0, & s > t.
\end{cases}
\tag{3}
\]

**Annotation:**
- \(M\) is lower-triangular → causal.
- Entry \(M_{t,s}\) captures influence of input at time \(s\) on output at time \(t\).
- The product of \(A\)-matrices transports the contribution forward in time.
- This shows: SSM is a particular structured linear operator mapping \(X\) → \(Y\).

### 3.2 Definition (3.2): Semiseparable / Structured Matrix Families

From the paper: A matrix is semiseparable if its off‐diagonal blocks admit a low‐rank factorization (row-vector × column-vector). In particular, a **1‐semiseparable** (1SS) matrix meets:

\[
M_{i,j} = \begin{cases}
d_i, & i = j,\\
u_i\,v_j, & i > j,\\
0, & i < j.
\end{cases}
\]

### 3.3 1-Semiseparable (1SS) Matrices – Deep Intuition

**Definition recap:**

\[
M =
\begin{pmatrix}
d_1 & 0 & 0 & \dots \\
u_2\,v_1 & d_2 & 0 & \dots \\
u_3\,v_1 & u_3\,v_2 & d_3 & \dots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}
\]

**Key points:**
- Each entry below diagonal is **outer-product style**: \(u_i \cdot v_j\).
- Storage cost: vectors \(u \in \mathbb R^T\), \(v \in \mathbb R^T\), diag \(d \in \mathbb R^T\). Total \(O(T)\).
- Matrix-vector product: can be done in \(O(T\,d)\) instead of \(O(T^2\,d)\) by using an accumulator:
  ```
  acc = 0
  for i = 1…T:
    y[i] = d[i]*x[i] + u[i]*acc
    acc += v[i]*x[i]
  ```

**Why relevant to SSD/SSM:**
- The SSM kernel matrix \(M\) from Eq.(3) often has this semiseparable form when \(A_k\) are scalar or diagonal, and \(B_s, C_t\) are rank-1.
- Thus the operator can be evaluated efficiently (linear time) while also supporting a blocked, parallel algorithm.

**Summary annotation:**
- Structured matrix = memory of how every past input affects future outputs.
- 1-SS constraint = structured but flexible enough for many temporal tasks.
- SSD uses these to bridge efficient recurrence and efficient parallelism.

---

## 4. Formal Algorithms

### Algorithm A: Attention Layer (Transformer Style)

**Input:**  
- \(X \in \mathbb R^{T \times d_{\text{model}}}\)  
- Projection matrices \(W_Q, W_K, W_V\)

**Output:**  
- \(Y \in \mathbb R^{T \times d_{\text{model}}}\)

```
Procedure AttentionLayer(X, W_Q, W_K, W_V, d_k):
  Q ← X · W_Q
  K ← X · W_K
  V ← X · W_V
  S ← Q · Kᵀ / √d_k
  A ← softmax_rows(S)
  Y ← A · V
  return Y
```

**Notes:**  
- Dense kernel → \(O(T^2)\) memory and compute.  
- Works well for moderate T but scales poorly to very long sequences.

---

### Algorithm B: SSM Layer (Recurrent Linear Form)

**Input:**  
- Sequence \(x_1…x_T\)  
- Parameter sequences \(A_1…A_T, B_1…B_T, C_1…C_T\)  
- Initial state \(h_0 = 0\)

**Output:**  
- Outputs \(y_1…y_T\)

```
Procedure SSM_Layer(x₁…x_T, A₁…A_T, B₁…B_T, C₁…C_T):
  h ← 0
  for t = 1 to T do
    h ← A_t · h + B_t · x_t
    y_t ← C_tᵀ · h
  end for
  return y₁…y_T
```

**Notes:**  
- Linear time \(O(T)\) (with structured \(A_t\)).  
- Sequential loop: can’t fully parallelize across time easily.

---

### Algorithm C: Mamba-1 (Selective SSM)

**Input:**  
- Sequence \(x_{1…T}\)

**Output:**  
- \(y_{1…T}\)

```
Procedure MAMBA-1(x₁…x_T):
  h ← 0
  for t = 1 to T do
    A_t ← f_A(x_t)
    B_t ← f_B(x_t)
    C_t ← f_C(x_t)
    h ← A_t · h + B_t · x_t
    y_t ← C_tᵀ · h
  end for
  return y₁…y_T
```

**Notes:**  
- Adds **input-dependence** of state transforms → selective memory.  
- Still sequential; GPU utilization limited.

---

### Algorithm D: Mamba-2 / SSD Blocked Algorithm

**Input:**  
- \(X \in \mathbb R^{T \times P}\)  
- Scalars \(a_{1…T}\), matrices \(B_{1…T}, C_{1…T}\)  
- Block size \(Q\), number of blocks \(B = T/Q\)

**Output:**  
- \(Y \in \mathbb R^{T \times P}\)

```
Procedure MAMBA-2_SSD(X, a₁…a_T, B₁…B_T, C₁…C_T, Q):
  Partition X into blocks j = 0…B−1 with index set I_j = [jQ:(j+1)Q−1]

  // 1. Intra-block local operator (parallel)
  for j = 0…B−1 in parallel do
    G ← C[I_j] · B[I_j]ᵀ      // shape Q×Q
    L ← BuildMask(a[I_j])      // Q×Q semiseparable mask
    M_j ← L ◦ G
    Y[I_j] ← M_j · X[I_j]
  end for

  // 2. Inter-block (block recurrence)
  S[right][0] ← B[I_0]ᵀ · X[I_0]
  for j = 1…B−1 do
    S[right][j] ← B[I_j]ᵀ · X[I_j]
  end for

  for j = 1…B−1 do
    α_j ← ∏_{k=jQ}^{(j+1)Q−1} a_k
    S[j] ← α_j · S[j−1] + S[right][j]
  end for

  for j = 1…B−1 do
    Y[I_j] ← Y[I_j] + C[I_j] · S[j−1]
  end for

  return Y
```

**Notes:**  
- Most compute is batched matmuls → great GPU efficiency.  
- Recurrence step only across blocks → reduces sequential cost from \(T\) to \(T/Q\).  
- Achieves near-linear scaling and high hardware throughput.

---

### Algorithm E: 1-Semiseparable (1SS) MatVec via Left/Center/Right

**Input:**  
- Sequence \(x \in \mathbb R^{T \times P}\)  
- Parameters \(u_{1…T}, v_{1…T}, d_{1…T}\)

**Output:**  
- \(y \in \mathbb R^{T \times P}\)

```
Procedure SEMISEP_MATVEC(x, u₁…u_T, v₁…v_T, d₁…d_T):
  acc ← 0  (in ℝ^P)
  for i = 1 to T do
    y[i] ← d_i · x[i] + u_i · acc
    acc ← acc + v_i · x[i]
  end for
  return y
```

**Interpretation:**
- **Right factor (v)**: accumulate input influence.
- **Center accumulator (acc)**: running summary.
- **Left factor (u)**: project summary into output.
- Very efficient: \(O(T\,P)\) instead of \(O(T^2\,P)\).

---

## 7. Deeper Questions

<details>
<summary>Q1 — When does structured (1-SS / semiseparable) kernel constraint become limiting?</summary>

**Discussion points:**
- Structured kernels encode decaying, low-rank dependencies.
- May fail when the task demands **dense, non-temporal, cross-dependent** interactions (e.g., arbitrary graphs, algorithmic code).  
- Investigate: How to detect which tasks align with structured kernels vs need full attention?  
- Consider: Adaptive-rank SSDs or hybrid architectures blending structured + dense kernels.
</details>

<details>
<summary>Q2 — If SSD becomes standard, how should model & hardware co-design evolve?</summary>

**Discussion points:**
- SSD turns SSM kernels into large batched matrix multiplications, but hardware is still tuned for classical attention / GEMMs.
- Think: What primitives should future accelerators expose?  
  - E.g., “semiseparable mat-mul” hardware.  
  - “Prefix-product kernels” similar to recurrent scans but vectorized.  
- Architecture implication: Could future backbone be SSM-first (efficient) with attention only as sparse correction?  
- System trade-offs: memory bandwidth, latency vs throughput, recurrence vs full parallelism.
</details>

---

## 8. References

- Vaswani et al., “Attention Is All You Need”, NeurIPS 2017.  
- Gu & Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”, arXiv:2312.00752 (2023).  
- Dao & Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”, ICML 2024.  
- Vandebril et al., “Semiseparable Matrices and Related Algorithms”, 2011.  
- Blog posts: “How transformers, RNNs and SSMs are more alike than you think”.

---

**Thank you!**  
Questions? → [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)
