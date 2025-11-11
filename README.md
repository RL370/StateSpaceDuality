# Transformers are SSMs
## Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Presenter:** Ryan Li  
**Institution:** Vanderbilt University – Data Science Institute  

**Paper:** Dao & Gu (2024), ICML — [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)

---

# Overview (13–15 minutes)

## 0. Understanding Duality - The Core Insight
## 1. The Problem - Two Competing Paradigms
## 2. State Space Models - How They Work
## 3. Attention Mechanisms - How They Work  
## 4. The Bridge - Structured Matrices Unify Them
## 5. Algorithms A-E - Theory to Practice  
## 6. Mamba-2 Results - 2-8X Speedup
## 7. Key Insights - Why This Matters
## 8. Discussion Questions

---

# Understanding Duality: The Core Insight

## What Does "Duality" Mean?

In mathematics, **duality** means that two seemingly different objects or operations are actually two different ways of viewing the **same underlying thing**. They are not competing alternatives—they are two faces of one truth.

### Classic Example: Sum of Integers

Consider computing the sum: $1 + 2 + 3 + 4 + 5$

**Sequential view (recurrence):**
```
result = 0
result = result + 1 → 1
result = result + 2 → 3
result = result + 3 → 6
result = result + 4 → 10
result = result + 5 → 15
```

Must process left-to-right, one step at a time.

**Parallel view (closed form):**
```
result = (5 × 6) / 2 = 15
```

Can compute instantly by recognizing the pattern.

**Same answer, two algorithms.** This is duality.

---

## What is Structured State Space Duality (SSD)?

The key discovery of Dao & Gu (2024):

**SSMs and Attention are dual representations of the same linear operator.**

### SSM's View (Recurrent/Sequential)

```
Process one token at a time
h_t = A_t h_{t-1} + B_t x_t
y_t = C_t^T h_t

Sequential chain:
x_1 → h_1 → y_1
       ↓
x_2 → h_2 → y_2
       ↓
x_3 → h_3 → y_3
```

This is like the **sequential sum** above—process step by step, maintaining state.

### Attention's View (Kernel/Parallel)

```
Process all tokens at once via matrix M
Y = M X

Matrix form:
M represents pairwise influences between all positions
Can parallelize computation
```

This is like the **closed-form formula** above—see the full picture at once.



## Why Is This Important?

Traditional view:
- SSMs = efficient but slow on GPUs
- Attention = inefficient but fast on GPUs
- Choose one or the other

SSD perspective:
- Both are computing the same $M X$ operation
- Choose the algorithm based on the structure of $M$
- If $M$ is semiseparable: use blocked algorithm (best of both worlds)

**Result:** Mamba-2 gets SSM's efficiency with Attention's parallelism.

### Duality Comparison Table

| **Aspect** | **SSM Algorithm** | **Attention Algorithm** | **SSD Solution** |
|:---|:---|:---|:---|
| **Perspective** | Sequential recurrence | Parallel kernel | Structured blocks |
| **Processing Style** | One token at a time | All tokens at once | Blocks in parallel |
| **Memory Usage** | $\mathcal{O}(N)$ state | $\mathcal{O}(T^2)$ matrix | $\mathcal{O}(Q^2)$ per block |
| **Time Complexity** | $\mathcal{O}(T \cdot N)$ | $\mathcal{O}(T^2 \cdot d)$ | $\mathcal{O}(T \cdot d)$ |
| **GPU Utilization** | 18% | 100% (dense) | 70-80% (structured) |
| **Efficiency Type** | ✅ Linear time | ❌ Quadratic | ✅ Linear + Parallel |
| **Best For** | Streaming, low latency | Dense reasoning | Long sequences |

**The Key Takeaway:**

All three are computing the **same underlying operation:** $Y = M X$ where $M$ is a transformation matrix.

- **SSM picks:** Fast sequential algorithm for $M$ (naturally recurrent)
- **Attention picks:** Fully parallel algorithm for $M$ (treats as dense)
- **SSD picks:** Clever blocked algorithm for $M$ when it's **semiseparable** (best of both worlds)

This is true **duality**: not competing methods, but different ways to compute the same thing.

---

# 1. The Problem: Two Paradigms

## Transformers (Attention-Based)

Compare every word with every other word in the sequence.

- **Cost:** $\mathcal{O}(T^2)$ — doubling sequence length quadruples computation
- **Parallelism:** Strong parallel processing on GPUs
- **Limitation:** Expensive memory for long sequences

## State Space Models (Recurrent)

Maintain a "memory buffer" that updates one word at a time.

- **Cost:** $\mathcal{O}(T)$ — scales linearly with sequence length  
- **Memory:** Very efficient
- **Limitation:** Sequential processing, poor GPU utilization

## The Central Question

*Are these fundamentally different models, or two ways to compute the same thing?*

**Answer (Dao & Gu, 2024):** They are dual algorithms computing the same mathematical structure.

**Result:** Mamba-2 achieves **2-8X speedup** — combining SSM efficiency with attention-style parallelism.

---

## 2. State Space Models Explained

## How SSMs Work

An SSM maintains a hidden state vector that evolves through the sequence. At each timestep $t$:

$$h_t = A_t h_{t-1} + B_t x_t$$

$$y_t = C_t^T h_t$$

Where:
- $x_t \in \mathbb{R}^d$ — input at timestep $t$
- $h_t \in \mathbb{R}^N$ — hidden state (memory buffer)
- $y_t \in \mathbb{R}^p$ — output at timestep $t$
- $A_t \in \mathbb{R}^{N \times N}$ — state transition matrix
- $B_t \in \mathbb{R}^{N \times d}$ — input injection matrix
- $C_t \in \mathbb{R}^{N \times p}$ — output projection matrix

## Unrolled View

Each output word is influenced by all previous inputs through the chain of state transitions:

$$y_t = \sum_{s=1}^{t} C_t^T \left(\prod_{k=s+1}^{t} A_k\right) B_s x_s$$

## Why This Works Well

- **Memory:** Only store the hidden state, not full history
- **Speed:** Linear scaling with sequence length
- **Streaming:** Works naturally with streaming input

## Why It Struggles

- **Sequential:** Must process words one at a time — cannot parallelize
- **GPU Underutilization:** GPUs designed for parallel matrix operations, not sequential loops

---

## 3. Attention Mechanisms Explained

## How Attention Works

For each word, compute how relevant every other word is, then create a weighted average. Three steps:

1. **Project input:** Convert words to queries, keys, and values

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

2. **Compute attention:** Similarity scores between queries and keys

$$S = \frac{QK^T}{\sqrt{d_k}}$$

3. **Aggregate:** Weighted average of values using attention scores

$$A = \text{softmax}(S), \quad Y = AV$$

For each output position $t$:

$$y_t = \sum_{s=1}^{T} \alpha_{t,s} v_s$$

where $\alpha_{t,s} = \frac{\exp(q_t \cdot k_s / \sqrt{d_k})}{\sum_{r=1}^T \exp(q_t \cdot k_r / \sqrt{d_k})}$

## Mathematical View

Attention creates a $T \times T$ matrix where entry $(i,j)$ represents "how much does word $j$ influence word $i$?"

## Why This Works Well

- **Expressiveness:** Each word can attend to any other word
- **Parallelism:** Compute all attention scores simultaneously
- **Flexibility:** Learn which positions are important

## Why It Struggles

- **Quadratic Cost:** $\mathcal{O}(T^2)$ — becomes prohibitive for long sequences
- **Memory:** Must store entire $T \times T$ attention matrix

---

## 4. The Mathematical Bridge: Structured Matrices

## Unified View: The Core Insight

Both models compute the same operation: 

$$\textbf{Output} = M \times \textbf{Input}$$

where $M$ is a $T \times T$ matrix describing **"how each input position affects each output position."**

**For Attention:** $M$ is **dense** (fully filled) — every input can affect every output

**For SSMs:** $M$ is **lower-triangular causal** (future cannot affect past) with special structure:

$$M_{t,s} = \begin{cases}
C_t^T \left(\prod_{k=s+1}^{t} A_k\right) B_s & \text{if } s \leq t \\
0 & \text{if } s > t
\end{cases}$$

This represents: Input at position $s$ flows through state transitions to reach output at position $t$.

---

## The Key Insight: 1-Semiseparable Matrix Factorization

A **1-semiseparable matrix** has a special low-rank structure in its lower triangle. Instead of storing $T^2$ values, we factor it:

$$M = \begin{pmatrix}
d_1 & 0 & 0 & \cdots \\
u_2 v_1 & d_2 & 0 & \cdots \\
u_3 v_1 & u_3 v_2 & d_3 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

Each element is computed as:

$$M_{i,j} = \begin{cases}
d_i & \text{if } i = j \text{ (diagonal)} \\
u_i v_j & \text{if } i > j \text{ (lower triangle)} \\
0 & \text{if } i < j \text{ (upper triangle)} 
\end{cases}$$

## Understanding $u_i v_j$: What Does This Mean?

The key is the **outer product** structure:

$$M_{i,j} = u_i v_j = \begin{pmatrix} u_i^{(1)} \\ u_i^{(2)} \\ \vdots \\ u_i^{(d)} \end{pmatrix} \left[ v_j^{(1)}, v_j^{(2)}, \ldots, v_j^{(d)} \right]$$

**Intuitive interpretation:**

- **$v_j$:** How much does position $j$ "broadcast" its information forward? (the input's influence)
- **$u_i$:** How much does position $i$ "absorb" past information? (how receptive the output is)
- **$u_i v_j$:** The influence of position $j$ on position $i$ is the **product** of:
  - How strongly $j$ sends information forward ($v_j$)
  - How strongly $i$ receives past information ($u_i$)

**Analogy:** Think of a communication network:
- $v_j$ = how loudly person $j$ broadcasts their message
- $u_i$ = how much person $i$ listens to broadcasts from earlier people
- $u_i v_j$ = how much person $i$ actually hears from person $j$

**Why this is powerful:**

Instead of storing a full $T \times T$ matrix of interactions, we only store:
- Vectors $u_1, \ldots, u_T \in \mathbb{R}^d$ (how each position absorbs information)
- Vectors $v_1, \ldots, v_T \in \mathbb{R}^d$ (how each position broadcasts information)
- Diagonal $d_1, \ldots, d_T \in \mathbb{R}$ (direct connections)

This reduces storage from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$.

---

## Why This Structure Enables SSD: The Blocking Technique

The semiseparable structure enables a clever decomposition into **blocks** that can be processed in parallel:

### Step 1: Partition into Blocks

Divide the sequence into $B$ blocks of size $Q$:

$$\text{Sequence} = \underbrace{[x_1, \ldots, x_Q]}_{\text{Block 1}} \quad \underbrace{[x_{Q+1}, \ldots, x_{2Q}]}_{\text{Block 2}} \quad \cdots \quad \underbrace{[x_{(B-1)Q+1}, \ldots, x_T]}_{\text{Block B}}$$

The matrix $M$ decomposes into **blocks**:

$$M = \begin{pmatrix}
M_{1,1} & 0 & 0 & \cdots \\
M_{2,1} & M_{2,2} & 0 & \cdots \\
M_{3,1} & M_{3,2} & M_{3,3} & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

where each $M_{i,j}$ is a $Q \times Q$ block.

### Step 2: Intra-Block Parallelism

**Within each block**, the matrix is small ($Q \times Q$) and can be **computed in parallel** using GPU tensor cores:

$$y_j = M_{j,j} x_j + \text{(carry-over from block } j-1\text{)}$$

Since $Q$ is small (e.g., 64-128), this is **fast and parallelizable**.

**Cost per block:** $\mathcal{O}(Q^2 d)$ — all $B$ blocks process in parallel

### Step 3: Inter-Block Recurrence

Between blocks, we maintain a **compressed state** that flows forward:

$$S_j = \alpha_j S_{j-1} + R_j$$

where:
- $\alpha_j$ is the accumulated decay across block $j$
- $R_j$ is the local contribution from block $j$
- **Key:** Only $B = T/Q$ sequential steps, not $T$ steps!

For $T = 32K$ tokens and $Q = 64$: only $500$ sequential steps instead of $32K$.

**Cost for recurrence:** $\mathcal{O}(Bd^2) = \mathcal{O}(Td^2/Q)$ — negligible

### Why This Works: The Total Cost

$$\text{Total Cost} = \underbrace{\mathcal{O}(Q^2 d) \text{ (in parallel)}}_{\text{Intra-block}} + \underbrace{\mathcal{O}(Td^2/Q) \text{ (sequential)}}_{\text{Inter-block}}$$

For optimal choice $Q \approx \sqrt{T}$:

$$\text{Total Cost} \approx \mathcal{O}(T \cdot d)$$

This is **linear in sequence length** while maintaining **high GPU utilization**.

---

## Key Properties of Semiseparable Structure

**Storage efficiency:**
- Dense attention: $\mathcal{O}(T^2)$ parameters
- 1-semiseparable: $\mathcal{O}(T)$ parameters ✓

**Computation efficiency:**
- Dense matrix-vector: $\mathcal{O}(T^2 d)$
- 1-semiseparable: $\mathcal{O}(T d)$ ✓

**Parallelism via blocking:**
- Intra-block operations: Fully parallel
- Inter-block recurrence: Only $T/Q$ steps ✓

**Unification:**
- SSMs naturally express as semiseparable when parameters are structured
- Attention can be approximated with semiseparable structure
- Both achieve efficiency through same factorization ✓

---

## Concrete Example: 1-Semiseparable Matrix in Action

To understand why semiseparable matrices are so powerful, let's walk through a concrete example.

### Example Setup

Consider a small matrix with $T = 4$ time steps:

**The factors:**
- Absorption vectors: $u = [1, 2, 3, 4]$
- Broadcasting vectors: $v = [0.5, 0.3, 0.2, 0.1]$
- Diagonal: $d = [5, 6, 7, 8]$

### Build the Full Matrix

Using the factorization rule: $M_{i,j} = d_i$ (if $i=j$), $u_i v_j$ (if $i > j$), $0$ (if $i < j$)

$$M = \begin{pmatrix}
5 & 0 & 0 & 0 \\
1.0 & 6 & 0 & 0 \\
1.5 & 0.9 & 7 & 0 \\
2.0 & 1.2 & 0.8 & 8
\end{pmatrix}$$

**Storage:** Full matrix needs 16 numbers. Our factors need only 12 numbers.

For $T = 1000$: Dense needs 1 million numbers vs Semiseparable needs 3,000. That's **333× compression**!

### Computing Matrix-Vector Product: Two Approaches

**Input:** $x = [1, 2, 3, 4]^T$

#### Dense Approach (What Attention Does)

Multiply full $4 \times 4$ matrix by vector:

$$y_1 = 5(1) = 5$$
$$y_2 = 1(1) + 6(2) = 13$$
$$y_3 = 1.5(1) + 0.9(2) + 7(3) = 24.3$$
$$y_4 = 2(1) + 1.2(2) + 0.8(3) + 8(4) = 38.8$$

**Operations:** 16 multiplications needed

#### Efficient Approach (What SSMs Do)

Use the left-center-right algorithm:

```
acc = 0
for i = 1 to 4:
  y[i] = d[i] * x[i] + u[i] * acc
  acc = acc + v[i] * x[i]
```

**Execution:**

*i=1:* $y[1] = 5 \cdot 1 + 1 \cdot 0 = 5$, then $\text{acc} = 0 + 0.5 \cdot 1 = 0.5$

*i=2:* $y[2] = 6 \cdot 2 + 2 \cdot 0.5 = 13$, then $\text{acc} = 0.5 + 0.3 \cdot 2 = 1.1$

*i=3:* $y[3] = 7 \cdot 3 + 3 \cdot 1.1 = 24.3$, then $\text{acc} = 1.1 + 0.2 \cdot 3 = 1.7$

*i=4:* $y[4] = 8 \cdot 4 + 4 \cdot 1.7 = 38.8$, then $\text{acc} = 1.7 + 0.1 \cdot 4 = 2.1$

**Result:** $y = [5, 13, 24.3, 38.8]^T$ ✓ (same as dense!)

**Operations:** 8 multiplications needed (2x faster)

### Scaling to Real Problem Sizes

For $T = 32,000$ tokens:

| Metric | Dense Matrix | Semiseparable |
|:---|---:|---:|
| Storage | 1,024,000,000 parameters | 96,000 parameters |
| Matvec ops | 1,024,000,000 | 64,000 |
| Speedup | — | **16,000×** |

This is why Mamba-2 processes 32K token sequences efficiently while dense attention struggles.

### Why SSMs Have This Structure Naturally

In SSMs with state size $N$:

- Output projection $C_i \in \mathbb{R}^{N}$ gives the "$u$" factors
- State transitions $A_{j+1} \cdots A_i B_j \in \mathbb{R}^{N}$ give the "$v$" factors
- Direct term $C_i B_i$ gives the diagonal

Since $N \ll T$ (state is much smaller than sequence), this factorization is **natural and automatic**—not an approximation!

---

## 5. Formal Algorithms (A–E)

## Algorithm A: Self-Attention (Transformer)

```
Algorithm:

1. Q ← X W_Q  ▷ Query projection
2. K ← X W_K  ▷ Key projection
3. V ← X W_V  ▷ Value projection
4. S ← (Q K^T) / √d_k  ▷ Compute all attention scores
5. A ← softmax(S)  ▷ Normalize rows to get attention weights
6. output ← A V  ▷ Weighted sum of values
7. output ← output W_o  ▷ Final projection
8. return output
```

**Mathematical formulation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

For each output position $t$:

$$y_t = \sum_{s=1}^{T} \alpha_{t,s} v_s$$

where attention weights are:

$$\alpha_{t,s} = \frac{\exp(q_t \cdot k_s / \sqrt{d_k})}{\sum_{r=1}^T \exp(q_t \cdot k_r / \sqrt{d_k})}$$

**Complexity:** $\mathcal{O}(T^2 \cdot d)$ — quadratic scaling makes this expensive for long sequences

---

## Algorithm B: SSM (Recurrent Linear Form)

```
Algorithm:

1. h ← 0  ▷ Initialize hidden state
2. for t = 1, ..., T do
3.     h ← A_t · h + B_t · x_t  ▷ State update
4.     y_t ← C_t^T · h  ▷ Output projection
5. end for
6. return (y_1, ..., y_T)
```

**Mathematical formulation:**

$$h_t = A_t h_{t-1} + B_t x_t$$

$$y_t = C_t^T h_t$$

**Complexity:** $\mathcal{O}(T \cdot N)$ where $N$ is state size — linear but sequential

---

## Algorithm C: Mamba-1 (Selective SSM)

```
Algorithm:

1. h ← 0  ▷ Initialize hidden state
2. for t = 1, ..., T do
3.     A_t ← f_A(x_t)  ▷ Input-dependent transition
4.     B_t ← f_B(x_t)  ▷ Input-dependent input gate
5.     C_t ← f_C(x_t)  ▷ Input-dependent output gate
6.     h ← A_t · h + B_t · x_t  ▷ Update with data-dependent parameters
7.     y_t ← C_t^T · h  ▷ Output projection
8. end for
9. return (y_1, ..., y_T)
```

**Key innovation:** Parameters $A_t, B_t, C_t$ depend on input $x_t$

**Benefit:** Selective memory — model learns what to remember

**Limitation:** Still sequential — only 18% GPU utilization

---

## Algorithm D: Mamba-2 (Structured State Space Duality)

The breakthrough: Use structured matrix decomposition to enable parallelization.

```
Algorithm:

Input: X ∈ ℝ^(T×d), a_1,...,a_T ∈ ℝ, B_1,...,B_T, C_1,...,C_T, block size Q

// Part 1: Intra-block computation (PARALLELIZABLE)
1. B ← ⌈T/Q⌉  ▷ Number of blocks
2. I_j ← [j·Q : (j+1)·Q - 1] for j = 0,...,B-1  ▷ Block index ranges
3. for j = 0 to B-1 in parallel do
4.     G_j ← C[I_j] · B[I_j]^T  ▷ Dense kernel: Q × Q
5.     L_j ← BuildSemiseparableMask(a[I_j])  ▷ 1-SS mask: Q × Q
6.     M_j ← L_j ⊙ G_j  ▷ Hadamard product (element-wise)
7.     Y[I_j] ← M_j · X[I_j]  ▷ Apply structured kernel
8. end for

// Part 2: Inter-block recurrence (SEQUENTIAL across B blocks)
9. for j = 0 to B-1 do
10.    S_right[j] ← B[I_j]^T · X[I_j]  ▷ Block-level state
11. end for

12. S[0] ← S_right[0]  ▷ State after block 0
13. for j = 1 to B-1 do
14.    α_j ← ∏_{k=j·Q}^{(j+1)·Q-1} a_k  ▷ Cumulative decay within block
15.    S[j] ← α_j · S[j-1] + S_right[j]  ▷ State after block j
16. end for

// Part 3: Broadcast inter-block contributions
17. for j = 1 to B-1 do
18.    Y[I_j] ← Y[I_j] + C[I_j] · S[j-1]  ▷ Add carry-in from previous block
19. end for

20. return Y
```

**Mathematical breakdown:**

For each block $j$, compute the structured kernel:

$$M_j^{(i,k)} = C_i \left(\prod_{\ell=k+1}^{i} a_\ell\right) B_k, \quad i,k \in I_j$$

Output from block $j$:

$$Y_j = M_j X_j + C_j S_{j-1}$$

where $S_j$ is the carry-over state from block $j-1$.

**Complexity:** 
- Intra-block: $\mathcal{O}(Q^2 d)$ per block, all parallel
- Inter-block: $\mathcal{O}(B \cdot d^2) = \mathcal{O}(Td^2/Q)$ sequential
- **Total:** $\mathcal{O}(Q^2 d + Td^2/Q) \approx \mathcal{O}(T \cdot d)$ with good constants

---

## Algorithm E: Efficient 1-Semiseparable Matrix-Vector Product

```
Algorithm:

Input: x ∈ ℝ^T, u_1,...,u_T, v_1,...,v_T ∈ ℝ^d, d_1,...,d_T ∈ ℝ

1. acc ← 0  ▷ Accumulator in ℝ^d
2. for i = 1 to T do
3.     y[i] ← d[i] · x[i] + u[i] · acc  ▷ Diagonal + left factor × accumulation
4.     acc ← acc + v[i] · x[i]  ▷ Update accumulation
5. end for
6. return y
```

**Interpretation:**
- $d[i] \cdot x[i]$: Direct connection from input to output
- $\text{acc} = \sum_{j < i} v_j x_j$: Running summary of past inputs
- $u[i] \cdot \text{acc}$: How much past history influences current output
- $v[i]$: How much current input contributes to future

**Complexity:** $\mathcal{O}(T \cdot d)$ with very small constants

---

## 6. Mamba-1 vs Mamba-2: Architecture Comparison

## Mamba-1 (Sequential)

```
Input → Projections → Conv1d → Sequential SSM Update → Gate → Output
         (all inputs)          (one word at a time)   (SiLU)
```

- Sequential processing limits parallelism
- GPU utilization: ~18%
- Simple but inefficient for long sequences

## Mamba-2 (SSD with Parallelism)

```
Input → Projections → Conv1d → Parallel Structured SSM → Gate + GroupNorm → Output
         (grouped)            (block-wise parallelism)
```

- Grouped projections enable multi-device parallelism
- Block-wise processing for parallel computation
- Group normalization for stability across parallel execution
- GPU utilization: 70-80%

## Key Differences

| **Aspect** | **Mamba-1** | **Mamba-2 (SSD)** |
|------------|-------------|------------------|
| Processing | Sequential per token | Parallel per block |
| GPU Utilization | 18% | 70-80% |
| Projections | Per-token | Grouped/shared |
| Normalization | None | Group norm |
| Speedup | Baseline | 2-8X faster |
| Accuracy (avg) | 59.9% | 60.2% |

---

## 7. Real-World Performance

## Speed Comparison (A100 GPU)

| **Sequence Length** | **FlashAttention-2** | **Mamba-1** | **Mamba-2** | **Speedup** |
|:---:|:---:|:---:|:---:|:---:|
| 512 tokens | 0.20 ms | 0.15 ms | 0.15 ms | 1X |
| 1K tokens | 0.50 ms | 0.30 ms | 0.25 ms | 2X |
| 4K tokens | 5.00 ms | 1.20 ms | 0.80 ms | 6X |
| 16K tokens | 40 ms | 5.0 ms | 3.0 ms | 8X |
| 32K tokens | 120 ms | 10 ms | 6 ms | 10X |

**Key insight:** Mamba-2 gets faster relative to attention as sequences get longer (linear vs quadratic scaling).

## Language Modeling Quality

| **Model** | **Pile Loss** | **LAMBADA** | **HellaSwag** | **PIQA** | **ARC-E** | **ARC-C** | **Average** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pythia | 6.73 | 64.7 | 59.3 | 74.0 | 64.1 | 32.9 | 55.7% |
| Mamba-1 | 6.22 | 69.2 | 66.1 | 75.2 | 69.7 | 36.3 | 59.9% |
| **Mamba-2** | **6.09** | **69.7** | **66.6** | **76.4** | **69.6** | **36.4** | **60.2%** |

**Result:** Better quality + faster execution

## Memory Task Performance

Multi-Query Associative Recall (MQAR): Given $N$ query-key-value triplets, retrieve correct value

| **Architecture** | **Memory Size** | **Accuracy** |
|:---:|:---:|:---:|
| Attention | — | 95% |
| Mamba-1 | 16 | 20% |
| Mamba-2 | 64 | 90% |
| **Mamba-2** | **256** | **98%** |

---

## 8. Why This Unification Matters

## Three Key Insights

### 1. Algorithm ≠ Representation

The same mathematical transformation can be computed via:

- **Sequential recurrence (SSM-style):** Good for streaming, uses little memory
- **Parallel matrix multiply (Attention-style):** Good for parallelism, uses more memory  
- **Structured blocks (Mamba-2 SSD):** Good for both

This is analogous to computing a sum: you can add left-to-right sequentially, or split into groups and parallelize.

### 2. Structure is Power

When the transformation matrix has special structure (low-rank patterns, decay, sparsity), you can:

- Store it efficiently: $\mathcal{O}(T)$ instead of $\mathcal{O}(T^2)$
- Compute with it efficiently: $\mathcal{O}(T)$ instead of $\mathcal{O}(T^2)$
- Parallelize computation while maintaining efficiency

Real sequences have this structure naturally — recent context matters more than distant.

### 3. Future Hybrid Models

Rather than choosing SSM or attention:

- Use attention for reasoning tasks (irregular dependencies)
- Use SSD for memory/efficiency (structured dependencies)
- Blend both in same model

---

## 9. Discussion Questions

<details>
<summary><strong>Q1: Why is $\mathcal{O}(T \cdot N)$ generally more favorable than $\mathcal{O}(T^2)$ for long sequences?</strong></summary>

Attention's $\mathcal{O}(T^2)$ scaling becomes prohibitive for long sequences: doubling $T$ quadruples compute and memory requirements.

SSMs/SSD with $\mathcal{O}(T \cdot N)$ scaling allows:
- 100K+ context lengths with consistent latency
- Linear memory growth with sequence length
- Streaming inference without full history

SSD maintains high GPU utilization by using structured parallelism (matrix cores operating over 1-semiseparable blocks), bridging the efficiency gap between sequential and parallel algorithms.

**Practical implication:** For $T > 10K$ tokens, SSD becomes strictly better than dense attention on both speed and memory.

</details>

<details>
<summary><strong>Q2: If SSD is more efficient, why do we still use dense Attention and sequential SSMs?</strong></summary>

**Attention excels at:**
- Reasoning tasks requiring non-local lookups
- Irregular dependencies (e.g., coreference resolution)
- Short contexts where $\mathcal{O}(T^2)$ is acceptable

**Sequential SSMs excel at:**
- Streaming inference (causal, low latency)
- Edge deployment (minimal memory)
- Hardware with no parallelism support

**SSD excels at:**
- Long sequences (both speed and memory)
- Structured dependencies (recent context > distant)
- Modern hardware with tensor cores

**Future direction:** Hybrid architectures combining all three:
- Attention layers for reasoning/comprehension
- SSD layers for memory/retrieval efficiency
- SSM layers for streaming/online updates

The choice depends on task structure and hardware constraints, not on a single "best" approach.

</details>

---

## 10. Summary

## Main Results

**Theoretical:** SSMs and attention are dual representations of the same transformation, connected through structured semiseparable matrices.

**Practical:** Mamba-2 achieves 2-8X speedup while maintaining language modeling quality through clever block-level parallelism.

**Architectural:** Different models emerge from choosing different algorithms for structured matrices, not from fundamentally different representations.

## Takeaways

1. **Linear scaling is achievable** without sacrificing expressiveness via structured matrices
2. **Parallelism and efficiency can coexist** through algorithmic duality
3. **Hybrid models are promising** — blend approaches based on task structure
4. **Hardware matters** — SSD designed for modern GPU tensor cores

## Impact

- Longer sequences tractable without quadratic memory
- Streaming inference improved through efficient block recurrence
- Foundation for next-generation efficient architectures
- Opens research in hardware-algorithm co-design

---

## References

1. Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. arXiv:2405.21060

2. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752

3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.

4. Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022.

5. Vandebril, R., Van Barel, M., & Golub, G. (2008). *Matrix Computations and Semiseparable Matrices.* Johns Hopkins Press.

---

**Contact:** ryan.li@vanderbilt.edu