# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Presenter:** Ryan Li  
**Email:** ryan.li@vanderbilt.edu  
**Institution:** Vanderbilt University – Data Science Institute  
**Paper:** Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (ICML 2024)

---

## 0. Talk Roadmap (13–15 minutes)

1. Motivation: runtime & why Transformers / SSMs matter  
2. Mathematics of SSMs (2.1)  
3. Mathematics of Attention (2.2)  
4. Why structured matrices matter (2.3)  
5. Recurrent linear vs dual quadratic forms (2.4)  
6. Matrix form of SSMs (Eq. 3), semiseparable & 1-SS matrices (3.1–3.2)  
7. Formal algorithms:
   - Attention layer
   - SSM layer
   - Mamba-1
   - Mamba-2 (SSD)
   - 1-SS (left/center/right) multiplication
8. Two runtime-focused SSD questions (collapsible)  
9. References

---

## 1. Motivation: Runtime & Importance

Let:

- $T$ = sequence length  
- $d$ = model dimension  
- $N$ = SSM state dimension  

**Transformers / Attention**

- Power almost all modern LLMs.
- Complexity: building attention weights costs
  \[
  \mathcal{O}(T^2 d),
  \]
  memory also $\mathcal{O}(T^2)$.
- Fantastic parallelism, painful when $T$ is large (long-context, code, documents).

**State Space Models (SSMs)**

- Come from control/dynamical systems.
- Maintain a compressed state of the past.
- Complexity:
  \[
  \mathcal{O}(T N)
  \]
  for structured variants → ideal scaling.
- Historically implemented as **sequential scans**, so hardware (GPU) is under-utilized.

**Central SSD perspective**

> Attention and SSMs are not opposites; they are two runtime points on the same operator family. SSD explains how to move between them using **structured matrices**, getting both:  
> - attention-like expressivity & parallelism,  
> - SSM-like linear runtime.

---

## 2. Mathematics

### 2.1 State Space Models (SSMs)

We track a latent state $h_t \in \mathbb{R}^N$:

\[
\begin{aligned}
h_t &= A_t h_{t-1} + B_t x_t, \\
y_t &= C_t^\top h_t,
\end{aligned}
\]

with:

- $x_t \in \mathbb{R}^d$: input,
- $y_t \in \mathbb{R}^p$: output,
- $A_t \in \mathbb{R}^{N \times N}$: transition (how memory propagates),
- $B_t \in \mathbb{R}^{N \times d}$: input write,
- $C_t \in \mathbb{R}^{N \times p}$: readout.

**Unrolled form**

For $t \ge s$:

\[
y_t
= \sum_{s=1}^{t}
C_t^\top
\left(
  \prod_{k=s+1}^{t} A_k
\right)
B_s x_s.
\]

Interpretation:

- Each $x_s$ is written into state via $B_s$.
- It is transported forward by the product of transitions $\prod_{k=s+1}^t A_k$.
- It is read out at time $t$ via $C_t^\top$.

**Runtime view**

- Direct evaluation via recurrence: one update per $t$ → $\mathcal{O}(T)$ multiplications (modulo $N$).
- Extremely memory-efficient (keep only $h_t$).
- But this update is inherently sequential across $t$.

---

### 2.2 Attention Mechanisms

Given $X \in \mathbb{R}^{T \times d_{\text{model}}}$:

\[
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V.
\]

Scores:

\[
S = \frac{Q K^\top}{\sqrt{d_k}},
\]

Row-wise softmax:

\[
A_{t,s} = \frac{\exp(S_{t,s})}{\sum_{r=1}^T \exp(S_{t,r})}.
\]

Output:

\[
Y = A V,\quad
y_t = \sum_{s=1}^T A_{t,s} v_s.
\]

Interpretation:

- $A_{t,s}$ tells us how much token $t$ attends to token $s$.
- The full $T \times T$ matrix $A$ is a **dense kernel over positions**.

**Runtime view**

- Forming $QK^\top$ and multiplying by $V$ costs $\mathcal{O}(T^2 d)$.
- All positions are updated in parallel (matrix multiplies).
- Ideal for GPUs, but becomes dominating cost when $T$ is large.

---

### 2.3 Why Structured Matrices Matter

Any *linear* sequence map can be written as:

\[
Y = M X,
\]

where $M \in \mathbb{R}^{T \times T}$ and

\[
(MX)_t = \sum_{s=1}^T M_{t,s} x_s.
\]

- Attention: $M = A$ is dense, learned implicitly via $Q,K$ and softmax.
- SSM: from 2.1,

  \[
  M_{t,s} =
  \begin{cases}
  C_t^\top \left( \displaystyle\prod_{k=s+1}^{t} A_k \right) B_s,
  & s \le t, \\[4pt]
  0, & s > t.
  \end{cases}
  \]

So SSM kernels are:

1. **Causal (lower triangular)**.
2. Built from structured products and low-rank factors.

**Structured matrices (e.g., semiseparable, Toeplitz, low-rank)** give us:

- Fewer parameters: often $\mathcal{O}(T)$ or $\mathcal{O}(T r)$ vs $\mathcal{O}(T^2)$.
- Faster matvec: can be evaluated in (near-)linear time.
- A way to keep attention-like *global* dependencies, while paying SSM-like runtime.

SSD’s key move: recognize SSM-induced $M$ as a **semiseparable matrix**, so we can evaluate the associated operator efficiently and in parallel.

---

### 2.4 Recurrent Linear vs Dual Quadratic Forms

SSMs admit two mathematically equivalent but computationally different views:

#### (a) Recurrent Linear Form

\[
\begin{aligned}
h_t &= A_t h_{t-1} + B_t x_t,\\
y_t &= C_t^\top h_t.
\end{aligned}
\]

- Complexity: $\mathcal{O}(T N^2)$ in general, $\mathcal{O}(T N)$ with structured $A_t$.
- Streaming-friendly, tiny memory.
- Poor sequence-parallelism: strict dependence on $h_{t-1}$.

#### (b) Dual Quadratic Form (Kernel Form)

Define

\[
K(t,s) =
\begin{cases}
C_t^\top \left( \displaystyle\prod_{k=s+1}^{t} A_k \right) B_s,
& s \le t, \\[6pt]
0, & s > t.
\end{cases}
\]

Then:

\[
y_t = \sum_{s=1}^{T} K(t,s) x_s, \quad
Y = K X.
\]

- Naively costs $\mathcal{O}(T^2)$ to materialize and apply.
- But: fully parallelizable (just matrix multiplications).
- Structurally very close to attention: both are “all-pairs with a kernel”.

**Contrast focusing on runtime & SSD:**

- Recurrent form:
  - Efficient in $T$, inefficient for GPUs (sequential).
- Dual form:
  - Perfect for GPUs, bad in $T$ if unstructured.
- SSD:
  - Observe $K$ is *structured semiseparable* ⇒ can compute $KX$ in *linear time* using algorithms that look like attention but avoid the $T^2$ blowup.

This is the runtime heart of SSD.

---

## 3. Matrix Form (Eq. 3), Semiseparable & 1-SS

### 3.1 Equation (3): Matrix Transformation Form of SSMs

SSD writes the SSM as:

\[
Y = M X,
\]

with

\[
M_{t,s} =
\begin{cases}
C_t^\top \left( \displaystyle\prod_{k=s+1}^{t} A_k \right) B_s,
& s \le t, \\[6pt]
0, & s > t.
\end{cases}
\tag{3}
\]

Intuition:

- Row $t$ of $M$ = “how do all earlier inputs $x_s$ flow into $y_t$?”
- Column $s$ of $M$ = “how does $x_s$ influence all future outputs?”
- The transition product is the **transport operator**; $B_s, C_t$ are write/read heads.
- Same $M$ underlies both:
  - SSM recurrence (efficient but sequential),
  - quadratic kernel (parallel but naive quadratic).

SSD’s job: exploit **structure in $M$**.

---

### 3.2 1-Semiseparable (1-SS) Matrices

A key structure in the paper is the **1-semiseparable (1-SS)** matrix.

A matrix $M \in \mathbb{R}^{T \times T}$ is 1-SS if there exist vectors
$u_1,\dots,u_T$, $v_1,\dots,v_T$, and diagonal $d_1,\dots,d_T$ such that:

\[
M_{i,j} =
\begin{cases}
d_i, & i = j,\\
u_i v_j, & i > j,\\
0, & i < j.
\end{cases}
\]

Example pattern:

\[
M =
\begin{pmatrix}
d_1 & 0 & 0 & \dots \\
u_2 v_1 & d_2 & 0 & \dots \\
u_3 v_1 & u_3 v_2 & d_3 & \dots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}.
\]

**Why 1-SS is powerful (runtime view):**

- Parameters: store $u$, $v$, $d$ → $\mathcal{O}(T)$ instead of $\mathcal{O}(T^2)$.
- Matvec can be done with a single pass:

  - Maintain accumulator
    \[
    r_i = r_{i-1} + v_i x_i
    \]
  - Then
    \[
    y_i = d_i x_i + u_i r_{i-1}.
    \]

  This is $\mathcal{O}(T)$.

**Connection to SSMs & SSD:**

- For scalar/diagonal $A_t$ and rank-1 $B_s, C_t$, the kernel $M$ in (3) becomes 1-SS.
- This means SSM kernels can be computed:
  - via recurrence (SSM),
  - or via one-pass semiseparable matvec (structured),
  - or via blocked parallel SSD algorithms.
- This structure is what lets Mamba-2 be both **fast in $T$** and **GPU-efficient**.

---

## 4. Formal Algorithms (Runtime-Aware)

### Algorithm A – Attention Layer

**Input:** $X \in \mathbb{R}^{T \times d_{\text{model}}}$, $W_Q, W_K, W_V$.  
**Output:** $Y \in \mathbb{R}^{T \times d_{\text{model}}}$.

1. $Q \leftarrow X W_Q,\quad K \leftarrow X W_K,\quad V \leftarrow X W_V$.
2. $S \leftarrow Q K^\top / \sqrt{d_k}$.
3. $A \leftarrow \mathrm{softmax}_{\text{row}}(S)$.
4. $Y \leftarrow A V$.

Runtime: $\mathcal{O}(T^2 d_k)$ (good parallelism, poor scaling in $T$).

---

### Algorithm B – SSM Layer (Recurrent Form)

**Input:** $x_1,\dots,x_T$, matrices $A_t, B_t, C_t$.  
**Output:** $y_1,\dots,y_T$.

1. Initialize $h_0 \leftarrow 0$.
2. For $t = 1 \dots T$:
   - $h_t \leftarrow A_t h_{t-1} + B_t x_t$.
   - $y_t \leftarrow C_t^\top h_t$.
3. Return $y_1,\dots,y_T$.

Runtime: $\mathcal{O}(T N^2)$ or $\mathcal{O}(T N)$; sequential across $t$.

---

### Algorithm C – Mamba-1 (Selective SSM)

**Idea:** Make $A_t, B_t, C_t$ depend on $x_t$ for selective memory; still evaluated via scan.

1. For each $t$:
   - $A_t \leftarrow f_A(x_t)$,
   - $B_t \leftarrow f_B(x_t)$,
   - $C_t \leftarrow f_C(x_t)$.
2. Apply Algorithm B with these.
3. Output $y_{1:T}$.

Runtime: linear in $T$, but low hardware utilization due to sequential scan.

---

### Algorithm D – Mamba-2 / SSD (Blocked Semiseparable)

**Goal:** Evaluate the same SSM-style operator using 1-SS/SSD structure with high parallelism.

Let $T = B Q$ (blocks of length $Q$), with parameters $a_t$ (scalar decays), $B_t$, $C_t$.

**Outline:**

1. **Intra-block (parallel)**
   - For each block $\mathcal{I}_j = [jQ, (j+1)Q-1]$ in parallel:
     - Compute Gram:
       \[
       G_j = C_{\mathcal{I}_j} B_{\mathcal{I}_j}^\top.
       \]
     - Build causal mask $L_j$ using products of $a_t$ within block.
     - Form $M_j = L_j \circ G_j$.
     - Update $Y_{\mathcal{I}_j} \mathrel{+}= M_j X_{\mathcal{I}_j}$.

2. **Right factor (block compression)**
   - For each block:
     \[
     S^{\text{right}}_j = B_{\mathcal{I}_j}^\top X_{\mathcal{I}_j}.
     \]

3. **Center factor (block recurrence)**
   - $S_0 = S^{\text{right}}_0$.
   - For $j = 1,\dots,B-1$:
     \[
     \alpha_j = \prod_{t \in \mathcal{I}_j} a_t, \quad
     S_j = \alpha_j S_{j-1} + S^{\text{right}}_j.
     \]

4. **Left factor (inject global state)**
   - For $j = 1,\dots,B-1$:
     \[
     Y_{\mathcal{I}_j} \mathrel{+}= C_{\mathcal{I}_j} S_{j-1}.
     \]

**Runtime view:**

- Most work: block-local matrix multiplies → highly parallel, GPU-friendly.
- Only $B = T/Q$ recurrence steps.
- Achieves near-linear scaling in $T$ with high utilization.

---

### Algorithm E – 1-SS Matrix–Vector (Left/Center/Right)

For 1-SS $M$ given by $u_i, v_i, d_i$:

1. Set accumulator $r \leftarrow 0$.
2. For $i = 1 \dots T$:
   - $y_i \leftarrow d_i x_i + u_i r$.
   - $r \leftarrow r + v_i x_i$.
3. Return $y_{1:T}$.

This is the canonical left/center/right factorization:
- **Right:** accumulate $v_i x_i$.
- **Center:** running sum $r$.
- **Left:** scale by $u_i$.
Runtime: $\mathcal{O}(T)$.

---

## 5. Two SSD Questions (Runtime-Focused)

<details>
<summary><strong>Q1 — In terms of runtime, when is enforcing a structured (1-SS / semiseparable) kernel worth it, and when might it hurt?</strong></summary>

**Key angles:**

- Structured kernels reduce cost from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$–$\mathcal{O}(T r)$, fantastic for long contexts.
- But they impose inductive bias: monotone decays, low effective rank.
- They may underfit tasks needing dense, irregular, or bidirectional interactions where full attention’s $T^2$ work is justified.
- The SSD view encourages us to ask:
  - For a given application (code, retrieval, math reasoning), does the performance gain of full attention compensate for its runtime cost?
  - Can adaptive-rank or hybrid SSD–attention models select structure where it helps and relax it where it hurts?
</details>

<details>
<summary><strong>Q2 — Given SSD’s unification, how should we balance Attention vs SSM vs SSD in real systems?</strong></summary>

**Key angles:**

- Attention is still ideal for:
  - Short/medium contexts,
  - Highly irregular dependencies,
  - Where $T^2$ cost is acceptable.
- SSM-style recurrences are ideal for:
  - Streaming and ultra-long contexts,
  - Edge or low-resource deployment,
  - Strict latency constraints.
- SSD / Mamba-2 provides a **middle ground**:
  - Same underlying operator as SSM,
  - Implemented with attention-like parallel block kernels,
  - Nearly linear runtime in $T$ with strong throughput.
- Practical question for system designers:
  - Where in the stack do we keep classic attention,
    and where do we swap to SSD-style layers for runtime savings,
    without sacrificing task performance?
</details>

---

## 6. References (for further reading)

- Vaswani et al., “Attention Is All You Need”, NeurIPS 2017.  
- Gu & Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”, arXiv:2312.00752.  
- Dao & Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”, ICML 2024.  
- Literature on semiseparable matrices (Vandebril et al.) for structured linear algebra background.

---

**Thank you!**  
Questions: ryan.li@vanderbilt.edu
