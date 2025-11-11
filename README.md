# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Presenter:** Ryan Li  
**Email:** [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)  
**Institution:** Vanderbilt University – Data Science Institute  
**Paper:** Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (ICML 2024)

---

## 0. Talk Roadmap (≈ 15 min / 8 pages)

1. Motivation — Runtime & Duality  
2. Mathematics  
   - 2.1 State Space Models (SSMs)  
   - 2.2 Attention Mechanisms  
   - 2.3 Structured Matrices  
   - 2.4 Recurrent vs Dual Forms  
3. Matrix Form (Eq. 3) & 1-SS Matrices  
4. Formal Algorithms (A–E)  
5. Experiments & Performance Tables  
6. Architectural Comparison: Mamba-1 vs Mamba-2 (SSD)  
7. Two SSD Discussion Questions  
8. References  

---

## 1. Motivation — Runtime & Duality

Let $T$ = sequence length, $d$ = model dimension, $N$ = state size.

- **Transformers (Attention):**  
  Runtime ≈ $\mathcal{O}(T^2 d)$, Memory ≈ $\mathcal{O}(T^2)$  
  → Excellent parallelism, poor long-context scaling.

- **State Space Models (SSMs):**  
  Runtime ≈ $\mathcal{O}(T N)$ (structured), Memory ≈ $\mathcal{O}(N)$  
  → Linear in time but traditionally sequential, under-utilizing GPUs.

**Key Question:**  
Can we achieve the **linear-time efficiency of SSMs** *and* the **parallelism of attention**?

**SSD Answer:**  
Dao & Gu (2024) show that SSMs, Attention, and Structured Matrices are **algorithmic duals** — different runtime realizations of the same linear operator.

---

## 2. Mathematics

### 2.1 State Space Models (SSMs)

$$
\begin{aligned}
h_t &= A_t h_{t-1} + B_t x_t, \\
y_t &= C_t^{\top} h_t,
\end{aligned}
$$

with  
$A_t ∈ \mathbb{R}^{N×N}$ (transition),  
$B_t ∈ \mathbb{R}^{N×d}$ (input mapping),  
$C_t ∈ \mathbb{R}^{N×p}$ (readout).

**Unrolled Form**

$$
y_t = \sum_{s=1}^{t} C_t^{\top}\!
\left(\prod_{k=s+1}^{t} A_k\right)
B_s x_s.
$$

**Interpretation**

- $A_t$: temporal decay / transition  
- $B_t$: input projection  
- $C_t$: output projection  
- Linear in time → $\mathcal{O}(T N)$  
- Sequential updates hinder parallel hardware efficiency

---

### 2.2 Attention Mechanisms

Given $X ∈ \mathbb{R}^{T×d_{\text{model}}}$:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V.
$$

Scores and weights:

$$
S = \frac{QK^{\top}}{\sqrt{d_k}}, \qquad
A = \mathrm{softmax}_{\text{row}}(S),
$$

Output:

$$
Y = A V, \quad
y_t = \sum_{s=1}^{T} A_{t,s} v_s.
$$

**Interpretation**

- Attention = global kernel operator with learned similarity weighting.  
- Each $y_t$ depends on *all* $x_s$.  
- Perfectly parallel but $\mathcal{O}(T^2)$ in cost.  

---

### 2.3 Structured Matrices in Sequence Modeling

Any linear sequence model is expressible as:

$$
Y = M X,
$$

where $M ∈ \mathbb{R}^{T×T}$ and

$$
(MX)_t = \sum_{s=1}^{T} M_{t,s} x_s.
$$

For SSMs:

$$
M_{t,s} =
\begin{cases}
C_t^{\top}\!\left(\displaystyle\prod_{k=s+1}^{t} A_k\right)\!B_s, & s ≤ t,\\[6pt]
0, & s > t.
\end{cases}
$$

Properties:

- **Causality:** lower-triangular structure.  
- **Low-rank and semiseparable:** reuse across diagonals.  
- **Runtime benefit:** semiseparable form → $\mathcal{O}(T)$ matvec.  

---

### 2.4 Recurrent Linear vs Dual Quadratic Forms

#### Recurrent Linear Form (Sequential)

$$
\begin{aligned}
h_t &= A_t h_{t-1} + B_t x_t,\\
y_t &= C_t^{\top} h_t.
\end{aligned}
$$

- Streaming-friendly, $\mathcal{O}(T N)$  
- Sequential — difficult to parallelize

#### Dual Quadratic Form (Kernel)

$$
K(t,s) =
\begin{cases}
C_t^{\top}\!\left(\displaystyle\prod_{k=s+1}^{t} A_k\right)\!B_s, & s ≤ t,\\[6pt]
0, & s > t.
\end{cases}
$$

$$
y_t = \sum_{s=1}^{T} K(t,s) x_s, \quad Y = K X.
$$

- Same transformation, evaluated in parallel  
- If $K$ structured → $\mathcal{O}(T)$ evaluation  

**SSD Observation:**  
Recurrent and dual forms are *two orders of tensor contraction*; SSD bridges them with structure preserving efficiency.

---

## 3. Matrix Form (Eq. 3) & 1-Semiseparable Matrices

### 3.1 Equation (3): Matrix Transformation of SSMs

$$
Y = M X,\quad
M_{t,s} =
\begin{cases}
C_t^{\top}\!\left(\displaystyle\prod_{k=s+1}^{t} A_k\right)\!B_s, & s ≤ t,\\[6pt]
0, & s > t.
\end{cases}
\tag{3}
$$

Row $t$ = how previous inputs affect $y_t$  
Column $s$ = how $x_s$ influences future outputs  
→ structure enables both recurrence and blockwise parallelism.

---

### 3.2 Definition — 1-Semiseparable (1-SS) Matrices

A 1-SS matrix $M$ has factors $u_i, v_i, d_i$ such that:

$$
M_{i,j} =
\begin{cases}
d_i, & i = j,\\
u_i v_j, & i > j,\\
0, & i < j.
\end{cases}
$$

**Efficient matvec evaluation:**

$$
r_i = r_{i-1} + v_i x_i, \quad
y_i = d_i x_i + u_i r_{i-1}.
$$

→ $\mathcal{O}(T)$ runtime; $\mathcal{O}(T)$ storage.  
→ Used by SSD to parallelize state propagation.

---

## 4. Formal Algorithms (A–E)

### Algorithm A — Self-Attention
```
Input: X ∈ ℝ^{T×d},  W_Q, W_K, W_V
1. Q ← X·W_Q,  K ← X·W_K,  V ← X·W_V
2. S ← Q·Kᵀ / √d_k
3. A ← softmax_rows(S)
4. Y ← A·V
Output: Y
```
→ Fully parallel, $\mathcal{O}(T^2 d)$.

---

### Algorithm B — State Space Model (Recurrent)
```
Input: x₁…x_T, A₁…A_T, B₁…B_T, C₁…C_T
h ← 0
for t = 1…T do
    h ← A_t·h + B_t·x_t
    y_t ← C_tᵀ·h
return y₁…y_T
```
→ Linear time, sequential scan.

---

### Algorithm C — Mamba-1 (Selective SSM)
```
for t = 1…T do
    A_t ← f_A(x_t)
    B_t ← f_B(x_t)
    C_t ← f_C(x_t)
    h_t ← A_t·h_{t−1} + B_t·x_t
    y_t ← C_tᵀ·h_t
return y₁…y_T
```
→ Input-dependent transitions, $\mathcal{O}(T)$ runtime, 18 % GPU utilization.

---

### Algorithm D — Mamba-2 (SSD Blocked)
```
Partition T tokens into blocks of size Q
Parallel within blocks:
    Build 1-SS mask L_j using decay a_t
    Compute G_j = C_j·B_jᵀ, M_j = L_j ∘ G_j
    Y_block ← M_j·X_block
Sequential across blocks:
    Compress Right → S_j = B_jᵀ·X_j
    Propagate Center → S_j = α_j·S_{j−1} + S_j
    Expand Left → Y_j ← Y_j + C_j·S_{j−1}
return Y
```
→ Linear scaling with 70–80 % GPU efficiency.

---

### Algorithm E — 1-SS Left–Center–Right Matvec
```
r ← 0
for i = 1…T do
    y_i ← d_i·x_i + u_i·r
    r ← r + v_i·x_i
return y
```
→ Core SSD routine: linear time, one-pass recurrence.

---

## 5. Experiments & Numerical Results

### 5.1 Speed (Forward Latency, A100 GPU)

| Seq Len | FlashAttn-2 | Mamba-1 | **Mamba-2 (SSD)** | Speedup |
|:--:|:--:|:--:|:--:|:--:|
| 512 | 0.20 ms | 0.15 ms | 0.15 ms | 1× |
| 1 k | 0.50 ms | 0.30 ms | 0.25 ms | 2× |
| 4 k | 5.00 ms | 1.20 ms | **0.80 ms** | 6× |
| 16 k | 40 ms | 5.0 ms | **3.0 ms** | 8× |
| 32 k | 120 ms | 10 ms | **6 ms** | 10× |

GPU utilization: **18 % → 78 %** (Mamba-1 → Mamba-2)

---

### 5.2 Language Modeling Benchmarks

| Model | Params | Pile PPL↓ | LAMBADA | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | **Avg** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Pythia-2.8B | 2.8 B | 6.73 | 64.7 | 59.3 | 74.0 | 64.1 | 32.9 | 59.7 | 55.7 |
| Mamba-1 | 2.8 B | 6.22 | 69.2 | 66.1 | 75.2 | 69.7 | 36.3 | 63.5 | 59.9 |
| **Mamba-2 (SSD)** | **2.7 B** | **6.09** | **69.7** | **66.6** | **76.4** | **69.6** | **36.4** | **64.0** | **60.2** |

---

### 5.3 Memory Task — Multi-Query Associative Recall (MQAR)

| Model | N | Accuracy (%) |
|:--|:--:|:--:|
| Attention | – | 95 |
| Mamba-1 | 16 | 20 |
| Mamba-2 | 64 | 90 |
| **Mamba-2 (SSD)** | **256** | **98** |

---

## 6. Architectural Comparison — Mamba-1 vs Mamba-2 (SSD)

### 6.1 Mamba-1 Architecture

$$
\begin{aligned}
x &= u W^{(x)\top} \in \mathbb{R}^{L \times e d}, \\
z &= u W^{(z)\top} \in \mathbb{R}^{L \times e d}, \\
x_c &= \text{conv1d}(x) \in \mathbb{R}^{L \times e d}, \\
\Delta, B, C &= \text{low-rank projection}(x_c), \\
y &= \text{SSM}_{A,B,C,\Delta}(x_c) \in \mathbb{R}^{L \times e d}, \\
y_g &= y \odot \phi(z), \\
\text{out} &= y_g W^{(o)\top} \in \mathbb{R}^{L \times d},
\end{aligned}
$$

where $\phi$ is a SiLU gating function.

**Notes**

- Sequential SSM across $L$.  
- Low GPU utilization (≈ 18 %).  
- Simpler but not block-parallelizable.  

---

### 6.2 Mamba-2 (SSD) Architecture

$$
\begin{aligned}
x &= u W^{(x)\top} \in \mathbb{R}^{L \times e d}, \\
z &= u W^{(z)\top} \in \mathbb{R}^{L \times e d}, \\
\Delta, B, C &= \text{projection}(u), \\
x_c &= \text{conv1d}(x) \in \mathbb{R}^{L \times e d}, \\
y &= \text{SSM}_{A,B,C,\Delta}(x_c) \in \mathbb{R}^{L \times e d}, \\
y_g &= y \odot \phi(z), \\
y_h &= \text{groupnorm}(y_g), \\
\text{out} &= y_h W^{(o)\top} \in \mathbb{R}^{L \times d}.
\end{aligned}
$$

**Differences**

- Grouped $B,C,\Delta$ projections → multi-GPU parallelism.  
- Added **group normalization** for stability.  
- Fully parallel SSD computation (via semiseparable structure).  
- Achieves 70–80 % GPU utilization.

---

### 6.3 ASCII Architecture Schematic

```
Mamba-1: Sequential Selective SSM
┌────────────┐
│  Input u   │
└─────┬──────┘
      ↓
 Linear Projections (W^{(x)}, W^{(z)})
      ↓
 Depthwise Conv1d → Low-rank Δ, B, C
      ↓
 Sequential SSM Update (slow, 1D scan)
      ↓
 Gate with φ(z) (SiLU)
      ↓
 Output W^{(o)}

─────────────────────────────────────

Mamba-2: Structured State Space Duality (SSD)
┌────────────┐
│  Input u   │
└─────┬──────┘
      ↓
 Linear Projections (W^{(x)}, W^{(z)})
      ↓
 Conv1d (depthwise)
      ↓
 Parallel Projections → Δ, B, C (multi-group)
      ↓
  Blocked 1-SS / SSD Computation (O(T))
      ↓
 Gate + GroupNorm
      ↓
 Output W^{(o)}
```

---

### 6.4 Architectural Comparison Table

| Aspect | **Mamba-1** | **Mamba-2 (SSD)** |
|:--|:--:|:--:|
| Evaluation | Sequential SSM scan | Parallel 1-SS block processing |
| GPU Utilization | 18 % | 70–80 % |
| Projections | Per-token | Grouped & shared |
| Normalization | None | GroupNorm |
| Complexity | $\mathcal{O}(T N)$ (sequential) | $\mathcal{O}(T N)$ (parallel SSD) |
| Memory | Low | Slightly higher (group states) |
| Parallelism | Poor (per-step) | Excellent (per-block) |
| Accuracy (avg) | 59.9 % | **60.2 %** |

---

### 6.5 Why Mamba-2 Enables SSD Efficiency

1. **Parallel State Updates:** Uses semiseparable recurrence decomposition → transforms scan into block matmul.  
2. **Hardware Utilization:** Tensor-parallel $\Delta,B,C$ projections exploit matrix cores.  
3. **Normalization:** GroupNorm stabilizes cross-GPU computation.  
4. **Runtime:** Keeps SSM linear scaling while matching attention’s parallel throughput.

---

## 7. Discussion — Runtime & Duality Questions

<details>
<summary><strong>Q1 — Why is $\mathcal{O}(T N)$ generally more favorable than $\mathcal{O}(T^2)$?</strong></summary>

- Attention’s $\mathcal{O}(T^2)$ scaling becomes prohibitive for long sequences: doubling $T$ quadruples compute.  
- SSMs/SSDs scale linearly ($\mathcal{O}(T N)$), allowing 100K+ contexts with consistent latency.  
- SSD maintains high GPU utilization by using structured parallelism (matrix cores over 1-SS blocks).
</details>

<details>
<summary><strong>Q2 — If SSD is more efficient, why do we still use Attention and classic SSMs?</strong></summary>

- **Attention:** excels in reasoning, irregular dependencies, and short contexts.  
- **SSMs:** dominate streaming, low-latency, edge tasks.  
- **SSD:** unifies both paradigms, maintaining accuracy while achieving linear runtime.  
Future models will likely blend them — attention for reasoning layers, SSD for memory layers.
</details>

---

## 8. References

1. Vaswani et al., “Attention Is All You Need,” *NeurIPS 2017.*  
2. Gu et al., “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” *arXiv:2312.00752 (2023).*  
3. Dao & Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality,” *ICML 2024.*  
4. Vandebril et al., “Semiseparable Matrices and Structured Linear Algebra,” *SIAM Review (2007).*

---

**Thank you!**  
For questions or collaborations: [ryan.li@vanderbilt.edu](mailto:ryan.li@vanderbilt.edu)
