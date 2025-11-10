# Transformers are SSMs - 9-Minute Presentation
**Tri Dao (Princeton) & Albert Gu (CMU) | [arXiv:2405.21060](https://arxiv.org/pdf/2405.21060)**

---

## Overview (90 sec)

### Problem
- **Transformers:** O(T²) training, O(T) inference memory
- **SSMs (Mamba):** O(T) training, O(1) inference, but GPU-unfriendly
- Developed separately, no connection understood

### Solution: State Space Duality
**SSMs = Semiseparable Matrices = Structured Masked Attention**

### Key Result
**Mamba-2:** 2-8× faster than Mamba, supports 8× larger states

### 4 Contributions
1. Theory: Proves SSMs ≡ semiseparable matrices
2. Algorithm: Block decomposition (O(TN²) + hardware-efficient)
3. Architecture: Mamba-2 with tensor parallelism
4. Performance: Pareto dominates Mamba & Transformers

---

## Architecture (2.5 min)

### Three Equivalent Forms

**SSM (Linear)**
```
h_t = A_t h_{t-1} + B_t x_t
y_t = C_t^T h_t
```
*Intuition:* Like a memory bank - take previous memory (h_{t-1}), decay it by A_t, add new input weighted by B_t, then read it out with C_t. Similar to how you remember yesterday's conversation but gradually forget.

**Semiseparable Matrix (Quadratic)**
```
M[j,i] = C_j^T · (A_j...A_{i+1}) · B_i
y = M · x
```
*Intuition:* Build a table showing "how much does input i affect output j?" The A terms control decay over time - distant past has less influence (like compound interest in reverse).

**Structured Attention (Quadratic)**
```
Y = (L ∘ QK^T) · V
```
*Intuition:* Just like attention, but the mask L isn't fixed - it's learned! Q≡C, K≡B, V≡X. Instead of "always attend to everything before," we learn "how much to attend based on distance."

### SSD Algorithm: Block Decomposition

**Intuition:** Think of your sequence like a book divided into chapters (chunks). 

**Partition into chunks (Q=64):**

1. **Diagonal blocks** (intra-chunk)
   - Use attention form: M = (L ∘ CB^T) · X
   - *What it does:* "How do words in this chapter relate to each other?"
   - Parallel matrix multiplications (GPU-friendly)
   
2. **Off-diagonal blocks** (inter-chunk)
   - Low-rank: C_factors · A_factors · B_factors
   - *What it does:* "How does chapter 3 influence chapter 7?"
   - 3 matmuls + small scan(T/Q)
   - Sequential but only Q times shorter!

**Result:** O(TN²) FLOPs using matrix multiplications

**Why it works:** Within a chapter, use attention (fast, parallel). Between chapters, use compressed summaries (the hidden state h). Best of both worlds!

### Comparison

| Model | State | Train | Infer | Memory | MatMul |
|-------|-------|-------|-------|--------|--------|
| Attention | T | T²N | TN | T² | ✓ |
| Mamba-1 | N | TN² | N² | TN² | ✗ |
| **Mamba-2** | **N** | **TN²** | **N²** | **TN** | **✓** |

---

## Mamba-2 Design (1.5 min)

### 3 Key Improvements

1. **Parallel Projections**
   - A, X, B, C created together (like Q, K, V)
   - Enables tensor parallelism

2. **Extra Normalization**
   - RMSNorm before output
   - Stability at scale

3. **Multi-Input SSM (MIS)**
   - X: (T, H, P) - independent per head
   - B, C: (T, 1, N) - shared
   - Best empirical performance

### Dimensions
- N, P = 64-128 (state & head)
- H = D/P (# heads)
- Q = 64 (chunk size)

---

## Critical Analysis (2 min)

### Strengths ✓
- **Theory:** First SSM = semiseparable proof; characterizes efficient attention
- **Algorithm:** Non-obvious, optimal complexity + hardware efficiency
- **Systems:** Tensor/sequence parallelism, variable-length training
- **Unification:** RetNet, GateLoop, Linear Attention → all special cases

### Limitations ⚠️
- **Scale:** Only to 2.7B params, 300B tokens (frontier: 100B+, trillions)
- **Theory:** Why certain masks work? How to set N?
- **Tradeoffs:** O(N²) state may limit ultra-long context
- **Eval:** Mainly Pile dataset, limited downstream

### Disputed 🤔
- **Title:** Overstates—only kernel attention proven, not softmax attention
- **Speedup:** Needs custom implementation
- **Baselines:** Pythia/Transformer++, not latest SOTA

---

## Impact (1 min)

### Applications
- Long context: docs, code repos, conversations
- Edge: mobile, IoT, real-time
- Efficiency: lower training costs

### Research Opened
- Other structured masks: Toeplitz, Cauchy, Fourier
- Theory: expressiveness, optimization landscapes
- Hybrids: attention for retrieval, SSMs for compression

### Paradigm Shift
**From:** "Attention is all you need"
**To:** Unified framework (attention & SSMs are spectrum extremes)

---

## Summary

**Core Insight:** SSMs ≡ Semiseparable Matrices ≡ Attention variants

**Innovation:** Block decomposition combines linear complexity + hardware efficiency

**Result:** Mamba-2 achieves 2-8× speedup with same O(TN²) cost

**Impact:** Unifies two major paradigms, enables systematic architecture design

---

## Resources
- [Paper](https://arxiv.org/pdf/2405.21060) | [Code](https://github.com/state-spaces/mamba)
- Checkpoints: 370M, 1.3B, 2.7B params

---

**Questions?**
