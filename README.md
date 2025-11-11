# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Presenter:** Ryan Li  
**Institution:** Vanderbilt University – Data Science Institute  

**Paper:** Dao & Gu (2024), ICML — [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)

---

## Overview (13–15 minutes)

1. **The Problem** - Two competing architectures  
2. **State Space Models** - How they work
3. **Attention Mechanisms** - How they work  
4. **The Bridge** - Structured matrices unify them
5. **Algorithms A-E** - From theory to practice  
6. **Mamba-2 Results** - 2-8X speedup
7. **Key Insights** - Why this matters
8. **Questions**

---

## 1. The Problem: Two Paradigms

### Transformers (Attention-Based)

Compare every word with every other word in the sequence.

- Cost: O(T²) — doubling sequence length quadruples computation
- Strong parallel processing on GPUs
- But: expensive memory for long sequences

### State Space Models (Recurrent)

Maintain a "memory buffer" that updates one word at a time.

- Cost: O(T) — scales linearly with sequence length  
- Memory efficient
- But: sequential processing, poor GPU utilization

### The Question

*Are these fundamentally different models, or two ways to compute the same thing?*

**Answer (Dao & Gu, 2024):** They're dual algorithms computing the same mathematical structure.

**Result:** Mamba-2 achieves 2-8X speedup — combining SSM efficiency with attention-style parallelism.

---

## 2. State Space Models Explained

### How SSMs Work

An SSM maintains a hidden state vector that evolves through the sequence:

**State Update (at each word):**
- New state = (old state × transition matrix) + (current input × input matrix)

**Output:**
- Output = state × output matrix

### Unrolled View

Each output word is influenced by all previous inputs through the chain of state transitions:

Output[t] = sum over all previous words s of (influence of word s through state transitions from s to t)

### Why This Works Well

- **Memory:** Only store the hidden state, not full history
- **Speed:** Linear scaling with sequence length
- **Streaming:** Works naturally with streaming input

### Why It Struggles

- **Sequential:** Must process words one at a time — can't parallelize
- **GPU Underutilization:** GPUs are designed for parallel matrix operations, not sequential loops

---

## 3. Attention Mechanisms Explained

### How Attention Works

For each word, compute how relevant every other word is, then create a weighted average.

**Three Steps:**

1. **Project input:** Convert words to queries, keys, and values
2. **Compute attention:** For each query, compute similarity scores against all keys
3. **Aggregate:** Weighted average of values using attention scores

### Mathematical View

Attention creates a T×T matrix where entry (i,j) represents "how much does word j influence word i?"

### Why This Works Well

- **Expressiveness:** Each word can attend to any other word
- **Parallelism:** Compute all attention scores simultaneously
- **Flexibility:** Learn which positions are important

### Why It Struggles

- **Quadratic Cost:** O(T²) — becomes prohibitive for long sequences
- **Memory:** Must store entire T×T attention matrix

---

## 4. The Mathematical Bridge: Structured Matrices

### Unified View

Both models compute: **Output = M × Input**

Where M is a T×T matrix describing "how each input affects each output."

**For Attention:** M is dense (fully filled)

**For SSMs:** M is lower-triangular (causal — future can't affect past) with special structure

### The Key Insight: Semiseparable Matrices

A **semiseparable matrix** has a special structure:

- Off-diagonal elements factor as products of two vectors
- Can be stored in O(T) space instead of O(T²)
- Can be multiplied with a vector in O(T) time instead of O(T²)

### Why Structure Matters

Structured matrices enable:
1. **Efficiency:** Linear time for matrix operations
2. **Parallelism:** Operations decompose into parallel blocks
3. **Unification:** Both SSMs and attention can be expressed this way with appropriate structure

---

## 5. Formal Algorithms

### Algorithm A: Attention Layer

```
Input: Sequence X, weight matrices W_Q, W_K, W_V
1. Q = X · W_Q  (query projection)
2. K = X · W_K  (key projection)
3. V = X · W_V  (value projection)
4. S = (Q · K^T) / sqrt(d_k)  (attention scores)
5. A = softmax(S)  (normalize to probabilities)
6. Output = A · V  (aggregate values)
```

**Cost:** O(T²·d) — quadratic scaling

---

### Algorithm B: SSM (Recurrent)

```
Input: Sequence x_1...x_T, parameters A_t, B_t, C_t
h = 0  (initialize hidden state)
for t = 1 to T:
    h = A_t · h + B_t · x_t  (update state)
    y_t = C_t^T · h  (compute output)
Output: y_1...y_T
```

**Cost:** O(T·N) where N is state size — linear scaling, but sequential

---

### Algorithm C: Mamba-1 (Selective SSM)

Extension of Algorithm B where state parameters depend on input:

```
Input: Sequence x_1...x_T
h = 0
for t = 1 to T:
    A_t = f_A(x_t)  (learned from input)
    B_t = f_B(x_t)  (learned from input)
    C_t = f_C(x_t)  (learned from input)
    h = A_t · h + B_t · x_t
    y_t = C_t^T · h
Output: y_1...y_T
```

**Key Innovation:** Input-dependent "gates" let the model choose what to remember

**Limitation:** Still sequential — only 18% GPU utilization

---

### Algorithm D: Mamba-2 (Structured State Space Duality)

The breakthrough: Use structured matrix decomposition to parallelize.

```
Input: Sequence X, block size Q

// Part 1: Parallel within blocks (70-80% GPU utilization)
for each block j in parallel:
    Build structured matrix M_j from C_j, B_j, decay parameters
    y_j = M_j · x_j  (apply structured kernel)

// Part 2: Sequential across blocks (only T/Q steps, not T)
for each block j sequentially:
    Propagate state from block j-1 to block j
    Update outputs y_j with carry-from previous block

Output: Y
```

**Key Benefits:**
- Intra-block parallelism via matrix operations (fast on GPUs)
- Reduced inter-block recurrence (only T/Q steps instead of T)
- Linear scaling: O(T) time with high GPU utilization

---

### Algorithm E: Efficient Structured Matrix-Vector Product

Core operation used in Mamba-2:

```
Input: Vector x, factors u, v, diagonal d

acc = 0  (accumulator)
for i = 1 to T:
    y[i] = d[i] · x[i] + u[i] · acc
    acc = acc + v[i] · x[i]
Output: y
```

**Interpretation:**
- **d[i] · x[i]:** Direct connection from input to output
- **acc:** Running summary of past inputs
- **u[i]:** How much past history influences current output
- **v[i]:** How much current input contributes to future

**Cost:** O(T) with simple operations — very efficient

---

## 6. Mamba-1 vs Mamba-2: Architecture Comparison

### Mamba-1 (Sequential)

```
Input → Projections → Conv1d → Sequential SSM Update → Gate → Output
         (all inputs)          (one word at a time)
```

- Sequential processing limits parallelism
- GPU utilization: ~18%
- Simple but inefficient

### Mamba-2 (SSD with Parallelism)

```
Input → Projections → Conv1d → Parallel Structured SSM → Gate + Normalization → Output
         (grouped)            (block-wise parallelism)
```

- Grouped projections enable multi-device parallelism
- Block-wise processing for parallel computation
- Group normalization for stability across parallel execution
- GPU utilization: 70-80%

### Key Differences

| Aspect | Mamba-1 | Mamba-2 |
|--------|---------|---------|
| Processing | Sequential per token | Parallel per block |
| GPU Utilization | 18% | 70-80% |
| Projections | Per-token | Grouped/shared |
| Normalization | None | Group norm |
| Speedup | Baseline | 2-8X faster |
| Accuracy | 59.9% avg | 60.2% avg |

---

## 7. Real-World Performance

### Speed Comparison (A100 GPU)

| Sequence Length | FlashAttention-2 | Mamba-1 | Mamba-2 | Speedup |
|-----------------|-----------------|---------|----------|----------|
| 512 tokens | 0.20 ms | 0.15 ms | 0.15 ms | 1X |
| 1K tokens | 0.50 ms | 0.30 ms | 0.25 ms | 2X |
| 4K tokens | 5.00 ms | 1.20 ms | 0.80 ms | 6X |
| 16K tokens | 40 ms | 5.0 ms | 3.0 ms | 8X |
| 32K tokens | 120 ms | 10 ms | 6 ms | 10X |

**Key Insight:** Mamba-2 gets faster relative to attention as sequences get longer (linear vs quadratic scaling).

### Language Modeling Quality

| Model | Pile Loss | LAMBADA | HellaSwag | PIQA | ARC-E | ARC-C | Average |
|-------|-----------|---------|-----------|------|-------|-------|---------|
| Pythia (Transformer) | 6.73 | 64.7 | 59.3 | 74.0 | 64.1 | 32.9 | 55.7% |
| Mamba-1 | 6.22 | 69.2 | 66.1 | 75.2 | 69.7 | 36.3 | 59.9% |
| **Mamba-2 (SSD)** | **6.09** | **69.7** | **66.6** | **76.4** | **69.6** | **36.4** | **60.2%** |

**Result:** Better quality + faster execution

### Memory Task Performance

Test: Given N query-key-value triplets, retrieve correct value for each query

| Architecture | Memory Size | Accuracy |
|--------------|-------------|----------|
| Attention | — | 95% |
| Mamba-1 | 16 | 20% |
| Mamba-2 | 64 | 90% |
| **Mamba-2** | **256** | **98%** |

**Insight:** Structured matrix representation captures complex memory patterns better than expected.

---

## 8. Why This Unification Matters

### Three Key Insights

**1. Algorithm ≠ Representation**

The same mathematical transformation can be computed via:
- Sequential recurrence (SSM-style): good for streaming, uses little memory
- Parallel matrix multiply (Attention-style): good for parallelism, uses more memory
- Structured blocks (Mamba-2 SSD): good for both

This is similar to computing a sum: you can add left-to-right sequentially, or split into groups and parallelize.

**2. Structure is Power**

When the transformation matrix has special structure (low-rank patterns, decay, sparsity), you can:
- Store it efficiently (O(T) instead of O(T²))
- Compute with it efficiently (O(T) instead of O(T²))
- Parallelize computation while maintaining efficiency

Real sequences have this structure naturally — recent context matters more than distant.

**3. Future Hybrid Models**

Rather than choosing SSM or attention:
- Use attention for reasoning tasks (irregular dependencies)
- Use SSD for memory/efficiency (structured dependencies)
- Blend both in same model

---

## 9. Open Questions

### Q1: When Does SSD Excel?

SSD works best when:
- Sequence length is long (T >> N)
- Information decays over time (nearby context matters most)
- Dependencies are structured (not highly irregular)

When might it struggle?
- Algorithmic tasks requiring non-local lookups
- Irregular dependency patterns
- Very short sequences (overhead not worth it)

### Q2: Hardware Co-Design

Modern GPUs optimized for dense attention. What hardware would optimally support SSD?

Opportunities:
- Specialized semiseparable matrix units
- Efficient prefix-sum operations
- Reduced precision for structured operations

---

## 10. Summary

### Main Results

**Theoretical:** SSMs and attention are dual representations of the same transformation, connected through structured semiseparable matrices.

**Practical:** Mamba-2 achieves 2-8X speedup while maintaining language modeling quality, through clever block-level parallelism.

**Architectural:** Different models emerge from choosing different algorithms for structured matrices, not from fundamentally different representations.

### Takeaways

1. **Linear scaling is achievable** without sacrificing expressiveness, via structured matrices
2. **Parallelism and efficiency can coexist** through algorithmic duality
3. **Hybrid models are promising** — blend approaches based on task structure
4. **Hardware matters** — SSD designed for modern GPU tensor cores

### Impact

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