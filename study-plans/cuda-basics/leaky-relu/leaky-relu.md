# <span style="font-size: 20px;">Leaky ReLU</span>

<span style="font-size: 14px;">Leaky ReLU passes positives through unchanged and scales negatives by a small slope $\alpha$, $\text{output}[i] = \text{input}[i] > 0\ ?\ \text{input}[i] : \alpha \cdot \text{input}[i]$. It is an **embarrassingly parallel map**: every output depends on one input at the same index, with zero communication between threads. The systems interest is twofold - how the scalar $\alpha$ is broadcast to every thread without any memory traffic, and how to express the two-case select without serializing a warp.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$ and a slope $\alpha$, the kernel evaluates:</span>

$$
\text{output}[i] = \begin{cases} \text{input}[i] & \text{input}[i] > 0 \\ \alpha \cdot \text{input}[i] & \text{input}[i] \le 0 \end{cases}
$$

<span style="font-size: 14px;">Input and output are contiguous, row-major buffers of $N$ 32-bit floats in device (global) memory; $\alpha$ is a single `float` scalar (commonly $0.01$). Output element $i$ reads only input element $i$ and writes only output element $i$. Nothing is shared, reused, or reordered.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $N$ outputs are mutually independent, the natural decomposition is **one thread per element**. A one-dimensional grid of one-dimensional blocks covers the array, and each thread reconstructs its global position:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 threads is conventional: it is a multiple of the 32-lane **warp** so no lanes are wasted, it gives the scheduler many warps per block for latency hiding, and many such blocks fit on one **SM (Streaming Multiprocessor)**, keeping **occupancy** high. The grid needs $\lceil N / 256 \rceil$ blocks.</span>

<span style="font-size: 14px;">The body is guarded by `if (idx < N)` because rounding the grid up leaves surplus tail threads; without the check they read and write past the buffers. There is no `__syncthreads()` and no shared state - the whole computation is one flat wave of independent work.</span>

---

## <span style="font-size: 16px;">The Scalar $\alpha$: Broadcast By Value</span>

<span style="font-size: 14px;">A defining detail of this kernel is how $\alpha$ reaches the threads. It is passed **by value** as a kernel argument, not through a pointer to device memory. Kernel arguments are placed in a small per-launch constant region and delivered into each thread's registers when the block starts, so every one of the millions of threads sees the same $\alpha$ in a register with **zero global-memory traffic**. There is no load to coalesce, no cache line to fetch, and no contention - the value is simply present.</span>

<span style="font-size: 14px;">This is the right pattern for any read-only scalar broadcast to all threads. Routing $\alpha$ through a one-element global buffer instead would force every thread (or at least every warp) to issue a load, adding pointless latency and a dependency the compiler cannot fold into the arithmetic. By value, $\alpha$ becomes a register operand that fuses directly into the multiply.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel touches two global arrays per element: it loads `input[idx]` and stores `output[idx]`. There is no reuse, so nothing belongs in `__shared__` memory; the selected value lives in a register for its brief lifetime. The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `input` and write 32 consecutive addresses of `output`, served in the minimum number of transactions. With $\alpha$ in a register, the entire memory footprint per element is just the 8 bytes of load and store.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 8 bytes and performs a compare, a select, and a multiply - two or three FLOPs. The **arithmetic intensity** is about:</span>

$$
\frac{\sim 3 \text{ FLOP}}{8 \text{ bytes}} \approx 0.375 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** the ridge point sits in the tens of FLOPs per byte, so at $\approx 0.375$ Leaky ReLU is **deeply memory-bound**. The compare-select-multiply is nearly free; the only levers that change runtime are coalesced access (already optimal) and enough warps to hide DRAM latency. The runtime is just $8N$ bytes divided by achievable bandwidth.</span>

---

## <span style="font-size: 16px;">Predication vs Warp Divergence</span>

<span style="font-size: 14px;">The two-case select looks like a branch, and a literal `if (x > 0) ... else ...` would risk **warp divergence**: when some lanes in a warp see positive inputs and others negative, the SIMT hardware would execute both arms with the inactive lanes masked. But this kernel's branch is trivial - both arms produce a single value of the same type - so the compiler converts the ternary into **predication**: it computes both candidate results (`x` and `alpha * x`) and selects between them per lane with a predicated move, no actual control-flow split.</span>

<span style="font-size: 14px;">Predication means the warp never diverges: every lane executes the same straight-line instruction stream and just keeps the result its predicate selects. The cost is computing one extra multiply on lanes that did not need it, which on a memory-bound kernel is invisible. Writing it branchlessly - for instance `x > 0 ? x : alpha * x`, or equivalently a `fmaxf(x, alpha * x)` when $0 < \alpha < 1$ - makes the intent explicit and keeps divergence at zero. The contrast with a heavy, divergent branch is the lesson: tiny ternaries predicate cleanly, so divergence here is minimal regardless.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel is the divergent `if/else`; the first refinement is the predicated ternary or the `fmaxf` form, which guarantees no control-flow split. Two further refinements approach the bandwidth ceiling:</span>

<span style="font-size: 14px;">1. **Grid-stride loop**: launch a fixed, device-sized grid and let each thread process multiple elements by striding in steps of `blockDim.x * gridDim.x`, with $\alpha$ still resident in a register the whole time. One configuration handles any $N$.</span>

<span style="font-size: 14px;">2. **Vectorized loads**: reinterpreting the arrays as `float4` lets each thread load and store 16 bytes per instruction and apply the select across the four components, issuing fewer, wider transactions and nudging the kernel toward the memory ceiling.</span>

<span style="font-size: 14px;">All sit on top of an already-saturated memory pipeline; for a map you can only approach the bandwidth roofline.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$, $\alpha = 0.01$, block size 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$ and write `output[0..3]`.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ write `output[4..5]`; indices $6$ and $7$ fail `idx < 6` and exit.</span>

<span style="font-size: 14px;">With `input` $= [-3, 2, -1, 5, 0, -4]$ and $\alpha$ in every lane's register, the active threads produce `output` $= [-0.03, 2, -0.01, 5, 0, -0.04]$. Under predication each lane computes both `x` and `0.01 * x` and keeps the one its sign selects, so the warp runs a single straight-line stream even though the signs alternate - the alternation that would have split a true `if`.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Passing $\alpha$ through global memory.** A scalar passed by value lands in a register for free; routing it through a device buffer forces a needless load per thread and a dependency the compiler cannot fuse.</span>
* <span style="font-size: 14px;">**Forcing a heavy branch.** Keep the select a ternary or `fmaxf` so the compiler predicates it; a bulky `if/else` body can defeat predication and cause real **warp divergence**.</span>
* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up; without `if (idx < N)` the tail threads read and write out of bounds.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth.</span>

---