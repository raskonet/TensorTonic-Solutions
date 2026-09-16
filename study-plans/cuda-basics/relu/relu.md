# <span style="font-size: 20px;">ReLU</span>

<span style="font-size: 14px;">ReLU clamps every negative value to zero and passes positives through unchanged, $\text{output}[i] = \max(\text{input}[i], 0)$. It is an **embarrassingly parallel map**: every output depends on exactly one input at the same index, with zero communication between threads. The interesting systems lesson is not the arithmetic - there is none worth counting - but how to express the clamp without making a warp branch against itself.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$, the kernel evaluates:</span>

$$
\text{output}[i] = \max(\text{input}[i], 0)
$$

<span style="font-size: 14px;">Input and output are contiguous, row-major buffers of $N$ 32-bit floats in device (global) memory. Output element $i$ reads only input element $i$ and writes only output element $i$. Nothing is shared, reused, or reordered - the index is the only structure.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $N$ outputs are mutually independent, the natural decomposition is **one thread per element**. A one-dimensional grid of one-dimensional blocks covers the array, and each thread reconstructs its global position from its block and lane coordinates:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 threads is conventional: it is a multiple of the 32-lane **warp** so no lanes are wasted, it gives the scheduler many warps per block for latency hiding, and many such blocks fit on one **SM (Streaming Multiprocessor)**, keeping **occupancy** high. The grid needs $\lceil N / 256 \rceil$ blocks to cover the array.</span>

<span style="font-size: 14px;">Rounding the grid up means the last block holds more threads than there are remaining elements, so the body is guarded by `if (idx < N)`. Without that check the tail threads read and write past the end of the buffers. There is no `__syncthreads()` and no shared state: the whole computation is one flat wave of independent work.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel touches two global arrays per element: it loads `input[idx]` and stores `output[idx]`. There is no reuse, so nothing belongs in `__shared__` memory - shared memory exists to amortize repeated loads, and a map loads each datum exactly once. The single clamped value lives in a register for its brief lifetime and is gone.</span>

<span style="font-size: 14px;">The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `input` and write 32 consecutive addresses of `output`. The memory controller serves each warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth. This unit-stride layout is the best case the hardware offers.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 8 bytes of global memory - one 4-byte load and one 4-byte store - and performs a single comparison-select. Counting that as one FLOP, the **arithmetic intensity** is about:</span>

$$
\frac{1 \text{ FLOP}}{8 \text{ bytes}} \approx 0.125 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** model a kernel is compute-bound only when its intensity exceeds the ridge point, which sits in the tens of FLOPs per byte on modern hardware. At $0.125$ ReLU is two to three orders of magnitude below that line: it is **deeply memory-bound**. The select unit is idle almost the entire time, waiting for operands from DRAM.</span>

<span style="font-size: 14px;">This classification fixes the optimization story. Cleverer arithmetic is pointless - there is only one comparison. The only levers that change runtime are those that raise effective bandwidth: coalesced access (already optimal) and enough warps in flight to hide DRAM latency. The runtime is just $8N$ bytes divided by achievable bandwidth.</span>

---

## <span style="font-size: 16px;">Branchless Select vs Warp Divergence</span>

<span style="font-size: 14px;">The clamp can be written two ways, and the choice is the whole point of this kernel. The obvious form is a branch: `if (input[idx] > 0) output[idx] = input[idx]; else output[idx] = 0;`. Within a warp, the 32 lanes execute in lockstep under the SIMT model, so when some lanes take the true path and others the false path the warp **diverges**: the hardware runs both paths with the inactive lanes masked off, then reconverges. For random inputs roughly half the lanes are masked in each path, halving useful throughput on the branchy region.</span>

<span style="font-size: 14px;">The branchless form `fmaxf(input[idx], 0.0f)` compiles to a single max instruction that every lane executes unconditionally with no masking. There are no divergent paths to serialize, so the warp stays fully active. Even though ReLU is memory-bound and the arithmetic is nearly free, the branchless intrinsic is the correct idiom: it removes a divergence hazard at zero cost and keeps the kernel a clean map. `fmaxf` also handles the sign of zero and NaN propagation in a well-defined way that an ad-hoc compare may not.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles. The GPU hides that not with large caches but with **massive multithreading**: when a warp issues its load of `input` and stalls, the SM scheduler switches to another resident warp that is ready. With high occupancy there is always other work to issue and the memory pipeline stays saturated. Because the branchless kernel has no divergence, no synchronization, and no shared-memory pressure, occupancy is limited only by having launched enough blocks - automatic for large $N$.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel is the divergent `if` form; the first optimization is simply replacing it with `fmaxf` to kill divergence. Beyond that, two refinements approach the bandwidth ceiling:</span>

<span style="font-size: 14px;">1. **Grid-stride loop**: launch a fixed, device-sized grid and let each thread process multiple elements by striding in steps of `blockDim.x * gridDim.x`. One configuration handles any $N$ and launch overhead is amortized.</span>

<span style="font-size: 14px;">2. **Vectorized loads**: reinterpreting the arrays as `float4` lets each thread load and store 16 bytes per instruction instead of 4. Applying `fmaxf` componentwise across the four lanes issues fewer, wider transactions and nudges the kernel toward the memory ceiling.</span>

<span style="font-size: 14px;">All of these sit on top of an already-saturated memory pipeline. For a map you cannot beat the bandwidth roofline, you can only approach it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ with a block size of 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$. All pass `idx < 6` and write `output[0..3]`.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ pass and write `output[4..5]`; indices $6$ and $7$ fail `idx < 6` and exit without touching memory.</span>

<span style="font-size: 14px;">With `input` $= [-3, 2, -1, 5, 0, -4]$, the six active threads independently produce `output` $= [0, 2, 0, 5, 0, 0]$. Under the branchless `fmaxf` every lane runs the same max instruction regardless of sign, so the warp never splits even though the signs alternate. A divergent `if` on this same data would have masked roughly half the lanes per path.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Warp divergence from a branch.** Writing the clamp as `if (x > 0)` makes mixed-sign warps execute both paths masked; `fmaxf(x, 0.0f)` is branchless and keeps every lane active.</span>
* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up; without `if (idx < N)` the tail threads read and write out of bounds.</span>
* <span style="font-size: 14px;">**Expecting arithmetic optimizations to help.** At $\approx 0.125$ FLOP/byte the kernel is memory-bound, so only coalescing, occupancy, and wider loads change runtime.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth.</span>

---