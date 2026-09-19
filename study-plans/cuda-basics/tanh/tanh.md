# <span style="font-size: 20px;">Tanh</span>

<span style="font-size: 14px;">Tanh maps each value into $(-1, 1)$ via $\text{output}[i] = \tanh(\text{input}[i])$. It is an **embarrassingly parallel map**: every output depends on exactly one input at the same index, with zero communication between threads. The systems lesson here is the choice of how to evaluate the transcendental - the hardware intrinsic `tanhf` versus building it by hand from `expf` ratios - and why one is both faster and more numerically stable than the other.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$, the kernel evaluates:</span>

$$
\text{output}[i] = \tanh(\text{input}[i]) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

<span style="font-size: 14px;">Input and output are contiguous, row-major buffers of $N$ 32-bit floats in device (global) memory. Output element $i$ reads only input element $i$ and writes only output element $i$. Nothing is shared, reused, or reordered - the index is the only structure.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $N$ outputs are mutually independent, the natural decomposition is **one thread per element**. A one-dimensional grid of one-dimensional blocks covers the array, and each thread reconstructs its global position:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 threads is conventional: it is a multiple of the 32-lane **warp** so no lanes are wasted, it gives the scheduler many warps per block for latency hiding, and many such blocks fit on one **SM (Streaming Multiprocessor)**, keeping **occupancy** high. The grid needs $\lceil N / 256 \rceil$ blocks.</span>

<span style="font-size: 14px;">The body is guarded by `if (idx < N)` because rounding the grid up leaves surplus tail threads; without the check they read and write past the buffers. There is no `__syncthreads()` and no shared state - the whole computation is one flat wave of independent work.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel touches two global arrays per element: it loads `input[idx]` and stores `output[idx]`. There is no reuse, so nothing belongs in `__shared__` memory - the transcendental result lives in a register for its brief lifetime and is gone. A map loads each datum exactly once.</span>

<span style="font-size: 14px;">The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `input` and write 32 consecutive addresses of `output`. The memory controller serves each warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth. This unit-stride layout is the best case the hardware offers.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 8 bytes - one 4-byte load and one 4-byte store - and performs the transcendental, which the GPU evaluates on the **Special Function Unit (SFU)** as a short sequence of instructions. Counting roughly a dozen effective FLOPs, the **arithmetic intensity** is about:</span>

$$
\frac{\sim 12 \text{ FLOP}}{8 \text{ bytes}} \approx 1.5 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">The transcendental nudges the kernel toward compute relative to a pure copy, but the ridge point of the **roofline** sits in the tens of FLOPs per byte, so at $\approx 1.5$ tanh is still well below it: **memory-bound at scale**. The SFU work overlaps with outstanding memory transactions, and once enough warps are in flight the DRAM bandwidth - not the SFU throughput - sets the runtime.</span>

---

## <span style="font-size: 16px;">Hardware Intrinsic vs Building From `expf`</span>

<span style="font-size: 14px;">The naive way to compute tanh is to evaluate the ratio of exponentials literally: call `expf(x)` and `expf(-x)`, subtract and add, then divide. That is two transcendental calls and a division per element. It is also numerically fragile: for large positive $x$, $e^{x}$ overflows to infinity while $e^{-x}$ underflows to zero, and the ratio degenerates even though the true answer is simply $1$. The same happens with reversed signs for large negative $x$.</span>

<span style="font-size: 14px;">The intrinsic `tanhf(x)` sidesteps both problems. It maps to a tuned SFU sequence - typically a single `expf`-class evaluation plus a few cheap operations rather than two full exponentials and a divide - so it is faster, and it is range-reduced internally so it saturates cleanly to $\pm 1$ without overflowing. The lesson generalizes across the activation track: when CUDA provides a dedicated math intrinsic, it is almost always both faster and more robust than reconstructing the function from `expf` primitives. The fast-math variant `__tanhf` trades a few low-order bits for still-higher SFU throughput, which at low occupancy can matter and at high occupancy is hidden behind memory.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles; the SFU `tanhf` costs a few. The GPU hides the dominant memory latency with **massive multithreading**: when a warp stalls on its load of `input`, the SM scheduler switches to another resident warp. High occupancy keeps the memory pipeline saturated, and the SFU work of one warp fills the gaps while other warps wait on DRAM, so the two overlap rather than serialize. Tanh has no data-dependent branch, so every active lane takes the same path and there is no **warp divergence**.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel computes the explicit `expf` ratio; the first and largest optimization is replacing it with `tanhf`, which cuts the transcendental work roughly in half and removes the overflow hazard. Two further refinements approach the bandwidth ceiling:</span>

<span style="font-size: 14px;">1. **Grid-stride loop**: launch a fixed, device-sized grid and let each thread process multiple elements by striding in steps of `blockDim.x * gridDim.x`. One configuration handles any $N$ and amortizes launch overhead.</span>

<span style="font-size: 14px;">2. **Vectorized loads**: reinterpreting the arrays as `float4` lets each thread load and store 16 bytes per instruction and apply `tanhf` across the four components, issuing fewer, wider transactions and nudging the kernel toward the memory ceiling.</span>

<span style="font-size: 14px;">All of these sit on top of an already memory-limited pipeline. For a transcendental map the bandwidth roofline is still the wall; the intrinsic only helps until memory becomes the constraint again.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ with a block size of 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$ and write `output[0..3]`.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ write `output[4..5]`; indices $6$ and $7$ fail `idx < 6` and exit.</span>

<span style="font-size: 14px;">With `input` $= [0, 1, -1, 20, \ldots]$, the intrinsic gives `output` $\approx [0, 0.762, -0.762, 1.0, \ldots]$. The fourth lane is the instructive one: `tanhf(20)` saturates cleanly to $1.0$, whereas the hand-rolled ratio would compute $e^{20}/(e^{20}+e^{-20})$ where $e^{20}$ is enormous and $e^{-20}$ negligible - a needless near-overflow that the intrinsic avoids. Every lane runs the same instruction sequence, so the warp never diverges.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Hand-rolling tanh from `expf`.** The explicit ratio doubles the transcendental work and overflows for large-magnitude inputs; `tanhf` is faster and range-reduced to saturate at $\pm 1$.</span>
* <span style="font-size: 14px;">**Treating it as compute-bound.** At $\approx 1.5$ FLOP/byte the kernel is memory-bound at scale; coalescing and occupancy, not faster math, set the runtime.</span>
* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up; without `if (idx < N)` the tail threads read and write out of bounds.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth.</span>

---