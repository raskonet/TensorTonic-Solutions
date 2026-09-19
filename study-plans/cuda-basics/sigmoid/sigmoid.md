# <span style="font-size: 20px;">Sigmoid</span>

<span style="font-size: 14px;">Sigmoid squashes each value into $(0, 1)$ via $\text{output}[i] = 1 / (1 + e^{-\text{input}[i]})$. It is an **embarrassingly parallel map**: every output depends on exactly one input at the same index, with zero communication between threads. The systems twist over ReLU is the transcendental `expf`, which adds real arithmetic per element and shifts intensity slightly toward compute - but, as the roofline shows, the kernel stays bandwidth-limited at scale.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$, the kernel evaluates:</span>

$$
\text{output}[i] = \frac{1}{1 + e^{-\text{input}[i]}}
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

<span style="font-size: 14px;">The kernel touches two global arrays per element: it loads `input[idx]` and stores `output[idx]`. There is no reuse, so nothing belongs in `__shared__` memory - the intermediate `expf` result lives in a register for its brief lifetime and is gone. A map loads each datum exactly once.</span>

<span style="font-size: 14px;">The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `input` and write 32 consecutive addresses of `output`. The memory controller serves each warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth. This unit-stride layout is the best case the hardware offers.</span>

<span style="font-size: 14px;">The pointers passed to the kernel already point at device memory - the host copied the input across the PCIe bus before launch. The kernel never dereferences host pointers or attempts its own transfers; it operates purely on the device-resident buffers it is given, and the `expf` evaluation happens entirely in registers and the SFU pipeline between the load and the store.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 8 bytes - one 4-byte load and one 4-byte store - and performs a handful of operations: a negate, the transcendental `expf`, an add, and a reciprocal. The `expf` is the costly one; the GPU evaluates it on the **Special Function Unit (SFU)** as a small sequence of instructions rather than a single cycle. Counting roughly a dozen effective FLOPs, the **arithmetic intensity** is about:</span>

$$
\frac{\sim 12 \text{ FLOP}}{8 \text{ bytes}} \approx 1.5 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That is higher than ReLU's $0.125$, so the transcendental genuinely shifts the kernel toward compute. But the ridge point of the **roofline** sits in the tens of FLOPs per byte, so at $\approx 1.5$ sigmoid is still well below it: **memory-bound at scale**. The SFU work overlaps with outstanding memory transactions, and once enough warps are in flight the DRAM bandwidth - not the SFU throughput - sets the runtime.</span>

<span style="font-size: 14px;">This classification matters because it tells you which knob to turn. The intuition that "sigmoid is expensive because of the exponential" is true per instruction but false for the kernel as a whole: the cost of $N$ exponentials is hidden behind the cost of moving $8N$ bytes. A faster `expf` shaves SFU cycles that were already overlapped, so it leaves wall-clock time roughly unchanged. Only at very small $N$, or with too few warps to saturate the bus, does the SFU surface as the bottleneck.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles; the SFU `expf` costs a few. The GPU hides the dominant memory latency with **massive multithreading**: when a warp stalls on its load of `input`, the SM scheduler switches to another resident warp. High occupancy keeps the memory pipeline saturated. Crucially the `expf` work of one warp is exactly the kind of ready instruction that fills the gaps while other warps wait on DRAM, so the SFU and the memory subsystem overlap rather than serialize. Every active lane takes the same code path - sigmoid has no data-dependent branch - so there is no **warp divergence** to worry about.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel computes the formula literally. Two refinements approach the bandwidth ceiling:</span>

<span style="font-size: 14px;">1. **Grid-stride loop**: launch a fixed, device-sized grid and let each thread process multiple elements by striding in steps of `blockDim.x * gridDim.x`. One configuration handles any $N$ and amortizes launch overhead; it also raises the ratio of resident SFU work to launch cost.</span>

<span style="font-size: 14px;">2. **Cheaper transcendental**: using the fast-math intrinsic `__expf` instead of the precise `expf` trades a few low-order bits for higher SFU throughput. Because the kernel is memory-bound this rarely changes wall-clock time at scale, but it reduces the chance the SFU becomes a local bottleneck at low occupancy. Vectorized `float4` loads further widen each memory transaction.</span>

<span style="font-size: 14px;">All of these sit on top of an already memory-limited pipeline. For a transcendental map the bandwidth roofline is still the wall; faster math only helps until memory becomes the constraint again.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ with a block size of 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$ and write `output[0..3]`.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ write `output[4..5]`; indices $6$ and $7$ fail `idx < 6` and exit.</span>

<span style="font-size: 14px;">With `input` $= [0, 2, -2, \ldots]$, thread 0 computes $1/(1+e^{0}) = 0.5$, thread 1 computes $1/(1+e^{-2}) \approx 0.881$, thread 2 computes $1/(1+e^{2}) \approx 0.119$. Each lane runs the same `expf`-then-reciprocal sequence regardless of input sign, so the warp never diverges; the only difference between lanes is the data flowing through the SFU.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Treating it as compute-bound.** Even with `expf` the intensity is $\approx 1.5$ FLOP/byte, far below the roofline ridge; coalescing and occupancy, not faster math, set the runtime at scale.</span>
* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up; without `if (idx < N)` the tail threads read and write out of bounds.</span>
* <span style="font-size: 14px;">**Low occupancy exposing the SFU.** With too few warps the `expf` latency stops being hidden behind memory stalls and the SFU can throttle; launch enough blocks to keep many warps resident.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth.</span>

---