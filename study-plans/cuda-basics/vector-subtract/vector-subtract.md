# <span style="font-size: 20px;">Vector Subtraction</span>

<span style="font-size: 14px;">Vector subtraction forms a pointwise difference of two arrays, $C[i] = A[i] - B[i]$. Like its addition twin, it is the textbook **embarrassingly parallel map**: each output is a function of exactly one element from each operand, and no thread ever needs to consult another. It is therefore one of the simplest kernels the platform offers, and what it really measures is not arithmetic at all but the rate at which the GPU can stream operands through global memory.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For every index $i$ in the half-open range $[0, N)$, the kernel computes:</span>

$$
C[i] = A[i] - B[i]
$$

<span style="font-size: 14px;">All three buffers are contiguous, row-major arrays of $N$ 32-bit floats resident in device (global) memory. The minuend `A`, the subtrahend `B`, and the result `C` share the same length and layout, and the only structure available is the index: element $i$ of the output reads element $i$ of each input and nothing else. Note that subtraction is not commutative, so the operand order is part of the contract - swapping `A` and `B` negates every result.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Since the $N$ differences are mutually independent, the decomposition is the obvious one: **one thread per output element**. A one-dimensional grid of one-dimensional blocks blankets the array, and each thread derives its global position from its block and lane identity:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block of 256 threads is the standard pick. It is a whole number of 32-lane **warps**, so no lanes go to waste; it provides enough warps for the scheduler to hide memory latency; and it is compact enough that several blocks coexist on one **SM (Streaming Multiprocessor)**, keeping **occupancy** healthy. Covering the array then takes $\lceil N / 256 \rceil$ blocks.</span>

<span style="font-size: 14px;">Because the grid rounds up to a whole number of blocks, the final block almost always launches more threads than there are leftover elements. Those extra threads must stay silent, which is the entire reason for the `if (idx < N)` guard. Drop the guard and the tail threads load and store beyond the buffers, which is undefined behavior that usually clobbers an adjacent allocation or triggers a fault.</span>

<span style="font-size: 14px;">No `__syncthreads()`, no shared memory, and no cross-thread state appear anywhere. Threads neither produce nor consume each other's results, so the kernel runs as a single flat wave of independent subtractions. That is what makes the map embarrassingly parallel: structurally it is the serial loop with the loop counter handed to it by the hardware.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Per element the kernel performs three global-memory accesses: a load of `A[idx]`, a load of `B[idx]`, and a store of `C[idx]`. Every value is touched once and discarded, so there is nothing to cache. Staging anything in `__shared__` memory or holding it across iterations in registers would only add cost, because shared memory pays for itself through reuse and a map has no reuse to amortize.</span>

<span style="font-size: 14px;">The layout is, however, perfect for the property that actually governs performance: **coalescing**. The 32 lanes of a warp carry consecutive `idx` values, so together they touch 32 consecutive words of `A`, 32 of `B`, and write 32 of `C`. The memory controller satisfies each warp-wide request in the fewest possible transactions, so the kernel sees close to peak effective bandwidth. Unit-stride, aligned access like this is the most favorable case the hardware provides.</span>

<span style="font-size: 14px;">It is worth being precise about what "coalesced" buys here. A warp's 32 lanes requesting 32 consecutive 4-byte words span 128 contiguous bytes, which the controller can pull as a single aligned cache-line-sized transaction per operand. Scatter those same 32 accesses across the address space and the hardware issues up to 32 separate transactions, moving the same useful bytes but wasting most of the bandwidth on transaction overhead. A map gives this for free as long as the base pointers are aligned and the stride is one.</span>

<span style="font-size: 14px;">The pointers the kernel receives already address device memory; the host moved the inputs across the PCIe bus before the launch. The kernel must not dereference host pointers or perform transfers of its own - it works only on the device-resident buffers passed in.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Each output element costs 12 bytes of global traffic - two 4-byte loads and one 4-byte store - against a single floating-point subtraction. The **arithmetic intensity** is thus:</span>

$$
\frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.083 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** model a kernel only becomes compute-bound once its intensity climbs past the ridge point, which lands in the tens of FLOPs per byte on current GPUs. At $0.083$, vector subtraction sits two to three orders of magnitude below that boundary, so it is **deeply memory-bound**. The arithmetic unit spends nearly all its time idle, waiting on operands streaming up from DRAM.</span>

<span style="font-size: 14px;">That single fact settles the whole optimization story. Refining the math is futile, because there is exactly one subtraction. Only bandwidth-side levers move the runtime: coalesced access (already in hand), enough warps in flight to mask DRAM latency (occupancy), and wider transactions. A correct kernel runs about as fast as the memory subsystem permits, with a wall-clock roughly equal to $12N$ bytes divided by achievable bandwidth.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global load is a hundreds-of-cycles affair. The GPU does not cover that latency with deep caches the way a CPU does; it covers it through **massive multithreading**. When a warp fires its loads of `A` and `B` and stalls, the SM scheduler immediately hands the issue slot to another resident warp that is ready. With enough warps per SM - high occupancy - there is always something to issue, and the memory pipeline never drains.</span>

<span style="font-size: 14px;">This is why the launch configuration, not the kernel body, decides throughput for a map. The kernel has no divergent branches (every active lane walks the same path), no synchronization, and no shared-memory pressure, so occupancy is capped only by launching enough blocks. For large $N$ that happens for free: the grid is enormous and the SMs are saturated with warps.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The one-thread-per-element version is already close to optimal: the work is bandwidth-bound and the access is coalesced, so the ceiling is in sight. The remaining slack is small, but two refinements claim it:</span>

<span style="font-size: 14px;">1. **Grid-stride loop**: rather than sizing the grid to exactly span $N$, launch a fixed device-sized grid and let each thread walk multiple elements by striding `blockDim.x * gridDim.x` per step. One configuration then handles any length, launch overhead is amortized over more work per thread, and the kernel stops caring about the precise value of $N$.</span>

<span style="font-size: 14px;">2. **Vectorized loads**: viewing the arrays as `float4` lets a thread move 16 bytes per memory instruction instead of 4. Fewer, wider transactions use the bus more efficiently and trim instruction overhead, edging the kernel nearer the bandwidth roofline.</span>

<span style="font-size: 14px;">Both sit on top of an already-saturated pipeline. The lesson does not change: for a map you cannot outrun the bandwidth roofline, you can only press up against it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ with a block size of 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, giving 8 threads in total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): the four threads form `idx` $= 0, 1, 2, 3$. All satisfy `idx < 6` and write $C[0..3]$.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): the four threads form `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ pass the guard and write $C[4]$ and $C[5]$; indices $6$ and $7$ fail `idx < 6` and return without touching memory.</span>

<span style="font-size: 14px;">With $A = [10, 20, 30, 40, 50, 60]$ and $B = [1, 2, 3, 4, 5, 6]$, the six active threads independently produce $C = [9, 18, 27, 36, 45, 54]$. No thread waits on another; the only idle work is the two guarded tail threads, an unavoidable side effect of rounding the grid up to whole blocks.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up, and the tail block holds threads with `idx >= N`. Without `if (idx < N)` they read and write out of bounds, the most common bug in this kernel.</span>
* <span style="font-size: 14px;">**Swapping operand order.** Subtraction is not commutative, so computing `B[idx] - A[idx]` negates every result; the minuend and subtrahend roles are fixed by the contract.</span>
* <span style="font-size: 14px;">**Expecting arithmetic tweaks to help.** At $\approx 0.083$ FLOP/byte the kernel is memory-bound, so only bandwidth-side changes (coalescing, occupancy, wider loads) move the needle; fusing instructions does nothing.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing splinters a warp's 32 requests into many transactions and craters effective bandwidth; keep access contiguous and aligned.</span>

---