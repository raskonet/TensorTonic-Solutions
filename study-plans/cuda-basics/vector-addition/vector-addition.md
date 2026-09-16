# <span style="font-size: 20px;">Vector Addition</span>

<span style="font-size: 14px;">Vector addition computes a pointwise sum of two arrays, $C[i] = A[i] + B[i]$. It is the canonical **embarrassingly parallel map**: every output depends on exactly one input from each operand, with zero communication between threads. That makes it the simplest possible CUDA kernel and, more importantly, a pure benchmark of one thing only - how fast the GPU can move data through global memory.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$, the kernel evaluates:</span>

$$
C[i] = A[i] + B[i]
$$

<span style="font-size: 14px;">All three arrays are contiguous, row-major buffers of $N$ 32-bit floats living in device (global) memory. There is no structure to exploit beyond the index itself: output element $i$ reads only input element $i$ from each operand and writes only output element $i$. Nothing is shared, reused, or reordered.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $N$ outputs are mutually independent, the natural decomposition is **one thread per output element**. The launch covers the array with a one-dimensional grid of one-dimensional blocks, and each thread reconstructs its global position from its block and lane coordinates:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 threads is the conventional choice. It is a multiple of the 32-lane **warp** (so no lanes are wasted), it is large enough to give the scheduler many warps per block for latency hiding, and it is small enough that many blocks fit on a single **SM (Streaming Multiprocessor)**, keeping **occupancy** high. With 256 threads per block, the grid needs $\lceil N / 256 \rceil$ blocks to cover the array.</span>

<span style="font-size: 14px;">Rounding the grid up means the last block usually contains more threads than there are remaining elements. Those surplus threads must do nothing, which is why the kernel body is guarded by `if (idx < N)`. Without that bounds check, the tail threads would read and write past the end of the buffers - undefined behavior that typically corrupts neighboring allocations or faults.</span>

<span style="font-size: 14px;">There is no `__syncthreads()` and no shared state anywhere in this kernel. Threads never need to see each other's results, so the entire computation is one flat wave of independent work. This is what "embarrassingly parallel" means in practice: the parallel version is structurally identical to the serial loop, just with the loop index supplied by the hardware instead of a counter.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel touches three global arrays per element: it loads `A[idx]`, loads `B[idx]`, and stores `C[idx]`. There are no intermediate values worth caching in `__shared__` memory and nothing to keep in registers beyond the single sum, because each datum is used exactly once and then never again. Shared memory exists to enable reuse; a map has no reuse, so introducing shared memory here would only add overhead.</span>

<span style="font-size: 14px;">The access pattern is, however, ideal for the one thing that does matter: **coalescing**. The 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `A` (and of `B`) and write 32 consecutive addresses of `C`. The memory controller serves each such warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth. This contiguous, unit-stride layout is the best case the hardware offers.</span>

<span style="font-size: 14px;">The pointers passed to the kernel already point at device memory - the host copied the inputs across the PCIe bus before launch. The kernel must never dereference host pointers or attempt its own transfers; it operates purely on the device-resident buffers it is given.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per output element the kernel moves 12 bytes of global memory - two 4-byte loads and one 4-byte store - and performs exactly one floating-point addition. Its **arithmetic intensity** is therefore about:</span>

$$
\frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.083 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** model, a kernel is compute-bound only when its intensity exceeds the GPU's ridge point, which sits in the range of tens of FLOPs per byte on modern hardware. At $0.083$, vector addition is two to three orders of magnitude below that line: it is **deeply memory-bound**. The single adder sits idle almost the entire time, waiting for operands to arrive from DRAM.</span>

<span style="font-size: 14px;">This classification dictates the entire optimization story. Cleverer arithmetic is pointless - there is only one add. The only levers that change runtime are those that raise effective memory bandwidth: coalesced access (already optimal), enough warps in flight to hide DRAM latency (occupancy), and wider memory transactions. A correct vector-add kernel is essentially as fast as the device's memory subsystem allows; the runtime is just $12N$ bytes divided by achievable bandwidth.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles of latency. The GPU does not hide that latency with large caches the way a CPU does; instead it hides it with **massive multithreading**. When a warp issues its loads of `A` and `B` and stalls waiting for the data, the SM's scheduler immediately switches to another resident warp that is ready to run. With enough warps per SM - high occupancy - there is always other work to issue, and the memory pipeline stays saturated.</span>

<span style="font-size: 14px;">This is why the launch configuration, not the kernel logic, determines performance for a map. The kernel has no branches that diverge (every active thread takes the same path), no synchronization, and no shared-memory pressure, so occupancy is limited only by having launched enough blocks. For very large $N$ this is automatic; the grid is huge and the SMs are flooded with warps.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The one-thread-per-element kernel is already near-optimal because the problem is bandwidth-bound and the access is coalesced. There is little headroom, but two refinements squeeze out the remainder:</span>

<span style="font-size: 14px;">1. **Grid-stride loop**: instead of sizing the grid to exactly cover $N$, launch a fixed, device-sized grid and let each thread process multiple elements by striding through the array in steps of `blockDim.x * gridDim.x`. This decouples the launch from $N$, lets one configuration handle any array size, and improves work-per-thread so launch overhead is amortized.</span>

<span style="font-size: 14px;">2. **Vectorized loads**: reinterpreting the arrays as `float4` lets each thread load and store 16 bytes per instruction instead of 4. Fewer, wider memory transactions use the bus more efficiently and reduce instruction overhead, nudging the kernel closer to the memory-bandwidth ceiling.</span>

<span style="font-size: 14px;">Both are micro-optimizations on top of an already-saturated memory pipeline. The headline lesson stands: for a map, you cannot beat the bandwidth roofline, you can only approach it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ with a block size of 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$. All pass `idx < 6`, so they write $C[0..3]$.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ pass the bounds check and write $C[4]$ and $C[5]$; indices $6$ and $7$ fail `idx < 6` and exit without touching memory.</span>

<span style="font-size: 14px;">With $A = [1, 2, 3, 4, 5, 6]$ and $B = [10, 20, 30, 40, 50, 60]$, the six active threads independently produce $C = [11, 22, 33, 44, 55, 66]$. No thread waits on any other; the only "wasted" work is the two guarded tail threads, an unavoidable consequence of rounding the grid up to a whole number of blocks.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size, the grid rounds up and the tail block has threads with `idx >= N`. Without `if (idx < N)`, those threads read and write out of bounds. This is the single most common vector-add bug.</span>
* <span style="font-size: 14px;">**Expecting arithmetic optimizations to help.** Because the kernel is memory-bound at $\approx 0.083$ FLOP/byte, only bandwidth-side changes (coalescing, occupancy, wider loads) matter; fusing or reducing instructions does nothing.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth. Keep access contiguous and aligned.</span>
* <span style="font-size: 14px;">**Reading results before `cudaDeviceSynchronize()`.** The launch is asynchronous; reading `C` too early observes stale, partially written data.</span>

---