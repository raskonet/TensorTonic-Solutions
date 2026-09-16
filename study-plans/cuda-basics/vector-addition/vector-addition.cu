#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t < N){
        C[t]=A[t]+B[t];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}