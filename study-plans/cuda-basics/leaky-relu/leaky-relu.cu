#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    int t = threadIdx.x + blockDim.x * blockIdx.x;
    if(t < N){
        output[t]=input[t] >=0 ? input[t] : alpha * input[t];
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}