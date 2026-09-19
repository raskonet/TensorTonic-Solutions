#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    int t= threadIdx.x + blockIdx.x * blockDim.x;
    if(t<N){
        float x=input[t];
        output[t]=(expf(x)-expf(-x))/(expf(x)+expf(-x));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}