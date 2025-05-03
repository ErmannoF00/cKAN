#include "kan_cuda.h"
#include <cuda_runtime.h>

__global__ void kan_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x * x; // Placeholder for φ(x)
    }
}

void runKANForwardCUDA(const float* input, float* output, int size) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kan_forward_kernel<<<blocks, threads>>>(d_input, d_output, size);

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
