#ifndef KAN_CUDA_H
#define KAN_CUDA_H

#ifdef __CUDACC__
extern "C" {
#endif

__global__ void kan_forward_kernel(const float* input, float* output, int size);
void runKANForwardCUDA(const float* input, float* output, int size);

#ifdef __CUDACC__
}
#endif

#endif // KAN_CUDA_H
