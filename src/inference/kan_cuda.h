#ifndef KAN_CUDA_H
#define KAN_CUDA_H

#ifdef __CUDACC__
    __global__ void kan_forward_kernel(const float* input, float* output, int size);
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Host function to launch the CUDA kernel
void runKANForwardCUDA(const float* input, float* output, int size);

#ifdef __cplusplus
}
#endif

#endif 
