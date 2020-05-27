#ifndef SOBEL_KERNEL_CUH
#define SOBEL_KERNEL_CUH
#include "cuda_includes.h"
#include "common_structs.cuh"

__global__ void apply_sobel_filter(d_sobel_params params);

#endif // SOBEL_KERNEL_CUH
