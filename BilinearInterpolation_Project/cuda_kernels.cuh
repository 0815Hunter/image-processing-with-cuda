#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "cuda_includes.h"
#include "common_structs.cuh"

__global__ void pixel_precalculation_kernel(pixel_precalculation* precalculation, const double pixel_weight_increment, unsigned int N);


__global__ void parallel_tasks_bilinear_nn(d_scale_params f_params);

__global__ void parallel_tasks_bilinear_nn_sobel(d_scale_params f_params, d_sobel_params s_params);

#endif // CUDA_KERNELS_CUH
