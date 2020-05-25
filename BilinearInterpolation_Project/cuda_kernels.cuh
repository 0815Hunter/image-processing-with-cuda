#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "cuda_includes.h"
#include "common_structs.h"

__device__ void apply_bilinear_filter(fill_params params, unsigned x, unsigned y);

__device__ void apply_nearest_neighbor(fill_params params, unsigned x, unsigned y);

__global__ void parallel_tasks(fill_params params);

__global__ void k_pixel_precalculation(pixel_precalculation* precalculation, const float pixel_weight_increment, unsigned int N);

#endif // CUDA_KERNELS_CUH
