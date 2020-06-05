#ifndef CUDA_NEAREST_NEIGHBOR_CUH
#define CUDA_NEAREST_NEIGHBOR_CUH

#include "cuda_includes.h"
#include "common_structs.cuh"


namespace device
{
	__device__ void apply_nearest_neighbor(d_scale_params params, unsigned x, unsigned y);
}

namespace global
{
	__global__ void apply_nearest_neighbor(d_scale_params params, unsigned y_offset, unsigned y_max);
}

#endif // CUDA_NEAREST_NEIGHBOR_CUH
