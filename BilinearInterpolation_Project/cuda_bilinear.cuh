#ifndef CUDA_BILINEAR_CUH
#define CUDA_BILINEAR_CUH

#include "cuda_includes.h"
#include "common_structs.cuh"

namespace device
{
	__device__ void apply_bilinear_filter(d_scale_params params, unsigned x, unsigned y);
}

namespace global
{
	__global__ void apply_bilinear_filter(d_scale_params params, unsigned y_offset, unsigned y_max);
}


#endif // CUDA_BILINEAR_CUH
