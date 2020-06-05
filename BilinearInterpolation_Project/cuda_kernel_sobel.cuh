#ifndef SOBEL_KERNEL_CUH
#define SOBEL_KERNEL_CUH
#include "cuda_includes.h"
#include "common_structs.cuh"

__global__ void apply_sobel_filter(d_sobel_params params, unsigned y_offset, unsigned y_max);

namespace sobel_framed_blocks
{
	__global__ void apply_sobel_filter(d_sobel_params params);
}

namespace sobel_cooperative_groups_tile2
{
	__global__ void apply_sobel_filter(d_sobel_params params);
}


namespace sobel_cooperative_groups_tile16_8
{
	__global__ void apply_sobel_filter(d_sobel_params params);
}

#endif // SOBEL_KERNEL_CUH
