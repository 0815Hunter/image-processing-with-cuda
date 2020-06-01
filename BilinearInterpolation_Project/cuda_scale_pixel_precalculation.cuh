#ifndef CUDA_SCALE_PIXEL_PRECALCULATION_CUH
#define CUDA_SCALE_PIXEL_PRECALCULATION_CUH
#include "common_structs.cuh"

namespace cuda_seq
{
	void create_pixel_precalculation(pixel_precalculation_memory* precalculation_xy, unsigned int old_width, unsigned int new_width, const unsigned old_height, const unsigned new_height);
}

#endif // CUDA_SCALE_PIXEL_PRECALCULATION_CUH
