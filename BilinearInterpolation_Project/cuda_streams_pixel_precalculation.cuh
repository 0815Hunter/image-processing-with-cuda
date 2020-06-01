#ifndef CUDA_STREAMS_PIXEL_PRECALCULATION_CUH
#define CUDA_STREAMS_PIXEL_PRECALCULATION_CUH
#include "common_structs.cuh"

namespace cuda_streams_example
{
	void create_pixel_precalculation(pixel_precalculation_memory* xy_precalculation,
		unsigned old_height, unsigned new_height, unsigned old_width, unsigned new_width);
}

#endif // CUDA_STREAMS_PIXEL_PRECALCULATION_CUH
