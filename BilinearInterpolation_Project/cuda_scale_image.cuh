#ifndef BILINEAR_FILTER_CUDA_CUH
#define BILINEAR_FILTER_CUDA_CUH

#include "common_structs.cuh"
#include "imageutil.h"

namespace cuda_seq
{
	void scale_image_apply_sobel(png_user_struct* source_image, png_user_struct* result_image);
}


#endif // BILINEAR_FILTER_CUDA_CUH
