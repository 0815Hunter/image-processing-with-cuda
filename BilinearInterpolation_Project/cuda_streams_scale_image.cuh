#ifndef BILINEAR_FILTER_CUDA_STREAMS_CUH
#define BILINEAR_FILTER_CUDA_STREAMS_CUH

#include "imageutil.h"

namespace cuda_streams_example
{
	void scale_image_apply_sobel(png_user_struct* source_image, png_user_struct* result_image);
}


#endif // BILINEAR_FILTER_CUDA_STREAMS_CUH
