#ifndef BILINEAR_FILTER_CUDA_CUH
#define BILINEAR_FILTER_CUDA_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "imageutil.h"

void scale_bilinear_cuda(png_user_struct* source_image, png_user_struct* image_to_scale);


__global__ void pixel_precalculation_gpu(int* precalculation);

#endif // BILINEAR_FILTER_CUDA_CUH
