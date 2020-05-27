#ifndef COMMON_STRUCTS_H
#define COMMON_STRUCTS_H

#include <png.h>

#include "cuda_includes.h"

typedef struct pixel_precalculation_def {
	unsigned int front_pixel;
	unsigned int rear_pixel;
	float front_weight;
	float rear_weight;
} pixel_precalculation;

typedef struct dimensions_info_def
{
	png_uint_32 source_image_width;
	png_uint_32 image_to_scale_width;
	png_uint_32 image_to_scale_height;
}dimensions_info;

typedef struct d_scale_params_def
{
	pixel_precalculation* x_precalculation_p;
	pixel_precalculation* y_precalculation_p;
	dimensions_info* dimensions_info_p;
	png_bytep source_bytes_sequential_p;
	volatile png_bytep image_to_scale_bytes_sequential_p;
}d_scale_params;

typedef struct d_sobel_params_def
{
	dimensions_info* dimensions_inf_p;
	png_bytep source_bytes_sequential_p;
	png_bytep result_bytes_sequential_p;
}d_sobel_params;

typedef struct pixel_precalculation_memory_def
{
	pixel_precalculation* d_x;
	pixel_precalculation* d_y;
	void* allocated_gpu_memory;
}pixel_precalculation_memory;

namespace cuda_streams_example
{
	typedef struct pixel_precalculation_memory_def
	{
		pixel_precalculation* d_x;
		pixel_precalculation* d_y;
		pixel_precalculation* h_x_pinned;
		pixel_precalculation* h_y_pinned;
		cudaStream_t precalculation_stream[2];
	}pixel_precalculation_memory;
}


#endif // COMMON_STRUCTS_H
