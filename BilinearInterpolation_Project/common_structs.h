#ifndef COMMON_STRUCTS_H
#define COMMON_STRUCTS_H

#include <png.h>


typedef struct pixel_precalculation_def {
	unsigned int frontPixel;
	unsigned int rearPixel;
	float frontWeight;
	float rearWeight;
} pixel_precalculation;

typedef struct dimensions_def
{
	png_uint_32 source_image_width;
	png_uint_32 image_to_scale_width;
	png_uint_32 image_to_scale_height;
}dimensions;

typedef struct fill_params_def
{
	pixel_precalculation* x_precalculation;
	pixel_precalculation* y_precalculation;
	dimensions* dimensions_inf_p;
	png_bytep source_bytes_sequential_p;
	png_bytep image_to_scale_bytes_sequential_p;
}fill_params;

typedef struct pixel_precalculation_memory_def
{
	pixel_precalculation* x;
	pixel_precalculation* y;
	void* allocated_gpu_data;
}pixel_precalculation_memory;


#endif // COMMON_STRUCTS_H
