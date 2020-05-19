#include <cstdio>
#include <cstdlib>
#include <png.h>
#include "bilinear_filter_cuda.cuh"
#include "imageutil.h"


typedef struct dimensions_def
{
	png_uint_32 source_image_width;
	png_uint_32 image_to_scale_width;
	png_uint_32 image_to_scale_height;
}dimensions;

typedef struct FillParams_def
{
	PixelPrecalculation* x_precalculation;
	PixelPrecalculation* y_precalculation;
	png_bytep source_bytes_sequential_p;
	png_bytep image_to_scale_bytes_sequential_p;
	dimensions *dimensions_inf_p;
	
}FillParams;

PixelPrecalculation* create_pixel_precalculation_for_x_row_cuda(unsigned int old_width, unsigned int new_width);

PixelPrecalculation* create_pixel_precalculation_for_y_row_cuda(unsigned int old_height, unsigned int new_height);

PixelPrecalculation* create_pixel_precalculation_cuda(unsigned int old_pixel_array_size, unsigned int new_pixel_array_size);

__device__ void precalc(PixelPrecalculation* precalculation, const float pixel_weight_increment);

__global__ void fill_image_to_scale_cuda1(FillParams* params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= params->dimensions_inf_p->image_to_scale_width) || (y >= params->dimensions_inf_p->image_to_scale_height))
	{
		return;
	}

	auto y_pixel_front = params->y_precalculation[y].frontPixel;
	auto y_pixel_rear = params->y_precalculation[y].rearPixel;
	auto y_front_weight = params->y_precalculation[y].frontWeight;
	auto y_rear_weight = params->y_precalculation[y].rearWeight;

	auto x_pixel_front = params->x_precalculation[x].frontPixel;
	auto x_pixel_rear = params->x_precalculation[x].rearPixel;
	auto x_front_weight = params->x_precalculation[x].frontWeight;
	auto x_rear_weight = params->x_precalculation[x].rearWeight;

	auto base_upper_row = y_pixel_front * params->dimensions_inf_p->source_image_width;
	auto p_00_index = base_upper_row + x_pixel_front;
	auto p_01_index = base_upper_row + x_pixel_rear;

	auto base_lower_row = y_pixel_rear * params->dimensions_inf_p->source_image_width;
	auto p_10_index = base_lower_row + x_pixel_front;
	auto p_11_index = base_lower_row + x_pixel_rear;


	auto source_bytes = params->source_bytes_sequential_p;

	png_byte point_00_01_value = source_bytes[p_00_index] * x_front_weight + source_bytes[p_01_index] * x_rear_weight;
	png_byte point_10_11_value = source_bytes[p_10_index] * x_front_weight + source_bytes[p_11_index] * x_rear_weight;

	png_byte pixel_p_value = point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight;

	auto i = y * params->dimensions_inf_p->image_to_scale_width + x;

	params->image_to_scale_bytes_sequential_p[i] = pixel_p_value;

}

__global__ void pixel_precalculation_gpu(PixelPrecalculation* precalculation, const float pixel_weight_increment)
{
	precalc(precalculation, pixel_weight_increment);
}

__device__ void precalc(PixelPrecalculation *precalculation, const float pixel_weight_increment)
{
	const auto i = threadIdx.x;
	const auto current_pixel_weight = pixel_weight_increment * i;
	const auto source_image_pixel_for_front_pixel_x = static_cast<unsigned int>(current_pixel_weight);
	const auto rear_pixel = source_image_pixel_for_front_pixel_x + 1;


	precalculation[i].frontPixel = source_image_pixel_for_front_pixel_x;
	precalculation[i].rearPixel = rear_pixel;

	const auto weight = current_pixel_weight - static_cast<float>(source_image_pixel_for_front_pixel_x);

	precalculation[i].frontWeight = 1 - weight;
	precalculation[i].rearWeight = weight;
	
}


void fill_image_to_scale_cuda(png_user_struct* image_to_scale, png_user_struct* source_image, PixelPrecalculation* d_x_pixel_precalculation_ptr, PixelPrecalculation* d_y_pixel_precalculation_ptr);


void scale_bilinear_cuda(png_user_struct* source_image, png_user_struct* image_to_scale)
{
	const auto old_height = source_image->image_info.height;
	const auto new_height = image_to_scale->image_info.height;
	const auto old_width = source_image->image_info.width;
	const auto new_width = image_to_scale->image_info.width;

	auto d_x_pixel_precalculation_ptr = create_pixel_precalculation_for_x_row_cuda(old_width, new_width);
	
	auto d_y_pixel_precalculation_ptr = create_pixel_precalculation_for_y_row_cuda(old_height, new_height);

	fill_image_to_scale_cuda(image_to_scale, source_image, d_x_pixel_precalculation_ptr, d_y_pixel_precalculation_ptr);

	cudaFree(d_x_pixel_precalculation_ptr);
	cudaFree(d_y_pixel_precalculation_ptr);

	printf("Hoarray");
}

PixelPrecalculation* create_pixel_precalculation_for_x_row_cuda(const unsigned int old_width, const unsigned int new_width)
{
	return create_pixel_precalculation_cuda(old_width, new_width);
}

PixelPrecalculation* create_pixel_precalculation_for_y_row_cuda(const unsigned int old_height, const unsigned int new_height)
{
	return create_pixel_precalculation_cuda(old_height, new_height);
}

PixelPrecalculation* create_pixel_precalculation_cuda(const unsigned int old_pixel_array_size,
	const unsigned int new_pixel_array_size)
{


	PixelPrecalculation* d_pixel_precalculation_for_x_rows;
	

	cudaMalloc(reinterpret_cast<void**>(&d_pixel_precalculation_for_x_rows), sizeof(PixelPrecalculation) * new_pixel_array_size);

	

	//old_pixel_array_size - 1, the last pixel is (old_pixel_array_size - 1)
	const float pixel_weight_increment = (1.0F / (float)new_pixel_array_size) * (float)(old_pixel_array_size - 1);


	pixel_precalculation_gpu <<< 1, new_pixel_array_size >>> (d_pixel_precalculation_for_x_rows, pixel_weight_increment);

	return d_pixel_precalculation_for_x_rows;
}


void fill_image_to_scale_cuda(png_user_struct* image_to_scale, png_user_struct* source_image, PixelPrecalculation* d_x_pixel_precalculation_ptr,
	PixelPrecalculation* d_y_pixel_precalculation_ptr)
{
	auto source_png_bytes_size = sizeof(png_byte) * source_image->image_info.width * source_image->image_info.height;
	
	png_bytep png_row_col_p = static_cast<png_bytep>(malloc(source_png_bytes_size) );

	png_bytep png_row_col_p_it = png_row_col_p;

	//h_png_source_image_bytepp: liegen die bytes nicht alle ab h_png_source_image_bytepp[0]
	for (png_uint_32 y = 0; y < source_image->image_info.height;y++)
	{
		for(png_uint_32 x = 0; x < source_image->image_info.width; x++)
		{
			*png_row_col_p_it = source_image->png_rows[y][x];
			png_row_col_p_it++;
		}
	}


	FillParams d_params;
	FillParams *d_params_p = &d_params;

	d_params.x_precalculation = d_x_pixel_precalculation_ptr;
	d_params.y_precalculation = d_y_pixel_precalculation_ptr;
	
	auto image_to_scale_bytes_size = sizeof(png_byte) * image_to_scale->image_info.width * image_to_scale->image_info.height;
	cudaMalloc(reinterpret_cast<void**>(&d_params_p->image_to_scale_bytes_sequential_p), image_to_scale_bytes_size);
	
	
	cudaMalloc(reinterpret_cast<void**>(&d_params_p->source_bytes_sequential_p), source_png_bytes_size);
	cudaMemcpy(d_params_p->source_bytes_sequential_p, png_row_col_p, source_png_bytes_size, cudaMemcpyHostToDevice);


	
	dimensions *h_dimensions_inf_p = static_cast<dimensions *>(malloc(sizeof(dimensions)));
	h_dimensions_inf_p->image_to_scale_width = image_to_scale->image_info.width;
	h_dimensions_inf_p->image_to_scale_height = image_to_scale->image_info.height;
	h_dimensions_inf_p->source_image_width = source_image->image_info.width;

	
	cudaMalloc(reinterpret_cast<void**>(&d_params_p->dimensions_inf_p), sizeof(dimensions));
	cudaMemcpy(d_params_p->dimensions_inf_p, h_dimensions_inf_p, sizeof(dimensions), cudaMemcpyHostToDevice);

	auto d_image_to_scale_bytes_sequential_p = d_params_p->image_to_scale_bytes_sequential_p;
	
	cudaMalloc(reinterpret_cast<void**>(&d_params_p), sizeof(FillParams));
	cudaMemcpy(d_params_p, &d_params, sizeof(FillParams), cudaMemcpyHostToDevice);
	
	dim3 blockSize(32, 32);

	auto bx = (image_to_scale->image_info.width + blockSize.x - 1) / blockSize.x;
	
	auto by = (image_to_scale->image_info.height  + blockSize.y - 1) / blockSize.y;

	dim3 gridSize = dim3(bx, by);

	fill_image_to_scale_cuda1 << <gridSize, blockSize >> > (d_params_p);


	png_bytep png_scaled_bytes_sequential_p = static_cast<png_bytep>(malloc(image_to_scale_bytes_size));

	cudaMemcpy(png_scaled_bytes_sequential_p, d_image_to_scale_bytes_sequential_p, image_to_scale_bytes_size, cudaMemcpyDeviceToHost);


	for(png_uint_32 y = 0; y < image_to_scale->image_info.height; y++)
	{
		image_to_scale->png_rows[y] = &png_scaled_bytes_sequential_p[y * image_to_scale->image_info.width];
	}
}
