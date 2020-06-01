#include <cstdlib>
#include <iostream>

#include "cuda_streams_scale_image.cuh"
#include "cuda_includes.h"
#include "common_structs.cuh"
#include "cuda_kernels.cuh"

namespace cuda_streams_example
{
	
	void create_pixel_precalculation(pixel_precalculation_memory *xy_precalculation, unsigned old_height, unsigned
	                                 new_height, unsigned old_width, unsigned new_width);

	void scale_and_sobel(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
		pixel_precalculation* d_y_pixel_precalculation_ptr);
	
	void scale_image_apply_sobel(png_user_struct* source_image, png_user_struct* result_image)
	{
		auto old_height = source_image->image_info.height;
		auto new_height = result_image->image_info.height;
		auto old_width = source_image->image_info.width;
		auto new_width = result_image->image_info.width;

		pixel_precalculation_memory precalculation_xy;

		create_pixel_precalculation(&precalculation_xy, old_height, new_height, old_width, new_width);

		scale_and_sobel(result_image, source_image, precalculation_xy.d_x, precalculation_xy.d_y);


		for (cudaStream_t precalculation_cuda_stream : precalculation_xy.precalculation_stream)
		{
			cudaStreamDestroy(precalculation_cuda_stream);
		}
		cudaFreeHost(precalculation_xy.h_x_pinned);
		cudaFreeHost(precalculation_xy.h_y_pinned);
		cudaFree(precalculation_xy.d_x);
		cudaFree(precalculation_xy.d_y);
		
	}

	void scale_and_sobel(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
		pixel_precalculation* d_y_pixel_precalculation_ptr)
	{
		auto dimensions_size_in_bytes = sizeof(dimensions_info);
		auto source_png_size_in_bytes = sizeof(png_byte) * source_image->image_info.width * source_image->image_info.height;
		auto result_image_size_in_bytes = sizeof(png_byte) * result_image->image_info.width * result_image->image_info.height;
		auto sobel_image_size_in_bytes = result_image_size_in_bytes;

		auto needed_memory_in_bytes = dimensions_size_in_bytes + source_png_size_in_bytes + result_image_size_in_bytes + sobel_image_size_in_bytes;

		_int8* allocated_memory_on_gpu_p;

		cudaMalloc(reinterpret_cast<void**>(&allocated_memory_on_gpu_p), needed_memory_in_bytes);

		d_scale_params d_scale_params;
		d_sobel_params d_sobel_params;

		d_scale_params.x_precalculation_p = d_x_pixel_precalculation_ptr;
		d_scale_params.y_precalculation_p = d_y_pixel_precalculation_ptr;
		d_scale_params.dimensions_info_p = reinterpret_cast<dimensions_info*>(allocated_memory_on_gpu_p);
		d_scale_params.source_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes);
		d_scale_params.result_image_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes);
		d_sobel_params.source_bytes_sequential_p = d_scale_params.result_image_bytes_sequential_p;
		d_sobel_params.result_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes + result_image_size_in_bytes);


		//pp_array that contains the source image needs to be flattened for fast memory allocation on gpu
		png_bytep png_source_bytes_p = png_util_create_flat_bytes_p_from_row_pp(source_image->png_rows, source_image->image_info.width, source_image->image_info.height, source_png_size_in_bytes);

		dimensions_info dimensions_inf;
		dimensions_inf.result_image_width = result_image->image_info.width;
		dimensions_inf.result_image_height = result_image->image_info.height;
		dimensions_inf.source_image_width = source_image->image_info.width;


		
		cudaStream_t image_processing_cuda_stream;
		
		cudaStreamCreate(&image_processing_cuda_stream);
		
		cudaMemcpyAsync(d_scale_params.source_bytes_sequential_p, png_source_bytes_p, source_png_size_in_bytes, cudaMemcpyHostToDevice, image_processing_cuda_stream);
		cudaMemcpyAsync(d_scale_params.dimensions_info_p, &dimensions_inf, sizeof(dimensions_info), cudaMemcpyHostToDevice, image_processing_cuda_stream);

		d_sobel_params.dimensions_inf_p = d_scale_params.dimensions_info_p;
		
		
		dim3 block_size(32, 32);
		
		auto blocks_in_x_direction = (result_image->image_info.width + block_size.x - 1) / block_size.x;
		auto blocks_in_y_direction = (result_image->image_info.height + block_size.y - 1) / block_size.y;
		
		auto grid_size = dim3(blocks_in_x_direction, blocks_in_y_direction);

		
		parallel_tasks_bilinear_nn_sobel << <grid_size, block_size, 0, image_processing_cuda_stream>> > (d_scale_params, d_sobel_params);

		cudaMemcpyAsync(result_image->png_rows[0], d_sobel_params.result_bytes_sequential_p, result_image_size_in_bytes, cudaMemcpyDeviceToHost, image_processing_cuda_stream);

		cudaStreamDestroy(image_processing_cuda_stream);
		cudaFree(allocated_memory_on_gpu_p);
	}
}
