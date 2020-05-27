#include <cstdlib>

#include "bilinear_filter_cuda_streams.cuh"

#include <iostream>


#include "cuda_includes.h"
#include "common_structs.cuh"
#include "cuda_kernels.cuh"

namespace cuda_streams_example
{
	
	void create_pixel_precalculation(pixel_precalculation_memory *xy_precalculation, unsigned old_height, unsigned
	                                 new_height, unsigned old_width, unsigned new_width);

	void fill_image_to_scale(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
		pixel_precalculation* d_y_pixel_precalculation_ptr);
	
	void scale_bilinear(png_user_struct* source_image, png_user_struct* image_to_scale)
	{
		auto old_height = source_image->image_info.height;
		auto new_height = image_to_scale->image_info.height;
		auto old_width = source_image->image_info.width;
		auto new_width = image_to_scale->image_info.width;

		pixel_precalculation_memory precalculation_xy;

		create_pixel_precalculation(&precalculation_xy, old_height, new_height, old_width, new_width);

		fill_image_to_scale(image_to_scale, source_image, precalculation_xy.d_x, precalculation_xy.d_y);


		for (cudaStream_t precalculation_cuda_stream : precalculation_xy.precalculation_stream)
		{
			cudaStreamDestroy(precalculation_cuda_stream);
		}
		cudaFreeHost(precalculation_xy.h_x_pinned);
		cudaFreeHost(precalculation_xy.h_y_pinned);
		cudaFree(precalculation_xy.d_x);
		cudaFree(precalculation_xy.d_y);
		
	}

	void create_pixel_precalculation(pixel_precalculation_memory *xy_precalculation,
	                                 unsigned old_height, unsigned new_height, unsigned old_width, unsigned new_width)
	{
		const auto x_pixel_weight_increment = (1.0F / static_cast<float>(new_width)) * static_cast<float>(old_width - 1);
		const auto y_pixel_weight_increment = (1.0F / static_cast<float>(new_height)) * static_cast<float>(old_height - 1);
		
		auto needed_memory_in_bytes_x = sizeof(pixel_precalculation) * new_width;
		auto needed_memory_in_bytes_y = sizeof(pixel_precalculation) * new_height;

		
		pixel_precalculation* d_x_precalculation_ptr;
		pixel_precalculation* d_y_precalculation_ptr;

		pixel_precalculation* h_x_precalculation_ptr;
		pixel_precalculation* h_y_precalculation_ptr;

		cudaMalloc(reinterpret_cast<void**>(&d_x_precalculation_ptr), needed_memory_in_bytes_x);
		cudaMalloc(reinterpret_cast<void**>(&d_y_precalculation_ptr), needed_memory_in_bytes_y);
		
		cudaMallocHost(reinterpret_cast<void**>(&h_x_precalculation_ptr), needed_memory_in_bytes_x);
		cudaMallocHost(reinterpret_cast<void**>(&h_y_precalculation_ptr), needed_memory_in_bytes_y);

		xy_precalculation->d_x = d_x_precalculation_ptr;
		xy_precalculation->d_y = d_y_precalculation_ptr;

		xy_precalculation->h_x_pinned = h_x_precalculation_ptr;
		xy_precalculation->h_y_pinned = h_y_precalculation_ptr;


		
		dim3 block_size_x(32);
		dim3 block_size_y(32);
		
		auto bx = (new_width + block_size_x.x - 1) / block_size_x.x;
		auto by = (new_height + block_size_y.x - 1) / block_size_y.x;

		auto grid_size_x = dim3(bx);
		auto grid_size_y = dim3(by);
		

		cudaStream_t x_precalculation_stream = xy_precalculation->precalculation_stream[0];
		cudaStream_t y_precalculation_stream = xy_precalculation->precalculation_stream[1];
		
		cudaStreamCreate(&x_precalculation_stream);
		cudaStreamCreate(&y_precalculation_stream);
		
		pixel_precalculation_kernel <<< grid_size_x, block_size_x, 0, x_precalculation_stream >>> (xy_precalculation->d_x, x_pixel_weight_increment, new_width);  // NOLINT(clang-diagnostic-unused-value)
		pixel_precalculation_kernel <<< grid_size_y, block_size_y, 0, y_precalculation_stream >>> (xy_precalculation->d_y, y_pixel_weight_increment, new_height); // NOLINT(clang-diagnostic-unused-value)
		
		cudaMemcpyAsync(xy_precalculation->h_x_pinned, xy_precalculation->d_x, needed_memory_in_bytes_x, cudaMemcpyDeviceToHost, xy_precalculation->precalculation_stream[0]);
		cudaStreamSynchronize(xy_precalculation->precalculation_stream[0]);

		
		//precalculation_xy->h_x_pinned now available
		
	}

	void fill_image_to_scale(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
		pixel_precalculation* d_y_pixel_precalculation_ptr)
	{
		auto dimensions_size_in_bytes = sizeof(dimensions_info);
		auto source_png_size_in_bytes = sizeof(png_byte) * source_image->image_info.width * source_image->image_info.height;
		auto image_to_scale_size_in_bytes = sizeof(png_byte) * image_to_scale->image_info.width * image_to_scale->image_info.height;
		auto sobel_image_size_in_bytes = image_to_scale_size_in_bytes;

		auto needed_memory_in_bytes = dimensions_size_in_bytes + source_png_size_in_bytes + image_to_scale_size_in_bytes + sobel_image_size_in_bytes;

		_int8* allocated_memory_on_gpu_p;

		cudaMalloc(reinterpret_cast<void**>(&allocated_memory_on_gpu_p), needed_memory_in_bytes);

		d_scale_params hd_params;
		d_sobel_params hd_sobel_params;

		hd_params.x_precalculation_p = d_x_pixel_precalculation_ptr;
		hd_params.y_precalculation_p = d_y_pixel_precalculation_ptr;
		hd_params.dimensions_info_p = reinterpret_cast<dimensions_info*>(allocated_memory_on_gpu_p);
		hd_params.source_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes);
		hd_params.image_to_scale_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes);
		hd_sobel_params.source_bytes_sequential_p = hd_params.image_to_scale_bytes_sequential_p;
		hd_sobel_params.result_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes + image_to_scale_size_in_bytes);


		//pp_array that contains the source image needs to be flattened for fast memory allocation on gpu
		png_bytep png_source_bytes_p = png_util_create_flat_bytes_p_from_row_pp(source_image->png_rows, source_image->image_info.width, source_image->image_info.height, source_png_size_in_bytes);

		dimensions_info dimensions_inf;
		dimensions_inf.image_to_scale_width = image_to_scale->image_info.width;
		dimensions_inf.image_to_scale_height = image_to_scale->image_info.height;
		dimensions_inf.source_image_width = source_image->image_info.width;


		
		cudaStream_t image_processing_cuda_stream;
		
		cudaStreamCreate(&image_processing_cuda_stream);
		
		cudaMemcpyAsync(hd_params.source_bytes_sequential_p, png_source_bytes_p, source_png_size_in_bytes, cudaMemcpyHostToDevice, image_processing_cuda_stream);
		cudaMemcpyAsync(hd_params.dimensions_info_p, &dimensions_inf, sizeof(dimensions_info), cudaMemcpyHostToDevice, image_processing_cuda_stream);

		hd_sobel_params.dimensions_inf_p = hd_params.dimensions_info_p;
		

		
		dim3 block_size(32, 32);
		
		auto blocks_in_x_direction = (image_to_scale->image_info.width + block_size.x - 1) / block_size.x;
		auto blocks_in_y_direction = (image_to_scale->image_info.height + block_size.y - 1) / block_size.y;
		
		auto grid_size = dim3(blocks_in_x_direction, blocks_in_y_direction);

		
		parallel_tasks_bilinear_nn_sobel << <grid_size, block_size, 0, image_processing_cuda_stream>> > (hd_params, hd_sobel_params);

		cudaMemcpy(image_to_scale->png_rows[0], hd_sobel_params.result_bytes_sequential_p, image_to_scale_size_in_bytes, cudaMemcpyDeviceToHost);

		cudaStreamDestroy(image_processing_cuda_stream);
		cudaFree(allocated_memory_on_gpu_p);
	}
}
