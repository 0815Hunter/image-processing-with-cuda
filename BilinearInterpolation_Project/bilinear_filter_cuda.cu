#include <cstdlib>

#include "bilinear_filter_cuda.cuh"

#include "cuda_includes.h"
#include "common_structs.cuh"
#include "cuda_kernels.cuh"

namespace cuda_seq
{

	void create_pixel_precalculation(pixel_precalculation_memory* precalculation_xy, unsigned int old_width, unsigned int new_width, const unsigned old_height, const unsigned new_height);

	void fill_image_to_scale(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
		pixel_precalculation* d_y_pixel_precalculation_ptr);

	void scale_bilinear(png_user_struct* source_image, png_user_struct* image_to_scale)
	{
		const auto old_height = source_image->image_info.height;
		const auto new_height = image_to_scale->image_info.height;
		const auto old_width = source_image->image_info.width;
		const auto new_width = image_to_scale->image_info.width;

		pixel_precalculation_memory precalculation_xy;

		create_pixel_precalculation(&precalculation_xy, old_width, new_width, old_height, new_height);

		fill_image_to_scale(image_to_scale, source_image, precalculation_xy.d_x, precalculation_xy.d_y);

		cudaFree(precalculation_xy.allocated_gpu_memory);
	}

	void create_pixel_precalculation(pixel_precalculation_memory* precalculation_xy,
		const unsigned int old_width, const unsigned int new_width, const unsigned old_height, const unsigned new_height)
	{
		auto needed_memory_in_bytes = sizeof(pixel_precalculation) * new_width + sizeof(pixel_precalculation) * new_height;

		auto offset_to_y_precalculation_data = new_width;

		pixel_precalculation* d_memory_on_gpu_p;

		cudaMalloc(reinterpret_cast<void**>(&d_memory_on_gpu_p), needed_memory_in_bytes);
		precalculation_xy->allocated_gpu_memory = d_memory_on_gpu_p;
		precalculation_xy->d_x = d_memory_on_gpu_p;
		precalculation_xy->d_y = d_memory_on_gpu_p + offset_to_y_precalculation_data;


		//old_size - 1, the last source pixel is (old_size - 1)
		const auto pixel_weight_increment_x = (1.0F / static_cast<float>(new_width)) * static_cast<float>(old_width - 1);
		const auto pixel_weight_increment_y = (1.0F / static_cast<float>(new_height)) * static_cast<float>(old_height - 1);

		dim3 blockSize_x(32);
		dim3 blockSize_y(32);

		auto bx = (new_width + blockSize_x.x - 1) / blockSize_x.x;
		auto by = (new_height + blockSize_y.x - 1) / blockSize_y.x;

		auto gridSize_x = dim3(bx);
		auto gridSize_y = dim3(by);

		pixel_precalculation_kernel << < gridSize_x, blockSize_x >> > (precalculation_xy->d_x, pixel_weight_increment_x, new_width);  // NOLINT(clang-diagnostic-unused-value)
		pixel_precalculation_kernel << < gridSize_y, blockSize_y >> > (precalculation_xy->d_y, pixel_weight_increment_y, new_height); // NOLINT(clang-diagnostic-unused-value)

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

		hd_params.d_x_precalculation_p = d_x_pixel_precalculation_ptr;
		hd_params.d_y_precalculation_p = d_y_pixel_precalculation_ptr;
		hd_params.d_dimensions_info_p = reinterpret_cast<dimensions_info*>(allocated_memory_on_gpu_p);
		hd_params.d_source_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes);
		hd_params.image_to_scale_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes);
		hd_sobel_params.source_bytes_sequential_p = hd_params.image_to_scale_bytes_sequential_p;
		hd_sobel_params.result_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes + image_to_scale_size_in_bytes);

		
		//pp_array that contains the source image needs to be flattened for fast memory allocation on gpu
		png_bytep png_source_bytes_p = png_util_create_flat_bytes_p_from_row_pp(source_image->png_rows, source_image->image_info.width, source_image->image_info.height, source_png_size_in_bytes);

		cudaMemcpy(hd_params.d_source_bytes_sequential_p, png_source_bytes_p, source_png_size_in_bytes, cudaMemcpyHostToDevice);
		
		dimensions_info dimensions_inf;
		dimensions_inf.image_to_scale_width = image_to_scale->image_info.width;
		dimensions_inf.image_to_scale_height = image_to_scale->image_info.height;
		dimensions_inf.source_image_width = source_image->image_info.width;

		cudaMemcpy(hd_params.d_dimensions_info_p, &dimensions_inf, sizeof(dimensions_info), cudaMemcpyHostToDevice);

		hd_sobel_params.dimensions_inf_p = hd_params.d_dimensions_info_p;

		dim3 blockSize(32, 32);

		auto bx = (image_to_scale->image_info.width + blockSize.x - 1) / blockSize.x;

		auto by = (image_to_scale->image_info.height + blockSize.y - 1) / blockSize.y;

		auto gridSize = dim3(bx, by);

		parallel_tasks_bilinear_nn_sobel << <gridSize, blockSize >> > (hd_params, hd_sobel_params);

		cudaMemcpy(image_to_scale->png_rows[0], hd_sobel_params.result_bytes_sequential_p, image_to_scale_size_in_bytes, cudaMemcpyDeviceToHost);

		cudaFree(allocated_memory_on_gpu_p);
	}

}
