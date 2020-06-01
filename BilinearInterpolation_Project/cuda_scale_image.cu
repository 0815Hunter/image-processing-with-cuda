#include <cstdlib>

#include "cuda_scale_image.cuh"

#include "cuda_includes.h"
#include "common_structs.cuh"
#include "cuda_kernels.cuh"
#include "cuda_scale_pixel_precalculation.cuh"
namespace cuda_seq
{

	void scale_and_apply_sobel(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
		pixel_precalculation* d_y_pixel_precalculation_ptr);

	void scale_image_apply_sobel(png_user_struct* source_image, png_user_struct* result_image)
	{
		const auto old_height = source_image->image_info.height;
		const auto new_height = result_image->image_info.height;
		const auto old_width = source_image->image_info.width;
		const auto new_width = result_image->image_info.width;

		pixel_precalculation_memory precalculation_xy;

		create_pixel_precalculation(&precalculation_xy, old_width, new_width, old_height, new_height);

		scale_and_apply_sobel(result_image, source_image, precalculation_xy.d_x, precalculation_xy.d_y);

		cudaFree(precalculation_xy.allocated_gpu_memory);
	}

	void scale_and_apply_sobel(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
		pixel_precalculation* d_y_pixel_precalculation_ptr)
	{
		auto dimensions_size_in_bytes = sizeof(dimensions_info);
		auto source_png_size_in_bytes = sizeof(png_byte) * source_image->image_info.width * source_image->image_info.height;
		auto result_image_size_in_bytes = sizeof(png_byte) * result_image->image_info.width * result_image->image_info.height;
		auto sobel_image_size_in_bytes = result_image_size_in_bytes;

		auto needed_memory_in_bytes = dimensions_size_in_bytes + source_png_size_in_bytes + result_image_size_in_bytes + sobel_image_size_in_bytes;

		_int8* allocated_memory_on_gpu_p;

		cudaMalloc(reinterpret_cast<void**>(&allocated_memory_on_gpu_p), needed_memory_in_bytes);

		d_scale_params d_params;
		d_sobel_params d_sobel_params;

		d_params.x_precalculation_p = d_x_pixel_precalculation_ptr;
		d_params.y_precalculation_p = d_y_pixel_precalculation_ptr;
		d_params.dimensions_info_p = reinterpret_cast<dimensions_info*>(allocated_memory_on_gpu_p);
		d_params.source_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes);
		d_params.result_image_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes);
		d_sobel_params.source_bytes_sequential_p = d_params.result_image_bytes_sequential_p;
		d_sobel_params.result_bytes_sequential_p = reinterpret_cast<png_bytep>(allocated_memory_on_gpu_p + dimensions_size_in_bytes + source_png_size_in_bytes + result_image_size_in_bytes);

		
		//pp_array that contains the source image needs to be flattened for fast memory allocation on gpu
		png_bytep png_source_bytes_p = png_util_create_flat_bytes_p_from_row_pp(source_image->png_rows, source_image->image_info.width, source_image->image_info.height, source_png_size_in_bytes);

		cudaMemcpy(d_params.source_bytes_sequential_p, png_source_bytes_p, source_png_size_in_bytes, cudaMemcpyHostToDevice);
		
		dimensions_info dimensions_inf;
		dimensions_inf.result_image_width = result_image->image_info.width;
		dimensions_inf.result_image_height = result_image->image_info.height;
		dimensions_inf.source_image_width = source_image->image_info.width;

		cudaMemcpy(d_params.dimensions_info_p, &dimensions_inf, sizeof(dimensions_info), cudaMemcpyHostToDevice);

		d_sobel_params.dimensions_inf_p = d_params.dimensions_info_p;

		dim3 blockSize(32, 32);

		auto bx = (result_image->image_info.width + blockSize.x - 1) / blockSize.x;

		auto by = (result_image->image_info.height + blockSize.y - 1) / blockSize.y;

		auto gridSize = dim3(bx, by);

		parallel_tasks_bilinear_nn_sobel <<<gridSize, blockSize >>> (d_params, d_sobel_params);

		cudaMemcpy(result_image->png_rows[0], d_params.result_image_bytes_sequential_p, result_image_size_in_bytes, cudaMemcpyDeviceToHost);

		cudaFree(allocated_memory_on_gpu_p);
	}

}
