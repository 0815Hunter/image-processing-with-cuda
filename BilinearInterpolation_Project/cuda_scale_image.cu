#include <cstdlib>

#include "cuda_scale_image.cuh"

#include "cuda_includes.h"
#include "common_structs.cuh"
#include "cuda_bilinear.cuh"
#include "cuda_kernels.cuh"
#include "cuda_kernel_sobel.cuh"
#include "cuda_nearest_neighbor.cuh"
#include "cuda_scale_pixel_precalculation.cuh"
namespace cuda_seq
{

	void scale_and_apply_sobel(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
	                           pixel_precalculation* d_y_pixel_precalculation_ptr, kernel_mode kernel_mode);

	void scale_image_apply_sobel(png_user_struct* source_image, png_user_struct* result_image, kernel_mode kernel_mode)
	{
		const auto old_height = source_image->image_info.height;
		const auto new_height = result_image->image_info.height;
		const auto old_width = source_image->image_info.width;
		const auto new_width = result_image->image_info.width;

		pixel_precalculation_memory precalculation_xy;

		create_pixel_precalculation(&precalculation_xy, old_width, new_width, old_height, new_height);

		scale_and_apply_sobel(result_image, source_image, precalculation_xy.d_x, precalculation_xy.d_y, kernel_mode);

		cudaFree(precalculation_xy.allocated_gpu_memory);
	}

	void scale_and_apply_sobel(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* d_x_pixel_precalculation_ptr,
	                           pixel_precalculation* d_y_pixel_precalculation_ptr, kernel_mode kernel_mode)
	{
		auto source_width = source_image->image_info.width;
		auto source_height = source_image->image_info.height;
		auto new_height = result_image->image_info.height;
		auto new_width = result_image->image_info.width;
		
		auto dimensions_size_in_bytes = sizeof(dimensions_info);
		auto source_png_size_in_bytes = sizeof(png_byte) * source_width * source_height;
		
		auto result_image_size_in_bytes = sizeof(png_byte) * new_width * new_height;
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
		png_bytep png_source_bytes_p = png_util_create_flat_bytes_p_from_row_pp(source_image->png_rows, source_width, source_height, source_png_size_in_bytes);

		cudaMemcpy(d_scale_params.source_bytes_sequential_p, png_source_bytes_p, source_png_size_in_bytes, cudaMemcpyHostToDevice);
		
		dimensions_info dimensions_inf;
		dimensions_inf.result_image_width = new_width;
		dimensions_inf.result_image_height = new_height;
		dimensions_inf.source_image_width = source_width;

		cudaMemcpy(d_scale_params.dimensions_info_p, &dimensions_inf, sizeof(dimensions_info), cudaMemcpyHostToDevice);

		d_sobel_params.dimensions_inf_p = d_scale_params.dimensions_info_p;

		dim3 block_size(32, 32);

		unsigned blocks_in_x_direction;
		unsigned blocks_in_y_direction;

		dim3 grid_size;

		unsigned y_offset = new_height / 2;

		switch (kernel_mode) {
			case kernel_mode::bilinear_nn:

				blocks_in_x_direction = (new_width + block_size.x - 1) / block_size.x;
				blocks_in_y_direction = ((new_height / 2) + block_size.y - 1) / block_size.y;

				grid_size = dim3(blocks_in_x_direction, blocks_in_y_direction);

				global::apply_bilinear_filter << <grid_size, block_size>> > (d_scale_params, 0, y_offset);
				global::apply_nearest_neighbor << <grid_size, block_size>> > (d_scale_params, y_offset, new_height);

				cudaMemcpy(result_image->png_rows[0], d_scale_params.result_image_bytes_sequential_p, result_image_size_in_bytes, cudaMemcpyDeviceToHost);

				break;
			case kernel_mode::bilinear_nn_sobel:

				blocks_in_x_direction = (new_width + block_size.x - 1) / block_size.x;
				blocks_in_y_direction = ((new_height / 2) + block_size.y - 1) / block_size.y;

				grid_size = dim3(blocks_in_x_direction, blocks_in_y_direction);

				global::apply_bilinear_filter << <grid_size, block_size>> > (d_scale_params, 0, y_offset);
				global::apply_nearest_neighbor << <grid_size, block_size>> > (d_scale_params, y_offset, new_height);
				
				blocks_in_y_direction = (new_height + block_size.y - 1) / block_size.y;
				grid_size = dim3(blocks_in_x_direction, blocks_in_y_direction);
				
				apply_sobel_filter << <grid_size, block_size >> > (d_sobel_params, 0, new_height);

				// blocks_in_x_direction = (new_width * 16 + block_size.x - 1) / block_size.x;
				// blocks_in_y_direction = (new_height + block_size.y - 1) / block_size.y;
				//
				// grid_size = dim3(blocks_in_x_direction, blocks_in_y_direction);
				//
				// sobel_cooperative_groups_tile16_8::apply_sobel_filter << <grid_size, block_size >> > (d_sobel_params);

				cudaMemcpy(result_image->png_rows[0], d_sobel_params.result_bytes_sequential_p, result_image_size_in_bytes, cudaMemcpyDeviceToHost);

				break;
			case kernel_mode::branch_bilinear_nn_dynamic_sobel:

				blocks_in_x_direction = (new_width + block_size.x - 1) / block_size.x;
				blocks_in_y_direction = (new_height + block_size.y - 1) / block_size.y;

				grid_size = dim3(blocks_in_x_direction, blocks_in_y_direction);

				bilinear_nn_sobel << <grid_size, block_size>> > (d_scale_params, d_sobel_params);

				cudaMemcpy(result_image->png_rows[0], d_sobel_params.result_bytes_sequential_p, result_image_size_in_bytes, cudaMemcpyDeviceToHost);

				break;
		default:;
		}

		cudaFree(allocated_memory_on_gpu_p);
	}

}
