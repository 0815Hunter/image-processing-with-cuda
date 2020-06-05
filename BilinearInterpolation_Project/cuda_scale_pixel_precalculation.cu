#include "cuda_scale_pixel_precalculation.cuh"

#include "cuda_kernels.cuh"

namespace cuda_seq
{
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
		const auto pixel_weight_increment_x = (1.0 / static_cast<double>(new_width)) * static_cast<double>(old_width - 1);
		const auto pixel_weight_increment_y = (1.0 / static_cast<double>(new_height)) * static_cast<double>(old_height - 1);

		dim3 blockSize_x(32);
		dim3 blockSize_y(32);

		auto bx = (new_width + blockSize_x.x - 1) / blockSize_x.x;
		auto by = (new_height + blockSize_y.x - 1) / blockSize_y.x;

		auto gridSize_x = dim3(bx);
		auto gridSize_y = dim3(by);

		pixel_precalculation_kernel << < gridSize_x, blockSize_x >> > (precalculation_xy->d_x, pixel_weight_increment_x, new_width);  // NOLINT(clang-diagnostic-unused-value)
		pixel_precalculation_kernel << < gridSize_y, blockSize_y >> > (precalculation_xy->d_y, pixel_weight_increment_y, new_height); // NOLINT(clang-diagnostic-unused-value)

	}
}
