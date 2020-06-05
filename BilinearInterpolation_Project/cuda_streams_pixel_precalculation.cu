#include "cuda_streams_pixel_precalculation.cuh"
#include "cuda_includes.h"
#include "cuda_kernels.cuh"

namespace cuda_streams_example
{

	void init_device_memory(pixel_precalculation_memory* xy_precalculation, unsigned long long needed_memory_in_bytes_x, unsigned long long needed_memory_in_bytes_y);
	void init_pinned_host_memory(pixel_precalculation_memory* xy_precalculation, unsigned long long needed_memory_in_bytes_x, unsigned long long needed_memory_in_bytes_y);
	
	void create_pixel_precalculation(pixel_precalculation_memory* xy_precalculation,
		unsigned old_height, unsigned new_height, unsigned old_width, unsigned new_width)
	{
		const auto x_pixel_weight_increment = (1.0 / static_cast<double>(new_width)) * static_cast<double>(old_width - 1);
		const auto y_pixel_weight_increment = (1.0 / static_cast<double>(new_height)) * static_cast<double>(old_height - 1);

		auto needed_memory_in_bytes_x = sizeof(pixel_precalculation) * new_width;
		auto needed_memory_in_bytes_y = sizeof(pixel_precalculation) * new_height;


		init_device_memory(xy_precalculation, needed_memory_in_bytes_x, needed_memory_in_bytes_y);

		init_pinned_host_memory(xy_precalculation, needed_memory_in_bytes_x, needed_memory_in_bytes_y);

		dim3 block_size_x(32);
		dim3 block_size_y(32);

		auto bx = (new_width + block_size_x.x - 1) / block_size_x.x;
		auto by = (new_height + block_size_y.x - 1) / block_size_y.x;

		auto grid_size_x = dim3(bx);
		auto grid_size_y = dim3(by);


		cudaStream_t* x_precalculation_stream_p = &xy_precalculation->precalculation_stream[0];
		cudaStream_t* y_precalculation_stream_p = &xy_precalculation->precalculation_stream[1];

		cudaStreamCreate(x_precalculation_stream_p);
		cudaStreamCreate(y_precalculation_stream_p);

		pixel_precalculation_kernel << < grid_size_x, block_size_x, 0, *x_precalculation_stream_p >> > (xy_precalculation->d_x, x_pixel_weight_increment, new_width);  // NOLINT(clang-diagnostic-unused-value)
		pixel_precalculation_kernel << < grid_size_y, block_size_y, 0, *y_precalculation_stream_p >> > (xy_precalculation->d_y, y_pixel_weight_increment, new_height); // NOLINT(clang-diagnostic-unused-value)

		//cudaMemcpyAsync(xy_precalculation->h_x_pinned, xy_precalculation->d_x, needed_memory_in_bytes_x, cudaMemcpyDeviceToHost, *x_precalculation_stream_p);
		//cudaStreamSynchronize(*x_precalculation_stream_p);
		//precalculation_xy->h_x_pinned now available

	}
	void init_device_memory(pixel_precalculation_memory* xy_precalculation, unsigned long long needed_memory_in_bytes_x, unsigned long long needed_memory_in_bytes_y)
	{
		pixel_precalculation* d_x_precalculation_ptr;
		pixel_precalculation* d_y_precalculation_ptr;

		cudaMalloc(reinterpret_cast<void**>(&d_x_precalculation_ptr), needed_memory_in_bytes_x);
		cudaMalloc(reinterpret_cast<void**>(&d_y_precalculation_ptr), needed_memory_in_bytes_y);

		xy_precalculation->d_x = d_x_precalculation_ptr;
		xy_precalculation->d_y = d_y_precalculation_ptr;
	}

	void init_pinned_host_memory(pixel_precalculation_memory* xy_precalculation, unsigned long long needed_memory_in_bytes_x, unsigned long long needed_memory_in_bytes_y)
	{
		pixel_precalculation* h_x_precalculation_ptr;
		pixel_precalculation* h_y_precalculation_ptr;

		cudaMallocHost(reinterpret_cast<void**>(&h_x_precalculation_ptr), needed_memory_in_bytes_x);
		cudaMallocHost(reinterpret_cast<void**>(&h_y_precalculation_ptr), needed_memory_in_bytes_y);

		xy_precalculation->h_x_pinned = h_x_precalculation_ptr;
		xy_precalculation->h_y_pinned = h_y_precalculation_ptr;
	}
}
