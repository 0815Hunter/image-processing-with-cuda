#include "cuda_kernels.cuh"

#include "cuda_kernel_sobel.cuh"


__device__ void apply_bilinear_filter(d_scale_params params, unsigned x, unsigned y);

__device__ void apply_nearest_neighbor(d_scale_params params, unsigned x, unsigned y);

__global__ void pixel_precalculation_kernel(pixel_precalculation* precalculation, const float pixel_weight_increment, unsigned int N)
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= N))
	{
		return;
	}

	const auto current_pixel_weight = pixel_weight_increment * i;
	const auto source_image_pixel_for_front_pixel_x = static_cast<unsigned int>(current_pixel_weight);
	const auto rear_pixel = source_image_pixel_for_front_pixel_x + 1;


	precalculation[i].front_pixel = source_image_pixel_for_front_pixel_x;
	precalculation[i].rear_pixel = rear_pixel;

	const auto weight = current_pixel_weight - static_cast<float>(source_image_pixel_for_front_pixel_x);

	precalculation[i].front_weight = 1 - weight;
	precalculation[i].rear_weight = weight;
}

__device__ unsigned int block_counter = 0;

__global__ void parallel_tasks_bilinear_nn_sobel(d_scale_params f_params, d_sobel_params s_params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= f_params.dimensions_info_p->image_to_scale_width) || (y >= f_params.dimensions_info_p->image_to_scale_height))
	{
		return;
	}

	if (y < f_params.dimensions_info_p->image_to_scale_height / 2)
	{
		apply_bilinear_filter(f_params, x, y);
	}
	else
	{
		apply_nearest_neighbor(f_params, x, y);
	}
	
	__threadfence();
	__syncthreads();
	
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		auto grid_count = (gridDim.x * gridDim.y);
		unsigned int doneGrids = atomicInc(&block_counter, grid_count);

		if(doneGrids == grid_count - 1)
		{
			dim3 blockSize(32, 32);

			auto bx = (f_params.dimensions_info_p->image_to_scale_width + blockSize.x - 1) / blockSize.x;

			auto by = (f_params.dimensions_info_p->image_to_scale_height + blockSize.y - 1) / blockSize.y;

			auto gridSize = dim3(bx, by);
			apply_sobel_filter<<<gridSize, blockSize>>>(s_params);
		}
	}
}

__global__ void parallel_tasks_bilinear_nn(d_scale_params f_params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= f_params.dimensions_info_p->image_to_scale_width) || (y >= f_params.dimensions_info_p->image_to_scale_height))
	{
		return;
	}

	if (y < f_params.dimensions_info_p->image_to_scale_height / 2)
	{
		apply_bilinear_filter(f_params, x, y);
	}
	else
	{
		apply_nearest_neighbor(f_params, x, y);
	}
}

__device__ void apply_bilinear_filter(d_scale_params params, unsigned x, unsigned y)
{
	auto y_pixel_front = params.y_precalculation_p[y].front_pixel;
	auto y_pixel_rear = params.y_precalculation_p[y].rear_pixel;
	auto y_front_weight = params.y_precalculation_p[y].front_weight;
	auto y_rear_weight = params.y_precalculation_p[y].rear_weight;

	auto x_pixel_front = params.x_precalculation_p[x].front_pixel;
	auto x_pixel_rear = params.x_precalculation_p[x].rear_pixel;
	auto x_front_weight = params.x_precalculation_p[x].front_weight;
	auto x_rear_weight = params.x_precalculation_p[x].rear_weight;

	auto base_upper_row = y_pixel_front * params.dimensions_info_p->source_image_width;
	auto p_00_index = base_upper_row + x_pixel_front;
	auto p_01_index = base_upper_row + x_pixel_rear;

	auto base_lower_row = y_pixel_rear * params.dimensions_info_p->source_image_width;
	auto p_10_index = base_lower_row + x_pixel_front;
	auto p_11_index = base_lower_row + x_pixel_rear;


	auto* source_bytes = params.source_bytes_sequential_p;

	png_byte point_00_01_value = source_bytes[p_00_index] * x_front_weight + source_bytes[p_01_index] * x_rear_weight;
	png_byte point_10_11_value = source_bytes[p_10_index] * x_front_weight + source_bytes[p_11_index] * x_rear_weight;

	png_byte pixel_p_value = point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight;

	auto i = y * params.dimensions_info_p->image_to_scale_width + x;

	params.image_to_scale_bytes_sequential_p[i] = pixel_p_value;

}

__device__ void apply_nearest_neighbor(d_scale_params params, unsigned x, unsigned y)
{
	auto y_pixel_front = params.y_precalculation_p[y].front_pixel;

	auto x_pixel_front = params.x_precalculation_p[x].front_pixel;
	auto x_pixel_rear = params.x_precalculation_p[x].rear_pixel;
	auto x_front_weight = params.x_precalculation_p[x].front_weight;
	auto x_rear_weight = params.x_precalculation_p[x].rear_weight;

	auto base_upper_row = y_pixel_front * params.dimensions_info_p->source_image_width;
	auto p_00_index = base_upper_row + x_pixel_front;
	auto p_01_index = base_upper_row + x_pixel_rear;

	auto* source_bytes = params.source_bytes_sequential_p;

	png_byte pixel_p_value;
	if (x_front_weight > x_rear_weight)
	{
		pixel_p_value = source_bytes[p_00_index];
	}
	else
	{
		pixel_p_value = source_bytes[p_01_index];
	}

	auto index_to_output_pixel = y * params.dimensions_info_p->image_to_scale_width + x;

	params.image_to_scale_bytes_sequential_p[index_to_output_pixel] = pixel_p_value;
}
