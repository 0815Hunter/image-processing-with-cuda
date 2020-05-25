#include "cuda_kernels.cuh"

__global__ void k_pixel_precalculation(pixel_precalculation* precalculation, const float pixel_weight_increment, unsigned int N)
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= N))
	{
		return;
	}

	const auto current_pixel_weight = pixel_weight_increment * i;
	const auto source_image_pixel_for_front_pixel_x = static_cast<unsigned int>(current_pixel_weight);
	const auto rear_pixel = source_image_pixel_for_front_pixel_x + 1;


	precalculation[i].frontPixel = source_image_pixel_for_front_pixel_x;
	precalculation[i].rearPixel = rear_pixel;

	const auto weight = current_pixel_weight - static_cast<float>(source_image_pixel_for_front_pixel_x);

	precalculation[i].frontWeight = 1 - weight;
	precalculation[i].rearWeight = weight;
}

__global__ void parallel_tasks(fill_params params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= params.dimensions_inf_p->image_to_scale_width) || (y >= params.dimensions_inf_p->image_to_scale_height))
	{
		return;
	}

	if (y < params.dimensions_inf_p->image_to_scale_height / 2)
	{
		apply_bilinear_filter(params, x, y);
	}
	else
	{
		apply_nearest_neighbor(params, x, y);
	}
}

__device__ void apply_bilinear_filter(fill_params params, unsigned x, unsigned y)
{

	auto y_pixel_front = params.y_precalculation[y].frontPixel;
	auto y_pixel_rear = params.y_precalculation[y].rearPixel;
	auto y_front_weight = params.y_precalculation[y].frontWeight;
	auto y_rear_weight = params.y_precalculation[y].rearWeight;

	auto x_pixel_front = params.x_precalculation[x].frontPixel;
	auto x_pixel_rear = params.x_precalculation[x].rearPixel;
	auto x_front_weight = params.x_precalculation[x].frontWeight;
	auto x_rear_weight = params.x_precalculation[x].rearWeight;

	auto base_upper_row = y_pixel_front * params.dimensions_inf_p->source_image_width;
	auto p_00_index = base_upper_row + x_pixel_front;
	auto p_01_index = base_upper_row + x_pixel_rear;

	auto base_lower_row = y_pixel_rear * params.dimensions_inf_p->source_image_width;
	auto p_10_index = base_lower_row + x_pixel_front;
	auto p_11_index = base_lower_row + x_pixel_rear;


	auto* source_bytes = params.source_bytes_sequential_p;

	png_byte point_00_01_value = source_bytes[p_00_index] * x_front_weight + source_bytes[p_01_index] * x_rear_weight;
	png_byte point_10_11_value = source_bytes[p_10_index] * x_front_weight + source_bytes[p_11_index] * x_rear_weight;

	png_byte pixel_p_value = point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight;

	auto i = y * params.dimensions_inf_p->image_to_scale_width + x;

	params.image_to_scale_bytes_sequential_p[i] = pixel_p_value;

}

__device__ void apply_nearest_neighbor(fill_params params, unsigned x, unsigned y)
{
	auto y_pixel_front = params.y_precalculation[y].frontPixel;

	auto x_pixel_front = params.x_precalculation[x].frontPixel;
	auto x_pixel_rear = params.x_precalculation[x].rearPixel;
	auto x_front_weight = params.x_precalculation[x].frontWeight;
	auto x_rear_weight = params.x_precalculation[x].rearWeight;

	auto base_upper_row = y_pixel_front * params.dimensions_inf_p->source_image_width;
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

	auto index_to_output_pixel = y * params.dimensions_inf_p->image_to_scale_width + x;

	params.image_to_scale_bytes_sequential_p[index_to_output_pixel] = pixel_p_value;
}