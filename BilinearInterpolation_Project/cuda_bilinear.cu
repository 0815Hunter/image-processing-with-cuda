#include "cuda_bilinear.cuh"

namespace device
{
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

		double point_00_01_value = static_cast<double>(source_bytes[p_00_index]) * x_front_weight + static_cast<double>(source_bytes[p_01_index]) * x_rear_weight;
		double point_10_11_value = static_cast<double>(source_bytes[p_10_index]) * x_front_weight + static_cast<double>(source_bytes[p_11_index]) * x_rear_weight;


		png_byte pixel_p_value = static_cast<png_byte>(lrint(point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight));

		auto i = y * params.dimensions_info_p->result_image_width + x;

		params.result_image_bytes_sequential_p[i] = pixel_p_value;

	}
}


namespace global
{
	__global__ void apply_bilinear_filter(d_scale_params params, unsigned y_offset, unsigned y_max)
	{
		auto x = blockIdx.x * blockDim.x + threadIdx.x;
		auto y = (blockIdx.y * blockDim.y + threadIdx.y) + y_offset;

		if ((x >= params.dimensions_info_p->result_image_width) || (y >= y_max))
		{
			return;
		}
		
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

		double point_00_01_value = static_cast<double>(source_bytes[p_00_index]) * x_front_weight + static_cast<double>(source_bytes[p_01_index]) * x_rear_weight;
		double point_10_11_value = static_cast<double>(source_bytes[p_10_index]) * x_front_weight + static_cast<double>(source_bytes[p_11_index]) * x_rear_weight;


		png_byte pixel_p_value = static_cast<png_byte>(lrint(point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight));

		auto i = y * params.dimensions_info_p->result_image_width + x;

		params.result_image_bytes_sequential_p[i] = pixel_p_value;

	}
}
