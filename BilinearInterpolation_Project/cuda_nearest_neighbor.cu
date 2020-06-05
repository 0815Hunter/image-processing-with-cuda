#include "cuda_nearest_neighbor.cuh"

namespace device
{
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

		auto index_to_output_pixel = y * params.dimensions_info_p->result_image_width + x;

		params.result_image_bytes_sequential_p[index_to_output_pixel] = pixel_p_value;
	}
}

namespace global
{
	__global__ void apply_nearest_neighbor(d_scale_params params, unsigned y_offset, unsigned y_max)
	{
		auto x = blockIdx.x * blockDim.x + threadIdx.x;
		auto y = (blockIdx.y * blockDim.y + threadIdx.y) + y_offset;

		if ((x >= params.dimensions_info_p->result_image_width) || (y >= y_max))
		{
			return;
		}
		
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

		auto index_to_output_pixel = y * params.dimensions_info_p->result_image_width + x;

		params.result_image_bytes_sequential_p[index_to_output_pixel] = pixel_p_value;
	}
}
