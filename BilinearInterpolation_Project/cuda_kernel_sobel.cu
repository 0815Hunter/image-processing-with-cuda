#include "cuda_kernel_sobel.cuh"

__global__ void apply_sobel_filter(d_sobel_params params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= params.dimensions_inf_p->image_to_scale_width) || (y >= params.dimensions_inf_p->image_to_scale_height) || x == 0 || x == (params.dimensions_inf_p->image_to_scale_width - 1) || y == 0 || y == (params.dimensions_inf_p->image_to_scale_height - 1))
	{
		return;
	}

	int sobel_operator_x[3][3] = { {-1, 0, 1}, {-2, 0, 2 }, {-1, 0, 1} };
	int sobel_operator_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };


	auto xy_pos = y * params.dimensions_inf_p->image_to_scale_width + x;

	auto p_01 = xy_pos - params.dimensions_inf_p->image_to_scale_width;
	auto p_00 = p_01 - 1;
	auto p_02 = p_01 + 1;


	auto p_10 = xy_pos - 1;
	auto p_11 = xy_pos;
	auto p_12 = xy_pos + 1;

	auto p_21 = xy_pos + params.dimensions_inf_p->image_to_scale_width;
	auto p_20 = p_21 - 1;
	auto p_22 = p_21 + 1;

	auto x_sobel_value =
		params.source_bytes_sequential_p[p_00] * sobel_operator_x[0][0]
		+ params.source_bytes_sequential_p[p_02] * sobel_operator_x[0][2]
		+ params.source_bytes_sequential_p[p_10] * sobel_operator_x[1][0]
		+ params.source_bytes_sequential_p[p_12] * sobel_operator_x[1][2]
		+ params.source_bytes_sequential_p[p_20] * sobel_operator_x[2][0]
		+ params.source_bytes_sequential_p[p_22] * sobel_operator_x[2][2];

	auto y_sobel_value =
		params.source_bytes_sequential_p[p_00] * sobel_operator_y[0][0]
		+ params.source_bytes_sequential_p[p_01] * sobel_operator_y[0][1]
		+ params.source_bytes_sequential_p[p_02] * sobel_operator_y[0][2]
		+ params.source_bytes_sequential_p[p_20] * sobel_operator_y[2][0]
		+ params.source_bytes_sequential_p[p_21] * sobel_operator_y[2][1]
		+ params.source_bytes_sequential_p[p_22] * sobel_operator_y[2][2];

	auto xy_sobel_value = sqrt(static_cast<double>((x_sobel_value * x_sobel_value) + (y_sobel_value * y_sobel_value)));

	xy_sobel_value = __float2uint_rn((xy_sobel_value / 1443) * 255);


	params.result_bytes_sequential_p[xy_pos] = xy_sobel_value;
}


__global__ void sobel_filter_y_child(d_scale_params params);

__global__ void sobel_filter_y_child(d_scale_params params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	int sobel_operator_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	auto xy_pos = y * params.dimensions_info_p->image_to_scale_width + x;

	auto p_01 = xy_pos - params.dimensions_info_p->image_to_scale_width;
	auto p_00 = p_01 - 1;
	auto p_02 = p_01 + 1;


	auto p_10 = xy_pos - 1;
	auto p_11 = xy_pos;
	auto p_12 = xy_pos + 1;

	auto p_21 = xy_pos + params.dimensions_info_p->image_to_scale_width;
	auto p_20 = p_21 - 1;
	auto p_22 = p_21 + 1;

	auto y_sobel_value =
		params.source_bytes_sequential_p[p_00] * sobel_operator_y[0][0]
		+ params.source_bytes_sequential_p[p_01] * sobel_operator_y[0][1]
		+ params.source_bytes_sequential_p[p_02] * sobel_operator_y[0][2]
		+ params.source_bytes_sequential_p[p_20] * sobel_operator_y[2][0]
		+ params.source_bytes_sequential_p[p_21] * sobel_operator_y[2][1]
		+ params.source_bytes_sequential_p[p_22] * sobel_operator_y[2][2];

}
