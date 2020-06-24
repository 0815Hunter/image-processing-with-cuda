#include "cuda_kernel_sobel.cuh"

__device__ void sobel(d_sobel_params params, unsigned xy_pos);

__global__ void apply_sobel_filter(d_sobel_params params, unsigned y_offset, unsigned y_max)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = (blockIdx.y * blockDim.y + threadIdx.y) + y_offset;

	if (x > params.dimensions_inf_p->result_image_width - 1 || y > y_max)
	{
		return;
	}

	auto xy_pos = y * params.dimensions_inf_p->result_image_width + x;


	if (x == 0 || y == 0 || x == params.dimensions_inf_p->result_image_width - 1 || y == params.dimensions_inf_p->result_image_height - 1)
	{
		params.result_bytes_sequential_p[xy_pos] = 255;
		return;
	}
	
	sobel(params, xy_pos);
}

namespace sobel_framed_blocks
{
	__global__ void apply_sobel_filter(d_sobel_params params)
	{
		auto x = blockIdx.x * blockDim.x + threadIdx.x;
		auto y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x > params.dimensions_inf_p->result_image_width - 1 || y > params.dimensions_inf_p->result_image_height - 1)
		{
			return;
		}

		auto xy_pos = y * params.dimensions_inf_p->result_image_width + x;


		if (x == 0 || y == 0 || x == params.dimensions_inf_p->result_image_width - 1 || y == params.dimensions_inf_p->result_image_height - 1)
		{
			params.result_bytes_sequential_p[xy_pos] = 255;
			return;
		}

		sobel(params, xy_pos);
	}
}


__device__ void sobel(d_sobel_params params, unsigned xy_pos)
{
	int sobel_operator_x[3][3] = { {-1, 0, 1}, {-2, 0, 2 }, {-1, 0, 1} };
	int sobel_operator_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };


	auto p_01 = xy_pos - params.dimensions_inf_p->result_image_width;
	auto p_00 = p_01 - 1;
	auto p_02 = p_01 + 1;


	auto p_10 = xy_pos - 1;
	auto p_12 = xy_pos + 1;

	auto p_21 = xy_pos + params.dimensions_inf_p->result_image_width;
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

	double xy_sobel_value_double = sqrt(static_cast<double>(x_sobel_value) * x_sobel_value + static_cast<double>(y_sobel_value) * y_sobel_value);

	auto xy_sobel_value = static_cast<png_byte>(lrint((xy_sobel_value_double / 1443) * 255));

	params.result_bytes_sequential_p[xy_pos] = xy_sobel_value;
}


namespace sobel_cooperative_groups_tile2
{
	using namespace cooperative_groups;
	
	__device__ void sobel(d_sobel_params params, unsigned xy_pos, thread_block_tile<2> tile2);
	
	__global__ void apply_sobel_filter(d_sobel_params params)
	{
		//two threads for each sobel calculation
		thread_block_tile<2> tile2 = tiled_partition<2>(this_thread_block());
		
		auto x = (blockIdx.x * blockDim.x + threadIdx.x) / tile2.size();
		auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

		
		if (x > params.dimensions_inf_p->result_image_width - 1 || y > params.dimensions_inf_p->result_image_height - 1)
		{
			return;
		}
		
		auto xy_pos = y * params.dimensions_inf_p->result_image_width + x;

		
		if ( x == 0 || y == 0  || x == params.dimensions_inf_p->result_image_width - 1 || y == params.dimensions_inf_p->result_image_height - 1)
		{
			params.result_bytes_sequential_p[xy_pos] = 255;
			return;
		}

		sobel(params, xy_pos, tile2);
	}

	__device__ void sobel(d_sobel_params params, unsigned xy_pos, thread_block_tile<2> tile2)
	{
		int sobel_operator_x[3][3] = { {-1, 0, 1}, {-2, 0, 2 }, {-1, 0, 1} };
		int sobel_operator_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };


		auto p_01 = xy_pos - params.dimensions_inf_p->result_image_width;
		auto p_00 = p_01 - 1;
		auto p_02 = p_01 + 1;


		auto p_10 = xy_pos - 1;
		auto p_12 = xy_pos + 1;

		auto p_21 = xy_pos + params.dimensions_inf_p->result_image_width;
		auto p_20 = p_21 - 1;
		auto p_22 = p_21 + 1;

		double x_sobel_value_squared = 0;
		double y_sobel_value_squared = 0;
		
		if(tile2.thread_rank() == 0)
		{
			double x_sobel_value =
				params.source_bytes_sequential_p[p_00] * sobel_operator_x[0][0]
				+ params.source_bytes_sequential_p[p_02] * sobel_operator_x[0][2]
				+ params.source_bytes_sequential_p[p_10] * sobel_operator_x[1][0]
				+ params.source_bytes_sequential_p[p_12] * sobel_operator_x[1][2]
				+ params.source_bytes_sequential_p[p_20] * sobel_operator_x[2][0]
				+ params.source_bytes_sequential_p[p_22] * sobel_operator_x[2][2];
			x_sobel_value_squared = x_sobel_value * x_sobel_value;
		}else
		{
			double y_sobel_value =
				params.source_bytes_sequential_p[p_00] * sobel_operator_y[0][0]
				+ params.source_bytes_sequential_p[p_01] * sobel_operator_y[0][1]
				+ params.source_bytes_sequential_p[p_02] * sobel_operator_y[0][2]
				+ params.source_bytes_sequential_p[p_20] * sobel_operator_y[2][0]
				+ params.source_bytes_sequential_p[p_21] * sobel_operator_y[2][1]
				+ params.source_bytes_sequential_p[p_22] * sobel_operator_y[2][2];
			y_sobel_value_squared = y_sobel_value * y_sobel_value;
		}

		tile2.sync();

		y_sobel_value_squared = tile2.shfl(y_sobel_value_squared, 1);

		tile2.sync();

		if(tile2.thread_rank() == 0)
		{
			double xy_sobel_value_double = sqrt(x_sobel_value_squared + y_sobel_value_squared);

			auto xy_sobel_value = static_cast<png_byte>(lrint((xy_sobel_value_double / 1443) * 255));

			params.result_bytes_sequential_p[xy_pos] = xy_sobel_value;
		}
	}
}


namespace sobel_cooperative_groups_tile16_8
{
	using namespace cooperative_groups;


	typedef struct sobel_thread_mapping_def
	{
		unsigned p_xx;
		unsigned y;
		unsigned x;
	}sobel_thread_mapping;

	__device__ void sobel(d_sobel_params params, unsigned xy_pos, thread_block_tile<16> tile16);
	__device__ double reduce_sum(double sum, thread_block_tile<8> tile8);

	__global__ void apply_sobel_filter(d_sobel_params params)
	{
		//two threads for each sobel calculation
		thread_block_tile<16> tile16 = tiled_partition<16>(this_thread_block());

		auto x = (blockIdx.x * blockDim.x + threadIdx.x) / tile16.size();
		auto y = (blockIdx.y * blockDim.y) + threadIdx.y;


		if (x > params.dimensions_inf_p->result_image_width - 1 || y > params.dimensions_inf_p->result_image_height - 1)
		{
			return;
		}

		auto xy_pos = y * params.dimensions_inf_p->result_image_width + x;

		if (x == 0 || y == 0 || x == params.dimensions_inf_p->result_image_width - 1 || y == params.dimensions_inf_p->result_image_height - 1)
		{
			return;
		}

		sobel(params, xy_pos, tile16);
	}

	__device__ void sobel(d_sobel_params params, unsigned xy_pos, thread_block_tile<16> tile16)
	{
		unsigned u_dummy = 0;
		int i_dummy = 0;
		
		int sobel_operator_x[4][4] = { {-1, 0, 1, i_dummy}, {-2, 0, 2, i_dummy }, {-1, 0, 1, i_dummy} };
		int sobel_operator_y[4][4] = { {-1, -2, -1, i_dummy}, {0, 0, 0, i_dummy}, {1, 2, 1, i_dummy} };
		

		auto p_01 = xy_pos - params.dimensions_inf_p->result_image_width;
		auto p_00 = p_01 - 1;
		auto p_02 = p_01 + 1;


		auto p_10 = xy_pos - 1;
		auto p_12 = xy_pos + 1;

		auto p_21 = xy_pos + params.dimensions_inf_p->result_image_width;
		auto p_20 = p_21 - 1;
		auto p_22 = p_21 + 1;

		
		double x_sobel_value_squared = 0;
		double y_sobel_value_squared = 0;
		
		thread_block_tile<8> tile8 = tiled_partition<8>(tile16);
		
		auto tile16_rank = tile16.thread_rank();
		auto tile8_rank = tile8.thread_rank();
		
		if(tile16_rank < 8)
		{
			
			sobel_thread_mapping x_map[8] = { {p_00, 0,0}, {p_02, 0 , 2}, {p_10, 1,0}, {p_12,1,2} , {p_20,2,0} , {p_22,2,2} , {u_dummy ,3,3} , {u_dummy,3,3} };
			
			auto pixel_index = x_map[tile8_rank].p_xx;
			auto sobel_y = x_map[tile8_rank].y;
			auto sobel_x = x_map[tile8_rank].x;
			
			double x_sum = params.source_bytes_sequential_p[pixel_index] * sobel_operator_x[sobel_y][sobel_x];

			tile8.sync();
			
			double x_sobel_value = reduce_sum(x_sum, tile8);

			x_sobel_value_squared = x_sobel_value * x_sobel_value;
		}
		else
		{
			sobel_thread_mapping y_map[8] = { {p_00,0,0}, {p_01,0,1}, {p_02,0,2}, {p_20,2,0} , {p_21,2,1} , {p_22,2,2} , {u_dummy, 3,3} , {u_dummy, 3, 3} };

			auto pixel_index = y_map[tile8_rank].p_xx;
			auto sobel_x = y_map[tile8_rank].x;
			auto sobel_y = y_map[tile8_rank].y;

			double y_sum = params.source_bytes_sequential_p[pixel_index] * sobel_operator_y[sobel_y][sobel_x];

			tile8.sync();

			double y_sobel_value = reduce_sum(y_sum, tile8);

			y_sobel_value_squared = y_sobel_value * y_sobel_value;
		}

		tile16.sync();
		y_sobel_value_squared = tile16.shfl(y_sobel_value_squared, 8);
		x_sobel_value_squared = tile16.shfl(x_sobel_value_squared, 0);
		tile16.sync();

		if (tile16.thread_rank() == 0)
		{
			double xy_sobel_value_double = sqrt(x_sobel_value_squared + y_sobel_value_squared);

			auto xy_sobel_value = static_cast<png_byte>(lrint((xy_sobel_value_double / 1443) * 255));

			params.result_bytes_sequential_p[xy_pos] = xy_sobel_value;
		}
	}

	__device__ double reduce_sum(double sum, thread_block_tile<8> tile8)
	{
		for (int i = tile8.size() / 2; i > 0; i /= 2)
		{ 
			sum += tile8.shfl_down(sum, i);
		}
		
		return sum;  // only thread with rank 0 has full sum
	}
}
