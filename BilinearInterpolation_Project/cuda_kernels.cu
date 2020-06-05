#include "cuda_kernels.cuh"

#include "cuda_kernel_sobel.cuh"
#include "cuda_bilinear.cuh"
#include "cuda_nearest_neighbor.cuh"
__global__ void pixel_precalculation_kernel(pixel_precalculation* precalculation, const double pixel_weight_increment, unsigned int N)
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= N))
	{
		return;
	}

	auto current_pixel_weight = pixel_weight_increment * i;
	
	auto source_image_front_pixel = static_cast<unsigned int>(current_pixel_weight);
	
	precalculation[i].front_pixel = source_image_front_pixel;
	precalculation[i].rear_pixel = source_image_front_pixel + 1;

	auto weight = current_pixel_weight - static_cast<double>(source_image_front_pixel);

	precalculation[i].front_weight = 1 - weight;
	precalculation[i].rear_weight = weight;
}

__global__ void bilinear_nn(d_scale_params f_params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= f_params.dimensions_info_p->result_image_width) || (y >= f_params.dimensions_info_p->result_image_height))
	{
		return;
	}

	if (y < f_params.dimensions_info_p->result_image_height / 2)
	{
		device::apply_bilinear_filter(f_params, x, y);
	}
	else
	{
		device::apply_nearest_neighbor(f_params, x, y);
	}
}

__device__ unsigned int block_counter = 0;

__global__ void bilinear_nn_sobel(d_scale_params f_params, d_sobel_params s_params)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= f_params.dimensions_info_p->result_image_width) || (y >= f_params.dimensions_info_p->result_image_height))
	{
		return;
	}

	if (y < f_params.dimensions_info_p->result_image_height / 2)
	{
		device::apply_bilinear_filter(f_params, x, y);
	}
	else
	{
		device::apply_nearest_neighbor(f_params, x, y);
	}

	
	__threadfence(); // make sure the data processed by the thread is written to global memory
	__syncthreads(); // every thread needs to be done before we can report that the block is done

	
	//fist thread of every block reports that the block is done
	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		auto grid_count = (gridDim.x * gridDim.y);
		unsigned int done_grids = atomicInc(&block_counter, grid_count);

		if(done_grids == grid_count - 1) // last done block starts the child kernel
		{
			dim3 block_size(32, 32);

			auto bx = (f_params.dimensions_info_p->result_image_width + block_size.x - 1) / block_size.x;

			auto by = (f_params.dimensions_info_p->result_image_height + block_size.y - 1) / block_size.y;

			auto grid_size = dim3(bx, by);
			apply_sobel_filter<<<grid_size, block_size>>>(s_params, 0, f_params.dimensions_info_p->result_image_height);
		}
		
		// if (done_grids == grid_count - 1) // last done block starts the child kernel
		// {
		// 	dim3 block_size(32, 32);
		//
		// 	auto bx = (f_params.dimensions_info_p->result_image_width * 16 + block_size.x - 1) / block_size.x;
		//
		// 	auto by = (f_params.dimensions_info_p->result_image_height + block_size.y - 1) / block_size.y;
		//
		// 	auto grid_size = dim3(bx, by);
		// 	sobel_cooperative_groups_tile16_8::apply_sobel_filter << <grid_size, block_size >> > (s_params);
		// }
		//
		// if (done_grids == grid_count - 1) // last done block starts the child kernel
		// {
		// 	dim3 block_size(32, 32);
		//
		// 	auto bx = (f_params.dimensions_info_p->result_image_width * 2 + block_size.x - 1) / block_size.x;
		//
		// 	auto by = (f_params.dimensions_info_p->result_image_height + block_size.y - 1) / block_size.y;
		//
		// 	auto grid_size = dim3(bx, by);
		// 	sobel_cooperative_groups_tile2::apply_sobel_filter << <grid_size, block_size >> > (s_params);
		// }
	}
}
