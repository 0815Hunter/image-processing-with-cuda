#include <cstdlib>


#include "common_structs.cuh"
#include "imageutil.h"

namespace seq
{
	pixel_precalculation* create_pixel_precalculation_for_x_row(unsigned int old_width, unsigned int new_width);

	pixel_precalculation* create_pixel_precalculation_for_y_row(unsigned int old_height, unsigned int new_height);

	pixel_precalculation* create_pixel_precalculation(unsigned int old_pixel_array_size, unsigned int new_pixel_array_size);

	void scale_bilinear_first_half(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr);

	void scale_nearest_neighbor_second_half(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr);

	void apply_sobel(png_user_struct* result_image);

	png_byte calculate_sobel_value(png_user_struct* result_image, png_uint_32 y_1, png_uint_32 x_1);


	void scale_image_apply_sobel(png_user_struct* source_image, png_user_struct* result_image)
	{
		const auto old_height = source_image->image_info.height;
		const auto new_height = result_image->image_info.height;
		const auto old_width = source_image->image_info.width;
		const auto new_width = result_image->image_info.width;

		auto* x_pixel_precalculation_ptr = create_pixel_precalculation_for_x_row(old_width, new_width);

		auto* y_pixel_precalculation_ptr = create_pixel_precalculation_for_y_row(old_height, new_height);

		scale_bilinear_first_half(result_image, source_image, x_pixel_precalculation_ptr, y_pixel_precalculation_ptr);

		scale_nearest_neighbor_second_half(result_image, source_image, x_pixel_precalculation_ptr,
			y_pixel_precalculation_ptr);

		apply_sobel(result_image);

		free(x_pixel_precalculation_ptr);
		free(y_pixel_precalculation_ptr);

	}

	pixel_precalculation* create_pixel_precalculation_for_x_row(const unsigned int old_width, const unsigned int new_width)
	{
		return create_pixel_precalculation(old_width, new_width);
	}

	pixel_precalculation* create_pixel_precalculation_for_y_row(const unsigned int old_height, const unsigned int new_height)
	{
		return create_pixel_precalculation(old_height, new_height);
	}

	pixel_precalculation* create_pixel_precalculation(const unsigned int old_pixel_array_size,
		const unsigned int new_pixel_array_size)
	{
		auto* pixel_precalculation_for_x_rows = static_cast<pixel_precalculation*>(malloc(
			sizeof(pixel_precalculation) * new_pixel_array_size));


		//old_pixel_array_size - 1, the last pixel is (old_pixel_array_size - 1)
		auto real_pixel_weight_increment = (1.0 / static_cast<double>(new_pixel_array_size)) * static_cast<double>(old_pixel_array_size - 1);
		double current_pixel_weight = 0;

		//precalculate weights and front pixel location
		for (unsigned int x = 0; x < new_pixel_array_size; x++)
		{
			const auto source_image_pixel_for_front_pixel_x = static_cast<unsigned int>(current_pixel_weight);
			const auto rear_pixel = source_image_pixel_for_front_pixel_x + 1;


			pixel_precalculation_for_x_rows[x].front_pixel = source_image_pixel_for_front_pixel_x;
			pixel_precalculation_for_x_rows[x].rear_pixel = rear_pixel;

			const auto weight = current_pixel_weight - static_cast<float>(source_image_pixel_for_front_pixel_x);

			pixel_precalculation_for_x_rows[x].front_weight = 1.0 - weight;
			pixel_precalculation_for_x_rows[x].rear_weight = weight;

			current_pixel_weight += real_pixel_weight_increment;
		}

		return pixel_precalculation_for_x_rows;
	}

	void scale_bilinear_first_half(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr)
	{
		for (png_uint_32 y = 0; y < result_image->image_info.height / 2; y++)
		{
			auto y_pixel_front = y_pixel_precalculation_ptr[y].front_pixel;
			auto y_pixel_rear = y_pixel_precalculation_ptr[y].rear_pixel;
			auto y_front_weight = y_pixel_precalculation_ptr[y].front_weight;
			auto y_rear_weight = y_pixel_precalculation_ptr[y].rear_weight;

			for (png_uint_32 x = 0; x < result_image->image_info.width; x++)
			{
				auto x_pixel_front = x_pixel_precalculation_ptr[x].front_pixel;
				auto x_pixel_rear = x_pixel_precalculation_ptr[x].rear_pixel;
				auto x_front_weight = x_pixel_precalculation_ptr[x].front_weight;
				auto x_rear_weight = x_pixel_precalculation_ptr[x].rear_weight;

				png_byte point_00_01_value = source_image->png_rows[y_pixel_front][x_pixel_front] * x_front_weight + source_image->png_rows[y_pixel_front][x_pixel_rear] * x_rear_weight;
				png_byte point_10_11_value = source_image->png_rows[y_pixel_rear][x_pixel_front] * x_front_weight + source_image->png_rows[y_pixel_rear][x_pixel_rear] * x_rear_weight;

				png_byte pixel_p_value = point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight;

				result_image->png_rows[y][x] = pixel_p_value;
			}
		}
	}
	
	void scale_nearest_neighbor_second_half(png_user_struct* result_image, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr)
	{
		for (png_uint_32 y = result_image->image_info.height / 2; y < result_image->image_info.height; y++)
		{
			auto y_pixel_front = y_pixel_precalculation_ptr[y].front_pixel;

			for (png_uint_32 x = 0; x < result_image->image_info.width; x++)
			{
				auto x_pixel_front = x_pixel_precalculation_ptr[x].front_pixel;
				auto x_pixel_rear = x_pixel_precalculation_ptr[x].rear_pixel;
				auto x_front_weight = x_pixel_precalculation_ptr[x].front_weight;
				auto x_rear_weight = x_pixel_precalculation_ptr[x].rear_weight;

				png_byte pixel_p_value;
				if (x_front_weight > x_rear_weight)
				{
					pixel_p_value = source_image->png_rows[y_pixel_front][x_pixel_front];
				}
				else
				{
					pixel_p_value = source_image->png_rows[y_pixel_front][x_pixel_rear];
				}
				
				result_image->png_rows[y][x] = pixel_p_value;
			}
		}
	}

	void apply_sobel(png_user_struct* result_image)
	{
		auto x_row_buffer_amount = (result_image->image_info.width - 2) * 2;
		auto* two_row_buffer = static_cast<png_bytep>(malloc(sizeof(png_byte) * x_row_buffer_amount));
		unsigned current_two_row_buffer_index = 0;


		auto x_row_buffer_half = (x_row_buffer_amount / 2);
		auto current_buffer_end_index = x_row_buffer_amount;
		for (png_uint_32 y_1 = 1; y_1 < result_image->image_info.height - 1; y_1++)
		{
			for (png_uint_32 x_1 = 1; x_1 < result_image->image_info.width - 1; x_1++)
			{
				png_byte xy_sobel_value = calculate_sobel_value(result_image, y_1, x_1);

				two_row_buffer[current_two_row_buffer_index] = xy_sobel_value;

				current_two_row_buffer_index++;
			}

			if (current_two_row_buffer_index == current_buffer_end_index)
			{
				if (current_buffer_end_index == x_row_buffer_amount)
				{
					//copy from first half
					for (unsigned x = 1, x_buf = 0; x < result_image->image_info.width - 1; x++, x_buf++)
					{
						result_image->png_rows[y_1 - 1][x] = two_row_buffer[x_buf];
					}
					//write to first half
					current_two_row_buffer_index = 0;
					current_buffer_end_index = x_row_buffer_half;
				}
				else
				{
					//copy from second half
					for (unsigned x = 1, x_buf = x_row_buffer_half; x < result_image->image_info.width - 1; x++, x_buf++)
					{
						result_image->png_rows[y_1 - 1][x] = two_row_buffer[x_buf];
					}
					//write to second half
					current_two_row_buffer_index = x_row_buffer_half;
					current_buffer_end_index = x_row_buffer_amount;

				}
			}
		}

		auto y_1 = result_image->image_info.height - 2; // two y iterations left
		if (current_two_row_buffer_index == x_row_buffer_amount)
		{
			for (unsigned x = 1, x_buf1 = 0; x < result_image->image_info.width - 1; x++, x_buf1++)
			{
				result_image->png_rows[y_1][x] = two_row_buffer[x_buf1];
			}
		}
		else
		{
			for (unsigned x = 1, x_buf = x_row_buffer_half; x < result_image->image_info.width - 1; x++, x_buf++)
			{
				result_image->png_rows[y_1][x] = two_row_buffer[x_buf];
			}
		}

		free(two_row_buffer);


		//add image frame

		int y_rows[2] = { 0, result_image->image_info.height - 1 };

		for (auto y_row : y_rows)
		{
			for(auto x = 0; x < result_image->image_info.width - 1; x++)
			{
				result_image->png_rows[y_row][x] = 255;
			}
		}

		int x_rows[2] = { 0, result_image->image_info.width - 1 };

		for(auto x_row : x_rows)
		{
			for(auto y = 1; y < result_image->image_info.height - 2; y++)
			{
				result_image->png_rows[y][x_row] = 255;
			}
		}
	}

	png_byte calculate_sobel_value(png_user_struct* result_image, png_uint_32 y_1, png_uint_32 x_1)
	{

		int sobel_operator_x[3][3] = { {-1, 0, 1}, {-2, 0, 2 }, {-1, 0, 1} };
		int sobel_operator_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

		auto y_0 = y_1 - 1;
		auto y_2 = y_1 + 1;
		auto x_0 = x_1 - 1;
		auto x_2 = x_1 + 1;

		auto x_sobel_value =
			result_image->png_rows[y_0][x_0] * sobel_operator_x[0][0]
			+ result_image->png_rows[y_0][x_2] * sobel_operator_x[0][2]
			+ result_image->png_rows[y_1][x_0] * sobel_operator_x[1][0]
			+ result_image->png_rows[y_1][x_2] * sobel_operator_x[1][2]
			+ result_image->png_rows[y_2][x_0] * sobel_operator_x[2][0]
			+ result_image->png_rows[y_2][x_2] * sobel_operator_x[2][2];

		auto y_sobel_value =
			result_image->png_rows[y_0][x_0] * sobel_operator_y[0][0]
			+ result_image->png_rows[y_0][x_1] * sobel_operator_y[0][1]
			+ result_image->png_rows[y_0][x_2] * sobel_operator_y[0][2]
			+ result_image->png_rows[y_2][x_0] * sobel_operator_y[2][0]
			+ result_image->png_rows[y_2][x_1] * sobel_operator_y[2][1]
			+ result_image->png_rows[y_2][x_2] * sobel_operator_y[2][2];

		double xy_sobel_value_double = sqrt(static_cast<double>(x_sobel_value) * (x_sobel_value) + static_cast<double>(y_sobel_value) * (y_sobel_value));

		int xy_sobel_value = static_cast<int>(round((xy_sobel_value_double / 1443) * 255));

		return static_cast<png_byte>(xy_sobel_value);
	}
	
}