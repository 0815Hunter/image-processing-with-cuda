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
		auto* precalculation = static_cast<pixel_precalculation*>(malloc(
			sizeof(pixel_precalculation) * new_pixel_array_size));


		//old_pixel_array_size - 1, the last pixel is (old_pixel_array_size - 1)
		auto weight_increment = (1.0 / static_cast<double>(new_pixel_array_size)) * static_cast<double>(old_pixel_array_size - 1);
		
		double current_weight = 0;

		//precalculate weights and front pixel location
		for (unsigned int i = 0; i < new_pixel_array_size; i++)
		{
			const auto source_image_front_pixel = static_cast<unsigned int>(current_weight);

			precalculation[i].front_pixel = source_image_front_pixel;
			precalculation[i].rear_pixel = source_image_front_pixel + 1;

			const auto weight = current_weight - static_cast<double>(source_image_front_pixel);

			precalculation[i].front_weight = 1.0 - weight;
			precalculation[i].rear_weight = weight;

			current_weight += weight_increment;
		}

		return precalculation;
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
		auto two_row_buffer_size = (result_image->image_info.width - 2) * 2;
		auto* two_row_buffer = static_cast<png_bytep>(malloc(sizeof(png_byte) * two_row_buffer_size));
		
		unsigned current_two_row_buffer_index = 0;

		auto row_buffer_first_row = (two_row_buffer_size / 2);
		auto current_buffer_end_index = two_row_buffer_size;
		for (png_uint_32 y = 1; y < result_image->image_info.height - 1; y++)
		{
			for (png_uint_32 x = 1; x < result_image->image_info.width - 1; x++)
			{
				png_byte xy_sobel_value = calculate_sobel_value(result_image, y, x);

				two_row_buffer[current_two_row_buffer_index] = xy_sobel_value;

				current_two_row_buffer_index++;
			}

			if (current_two_row_buffer_index == current_buffer_end_index)
			{
				if (current_buffer_end_index == two_row_buffer_size)
				{
					//copy from first half
					for (unsigned x = 1, x_buf = 0; x < result_image->image_info.width - 1; x++, x_buf++)
					{
						result_image->png_rows[y - 1][x] = two_row_buffer[x_buf];
					}
					//write to first half
					current_two_row_buffer_index = 0;
					current_buffer_end_index = row_buffer_first_row;
				}
				else
				{
					//copy from second half
					for (unsigned x = 1, buf_i = row_buffer_first_row; x < result_image->image_info.width - 1; x++, buf_i++)
					{
						result_image->png_rows[y - 1][x] = two_row_buffer[buf_i];
					}
					//write to second half
					current_two_row_buffer_index = row_buffer_first_row;
					current_buffer_end_index = two_row_buffer_size;

				}
			}
		}

		auto last_row_to_fill = result_image->image_info.height - 2; // one y iteration left
		if (current_two_row_buffer_index == two_row_buffer_size)
		{
			for (unsigned x = 1, buf_i = 0; x < result_image->image_info.width - 2; x++, buf_i++)
			{
				result_image->png_rows[last_row_to_fill][x] = two_row_buffer[buf_i];
			}
		}
		else
		{
			for (unsigned x = 1, buf_i = row_buffer_first_row; x < result_image->image_info.width - 2; x++, buf_i++)
			{
				result_image->png_rows[last_row_to_fill][x] = two_row_buffer[buf_i];
			}
		}

		free(two_row_buffer);


		//add image frame

		int first_and_last_col[2] = { 0, result_image->image_info.height - 1 };

		for (auto y : first_and_last_col)
		{
			for(auto x = 0; x < result_image->image_info.width - 1; x++)
			{
				result_image->png_rows[y][x] = 255;
			}
		}

		int first_and_last_row[2] = { 0, result_image->image_info.width - 1 };

		for(auto x_row : first_and_last_row)
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