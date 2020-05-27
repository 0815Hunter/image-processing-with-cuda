#include <cstdlib>


#include "common_structs.cuh"
#include "imageutil.h"

namespace seq
{
	pixel_precalculation* create_pixel_precalculation_for_x_row(unsigned int old_width, unsigned int new_width);

	pixel_precalculation* create_pixel_precalculation_for_y_row(unsigned int old_height, unsigned int new_height);

	pixel_precalculation* create_pixel_precalculation(unsigned int old_pixel_array_size, unsigned int new_pixel_array_size);

	void scale_bilinear_first_half(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr);

	void scale_nearest_neighbor_second_half(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr);

	void fill_image_to_scale(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr);


	void scale_bilinear(png_user_struct* source_image, png_user_struct* image_to_scale)
	{
		const auto old_height = source_image->image_info.height;
		const auto new_height = image_to_scale->image_info.height;
		const auto old_width = source_image->image_info.width;
		const auto new_width = image_to_scale->image_info.width;

		auto* x_pixel_precalculation_ptr = create_pixel_precalculation_for_x_row(old_width, new_width);

		auto* y_pixel_precalculation_ptr = create_pixel_precalculation_for_y_row(old_height, new_height);

		fill_image_to_scale(image_to_scale, source_image, x_pixel_precalculation_ptr, y_pixel_precalculation_ptr);

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
		const auto real_pixel_weight_increment = (1.0F / static_cast<float>(new_pixel_array_size)) * static_cast<float>(old_pixel_array_size - 1);
		float current_pixel_weight = 0;

		//precalculate weights and front pixel location
		for (unsigned int x = 0; x < new_pixel_array_size; x++)
		{
			const auto source_image_pixel_for_front_pixel_x = static_cast<unsigned int>(current_pixel_weight);
			const auto rear_pixel = source_image_pixel_for_front_pixel_x + 1;


			pixel_precalculation_for_x_rows[x].front_pixel = source_image_pixel_for_front_pixel_x;
			pixel_precalculation_for_x_rows[x].rear_pixel = rear_pixel;

			const auto weight = current_pixel_weight - (float)source_image_pixel_for_front_pixel_x;

			pixel_precalculation_for_x_rows[x].front_weight = 1 - weight;
			pixel_precalculation_for_x_rows[x].rear_weight = weight;

			current_pixel_weight += real_pixel_weight_increment;
		}

		return pixel_precalculation_for_x_rows;
	}

	void fill_image_to_scale(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr,
	                         pixel_precalculation* y_pixel_precalculation_ptr)
	{
		
		scale_bilinear_first_half(image_to_scale, source_image, x_pixel_precalculation_ptr, y_pixel_precalculation_ptr);

		scale_nearest_neighbor_second_half(image_to_scale, source_image, x_pixel_precalculation_ptr,
		                                    y_pixel_precalculation_ptr);
	}
	void scale_bilinear_first_half(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr)
	{
		for (png_uint_32 y = 0; y < image_to_scale->image_info.height / 2; y++)
		{
			auto y_pixel_front = y_pixel_precalculation_ptr[y].front_pixel;
			auto y_pixel_rear = y_pixel_precalculation_ptr[y].rear_pixel;
			auto y_front_weight = y_pixel_precalculation_ptr[y].front_weight;
			auto y_rear_weight = y_pixel_precalculation_ptr[y].rear_weight;

			for (png_uint_32 x = 0; x < image_to_scale->image_info.width; x++)
			{
				auto x_pixel_front = x_pixel_precalculation_ptr[x].front_pixel;
				auto x_pixel_rear = x_pixel_precalculation_ptr[x].rear_pixel;
				auto x_front_weight = x_pixel_precalculation_ptr[x].front_weight;
				auto x_rear_weight = x_pixel_precalculation_ptr[x].rear_weight;

				png_byte point_00_01_value = source_image->png_rows[y_pixel_front][x_pixel_front] * x_front_weight + source_image->png_rows[y_pixel_front][x_pixel_rear] * x_rear_weight;
				png_byte point_10_11_value = source_image->png_rows[y_pixel_rear][x_pixel_front] * x_front_weight + source_image->png_rows[y_pixel_rear][x_pixel_rear] * x_rear_weight;

				png_byte pixel_p_value = point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight;

				image_to_scale->png_rows[y][x] = pixel_p_value;
			}
		}
	}
	
	void scale_nearest_neighbor_second_half(png_user_struct* image_to_scale, png_user_struct* source_image, pixel_precalculation* x_pixel_precalculation_ptr, pixel_precalculation* y_pixel_precalculation_ptr)
	{
		for (png_uint_32 y = image_to_scale->image_info.height / 2; y < image_to_scale->image_info.height; y++)
		{
			auto y_pixel_front = y_pixel_precalculation_ptr[y].front_pixel;

			for (png_uint_32 x = 0; x < image_to_scale->image_info.width; x++)
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

				image_to_scale->png_rows[y][x] = pixel_p_value;
			}
		}
	}
	
}