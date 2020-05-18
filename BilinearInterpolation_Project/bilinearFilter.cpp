//
// Created by Sebastian on 19.01.2020.
//

#include "imageutil.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <png.h>



PixelPrecalculation* create_pixel_precalculation_for_x_row(unsigned int old_width, unsigned int new_width);

PixelPrecalculation* create_pixel_precalculation_for_y_row(unsigned int old_height, unsigned int new_height);

PixelPrecalculation* create_pixel_precalculation(unsigned int old_pixel_array_size, unsigned int new_pixel_array_size);


void fill_image_to_scale(png_user_struct* image_to_scale, png_user_struct* source_image, PixelPrecalculation* x_pixel_precalculation_ptr, PixelPrecalculation* y_pixel_precalculation_ptr);


void scale_bilinear(png_user_struct* source_image, png_user_struct* image_to_scale)
{
	const auto old_height = source_image->image_info.height;
	const auto new_height = image_to_scale->image_info.height;
	const auto old_width = source_image->image_info.width;
	const auto new_width = image_to_scale->image_info.width;

	auto x_pixel_precalculation_ptr = create_pixel_precalculation_for_x_row(old_width, new_width);

	auto y_pixel_precalculation_ptr = create_pixel_precalculation_for_y_row(old_height, new_height);

	fill_image_to_scale(image_to_scale, source_image, x_pixel_precalculation_ptr, y_pixel_precalculation_ptr);

	free(x_pixel_precalculation_ptr);
	free(y_pixel_precalculation_ptr);

	printf("Hoarray");
}

PixelPrecalculation* create_pixel_precalculation_for_x_row(const unsigned int old_width, const unsigned int new_width)
{
	return create_pixel_precalculation(old_width, new_width);
}

PixelPrecalculation* create_pixel_precalculation_for_y_row(const unsigned int old_height, const unsigned int new_height)
{
	return create_pixel_precalculation(old_height, new_height);
}

PixelPrecalculation* create_pixel_precalculation(const unsigned int old_pixel_array_size,
                                                 const unsigned int new_pixel_array_size)
{
	PixelPrecalculation* pixel_precalculation_for_x_rows = (PixelPrecalculation*)malloc(
		sizeof(PixelPrecalculation) * new_pixel_array_size);


	//old_pixel_array_size - 1, the last pixel is (old_pixel_array_size - 1)
	const float real_pixel_weight_increment = (1.0F / (float)new_pixel_array_size) * (float)(old_pixel_array_size - 1);
	float current_pixel_weight = 0;

	//precalculate weights and front pixel location
	for (unsigned int x = 0; x < new_pixel_array_size; x++)
	{
		const auto source_image_pixel_for_front_pixel_x = static_cast<unsigned int>(current_pixel_weight);
		const auto rear_pixel = source_image_pixel_for_front_pixel_x + 1;


		pixel_precalculation_for_x_rows[x].frontPixel = source_image_pixel_for_front_pixel_x;
		pixel_precalculation_for_x_rows[x].rearPixel = rear_pixel;

		const auto weight = current_pixel_weight - (float)source_image_pixel_for_front_pixel_x;

		pixel_precalculation_for_x_rows[x].frontWeight = 1 - weight;
		pixel_precalculation_for_x_rows[x].rearWeight = weight;

		current_pixel_weight += real_pixel_weight_increment;
	}

	return pixel_precalculation_for_x_rows;
}


void fill_image_to_scale(png_user_struct* image_to_scale, png_user_struct* source_image, PixelPrecalculation* x_pixel_precalculation_ptr,
	PixelPrecalculation* y_pixel_precalculation_ptr)
{

	for(png_uint_32 y = 0; y < image_to_scale->image_info.height;y++)
	{
		auto y_pixel_front = y_pixel_precalculation_ptr[y].frontPixel;
		auto y_pixel_rear = y_pixel_precalculation_ptr[y].rearPixel;
		auto y_front_weight = y_pixel_precalculation_ptr[y].frontWeight;
		auto y_rear_weight = y_pixel_precalculation_ptr[y].rearWeight;

		auto* png_source_image_bytepp = source_image->png_rows;

		auto png_row_bytes = source_image->png_row_bytes;

		

		//only black/white pictures atm!
		if (png_row_bytes != source_image->image_info.width)
		{
			// ReSharper disable once StringLiteralTypo
			printf("Rowbytes !=source image width");
		}

		for (png_uint_32 x = 0; x < image_to_scale->image_info.width; x++)
		{
			auto x_pixel_front = x_pixel_precalculation_ptr[x].frontPixel;
			auto x_pixel_rear = x_pixel_precalculation_ptr[x].rearPixel;
			auto x_front_weight = x_pixel_precalculation_ptr[x].frontWeight;
			auto x_rear_weight = x_pixel_precalculation_ptr[x].rearWeight;

			png_byte point_00_01_value = png_source_image_bytepp[y_pixel_front][x_pixel_front] * x_front_weight + png_source_image_bytepp[y_pixel_front][x_pixel_rear] * x_rear_weight;
			png_byte point_10_11_value = png_source_image_bytepp[y_pixel_rear][x_pixel_front] * x_front_weight + png_source_image_bytepp[y_pixel_rear][x_pixel_rear] * x_rear_weight;

			png_byte pixel_p_value = point_00_01_value * y_front_weight + point_10_11_value * y_rear_weight;

			image_to_scale->png_rows[y][x] = pixel_p_value;
			
		}
	}
	
}
