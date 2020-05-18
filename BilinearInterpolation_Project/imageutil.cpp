//
// Created by Sebastian on 28.11.2019.
//


extern "C" {
#include <png.h>
}

#include "imageutil.h"


#include <cstdio>
#include <cstdlib>

void set_image_row_info(png_user_struct* image);


png_user_struct *get_image(const char *fileName) {

	
	auto* const image_ptr =  static_cast<png_user_struct*>(malloc(sizeof(png_user_struct)));

	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	png_infop png_info_ptr = png_create_info_struct(png_ptr);

    FILE* fp = fopen(fileName, "rb");
	
    png_init_io(png_ptr, fp);

    png_read_png(png_ptr, png_info_ptr, 0, 0);
    fclose(fp);

    image_ptr->png_ptr = png_ptr;
    image_ptr->png_info_ptr = png_info_ptr;
	

    set_image_info_from_png(image_ptr);
    set_image_row_info(image_ptr);
	
    //png_data_freer(image_ptr->png_ptr, image_ptr->png_info_ptr)

    return image_ptr;

}

png_user_struct *create_image_to_scale(png_user_struct *src, double scaling_factor)
{
	auto* image_to_scale =  static_cast<png_user_struct*>(malloc(sizeof(png_user_struct)));

    png_structp png_write_struct = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    png_infop png_info_struct = png_create_info_struct(png_write_struct);

	
    png_uint_32 width = 0;
    png_uint_32 height = 0;
    int bit_depth = 0;
    int color_type = 0;
    int interlace_method = 0;
    int compression_method = 0;
    int filter_method = 0;

    png_get_IHDR(src->png_ptr, src->png_info_ptr, &width, &height, &bit_depth, &color_type, &interlace_method, &compression_method,
        &filter_method);

    height = (unsigned int)(height * scaling_factor);
    width = (unsigned int)(width * scaling_factor);

    png_set_IHDR(png_write_struct, png_info_struct, width, height, bit_depth, color_type, interlace_method, compression_method, filter_method);

    image_to_scale->png_ptr = png_write_struct;
    image_to_scale->png_info_ptr = png_info_struct;
    set_image_info_from_png(image_to_scale);

    auto* row_pointers = static_cast<png_bytepp>(png_malloc(image_to_scale->png_ptr, image_to_scale->image_info.height * sizeof(png_bytep)));

    const auto pixel_size = 1;// black/white pixel
    for (png_uint_32 row = 0; row < image_to_scale->image_info.height; row++)
    {
        row_pointers[row] = static_cast<png_bytep>(png_malloc(image_to_scale->png_ptr, sizeof(image_to_scale->image_info.bit_depth) * image_to_scale->image_info.width * pixel_size));
    }
    png_set_rows(image_to_scale->png_ptr, image_to_scale->png_info_ptr, row_pointers);
    set_image_row_info(image_to_scale);
   

    return image_to_scale;
}

unsigned int get_pixel_count(image_info image_resolution) {
    return image_resolution.height * image_resolution.width;
}


void set_image_row_info(png_user_struct* image)
{
    image->png_rows = png_get_rows(image->png_ptr, image->png_info_ptr);

    image->png_row_bytes = png_get_rowbytes(image->png_ptr, image->png_info_ptr);
}

void set_image_info_from_png(png_user_struct* image)
{
    png_uint_32 width;
    png_uint_32 height;
    int bit_depth;
    int color_type;
    int interlace_method;
    int compression_method;
    int filter_method;

    png_get_IHDR(image->png_ptr, image->png_info_ptr, &width, &height, &bit_depth, &color_type, &interlace_method, &compression_method,
        &filter_method);

    image->image_info.width = width;
    image->image_info.height = height;
    image->image_info.bit_depth = bit_depth;
    image->image_info.color_type = color_type;
    image->image_info.interlace_method = interlace_method;
    image->image_info.compression_method = compression_method;
    image->image_info.filter_method = filter_method;
}

void write_image(png_user_struct* image, const char* file_name) {

    FILE* fp;
    fp = fopen(file_name, "wb");

    png_init_io(image->png_ptr, fp);

    png_write_png(image->png_ptr, image->png_info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    fclose(fp);

}

void png_user_struct_free(png_user_struct* img, const struct_type type)
{
	switch (type)
	{
		case struct_type::READ:
		png_destroy_read_struct(&img->png_ptr, &img->png_info_ptr, static_cast<png_infopp>(nullptr));
		break;
		case struct_type::WRITE:
		png_destroy_write_struct(&img->png_ptr, &img->png_info_ptr);
		break;
		default: ;
	}
	
    free(img);
}