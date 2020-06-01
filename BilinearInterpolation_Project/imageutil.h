#ifndef LAB1_1_IMAGEUTIL_H
#define LAB1_1_IMAGEUTIL_H

#include <png.h>

typedef struct image_inf {
    png_uint_32 width;
    png_uint_32 height;
    int bit_depth;
    int color_type;
    int interlace_method;
    int compression_method;
    int filter_method;
} image_info;
typedef struct img {
    png_structp png_ptr;
    png_infop png_info_ptr;
	
    png_bytepp png_rows;
    size_t png_row_bytes;
	
    image_info image_info;
} png_user_struct;

enum class struct_type {
    read, write
};


void write_image(png_user_struct* image, const char* file_name);

void png_user_struct_free(png_user_struct* img, struct_type type);

png_user_struct* get_image(const char *fileName);

png_user_struct* create_result_image_container(png_user_struct* src, double scaling_factor);

png_bytep png_util_create_flat_bytes_p_from_row_pp(png_bytepp png_rows, png_uint_32 width, png_uint_32 height, png_uint_32 png_bytes_size);

#endif //LAB1_1_IMAGEUTIL_H
