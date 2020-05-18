//
// Created by Sebastian on 28.11.2019.
//

#ifndef LAB1_1_IMAGEUTIL_H
#define LAB1_1_IMAGEUTIL_H

extern "C"{
#include <png.h>
}


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

typedef struct pix {
    unsigned int x;
    unsigned int y;
} Pixel;

typedef struct precal {
    unsigned int frontPixel;
    unsigned int rearPixel;
    float frontWeight;
    float rearWeight;
} PixelPrecalculation;


enum class struct_type {
    READ, WRITE
};

png_user_struct* get_image(const char *fileName);

void write_image(png_user_struct *image, const char *file_name);

unsigned int get_pixel_count(image_info image_resolution);

void set_image_info_from_png(png_user_struct *image);

png_user_struct* create_image_to_scale(png_user_struct* src, double scaling_factor);

void png_user_struct_free(png_user_struct* img, struct_type type);

#endif //LAB1_1_IMAGEUTIL_H
