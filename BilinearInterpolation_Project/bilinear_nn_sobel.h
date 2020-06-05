#ifndef BILINEAR_NN_SOBEL_H
#define BILINEAR_NN_SOBEL_H
#include "imageutil.h"

namespace seq
{
	void scale_image_apply_sobel(png_user_struct* source_image, png_user_struct* result_image);
}

#endif // BILINEAR_NN_SOBEL_H
