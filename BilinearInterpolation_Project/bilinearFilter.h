#ifndef LAB1_1_BILINEARFILTER_H
#define LAB1_1_BILINEARFILTER_H

#include "imageutil.h"

namespace seq
{
	void scale_bilinear(png_user_struct* source_image, png_user_struct* image_to_scale);
}
#endif //LAB1_1_BILINEARFILTER_H
