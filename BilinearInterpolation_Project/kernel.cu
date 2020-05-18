#include <cstdio>
#include <cstdlib>
#include <host_defines.h>



#include "bilinearFilter.h"
#include "imageutil.h"
#include "nearestNeighbour.h"
#include "bilinear_filter_cuda.cuh"

enum class scaling_strategy {
    NEAREST_NEIGHBOUR, BILINEAR
};


png_user_struct* scale_image(png_user_struct* src_image_ptr, double scaling_factor, enum scaling_strategy scaling_strategy);




int main(void) {

    png_user_struct* source_image_ptr = get_image("czp.png");

    const double scaling_factor = 3;

    png_user_struct* scaled_image_ptr = scale_image(source_image_ptr, scaling_factor, scaling_strategy::BILINEAR);

    write_image(scaled_image_ptr, "czp_scaled.png");


    png_user_struct_free(source_image_ptr, struct_type::READ);
    png_user_struct_free(scaled_image_ptr, struct_type::WRITE);
   

    return 0;
}

png_user_struct* scale_image(png_user_struct* src_image_ptr, const double scaling_factor, const enum class scaling_strategy scaling_strategy) {

    png_user_struct* image_to_scale_ptr = create_image_to_scale(src_image_ptr, scaling_factor);

    if (scaling_factor < 1) {
        scaleImageNearestNeigbour(src_image_ptr, image_to_scale_ptr, scaling_factor);
        return image_to_scale_ptr;
    }

    switch (scaling_strategy) {
    case scaling_strategy::NEAREST_NEIGHBOUR:
        scaleImageNearestNeigbour(src_image_ptr, image_to_scale_ptr, scaling_factor);
        break;
    case scaling_strategy::BILINEAR:
        scale_bilinear_cuda(src_image_ptr, image_to_scale_ptr);
        break;
    }

    return image_to_scale_ptr;
}
