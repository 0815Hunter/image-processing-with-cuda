#include <iostream>

#include "Stopwatch.h"
#include "imageutil.h"

#include "nearestNeighbour.h"

#include "bilinearFilter.h"
#include "bilinear_filter_cuda.cuh"
#include "bilinear_filter_cuda_streams.cuh"

enum class scaling_strategy {
   seq, cuda_seq_kernel, cuda_parallel_kernel
};


png_user_struct* scale_image(png_user_struct* src_image_ptr, double scaling_factor, enum class scaling_strategy scaling_strategy);

win32::Stopwatch sw;


int main() {

    png_user_struct* source_image_ptr = get_image("czp.png");

    const double scaling_factor = 3;

    png_user_struct* scaled_image_ptr = scale_image(source_image_ptr, scaling_factor, scaling_strategy::cuda_parallel_kernel);
    std::cout << sw.ElapsedMilliseconds() << std::endl;

    //write_image(scaled_image_ptr, "czp_scaled.png");
	
    png_user_struct_free(source_image_ptr, struct_type::read);
    png_user_struct_free(scaled_image_ptr, struct_type::write);
  
	
    return 0;
}

png_user_struct* scale_image(png_user_struct* src_image_ptr, const double scaling_factor, const enum class scaling_strategy scaling_strategy) {

    png_user_struct* image_to_scale_ptr = create_image_to_scale(src_image_ptr, scaling_factor);

    if (scaling_factor < 1) {
        scaleImageNearestNeigbour(src_image_ptr, image_to_scale_ptr, scaling_factor);
        return image_to_scale_ptr;
    }
	
    sw.Start();
    switch (scaling_strategy) {
    case scaling_strategy::seq:
	    seq::scale_bilinear(src_image_ptr, image_to_scale_ptr);
    	break;
    case scaling_strategy::cuda_seq_kernel:
        cuda_seq::scale_bilinear(src_image_ptr, image_to_scale_ptr);
    	break;
    case scaling_strategy::cuda_parallel_kernel:
        cuda_streams_example::scale_bilinear(src_image_ptr, image_to_scale_ptr);
    	break;
    default: ;
    }
    sw.Stop();
	
    return image_to_scale_ptr;
}
