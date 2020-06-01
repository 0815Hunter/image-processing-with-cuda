#include <iostream>

#include "Stopwatch.h"
#include "imageutil.h"

#include "bilinearFilter.h"
#include "cuda_scale_image.cuh"
#include "cuda_streams_scale_image.cuh"

enum class scaling_strategy {
   seq, cuda_seq_kernel, cuda_streams
};


png_user_struct* scale_image(png_user_struct* src_image_ptr, double scaling_factor, enum class scaling_strategy scaling_strategy);

win32::Stopwatch sw;

void set_result_file_name(scaling_strategy strategy, const char*& file_name_result);


int main() {

    png_user_struct* source_image_ptr = get_image("table.png");
    
    const double scaling_factor = 2;

    auto strategy = scaling_strategy::cuda_streams;
    
    const char* file_name_result;
    set_result_file_name(strategy, file_name_result);
	
    png_user_struct* scaled_image_ptr = scale_image(source_image_ptr, scaling_factor, strategy);
    std::cout << sw.ElapsedMilliseconds() << std::endl;

    
    write_image(scaled_image_ptr, file_name_result);
	
    png_user_struct_free(source_image_ptr, struct_type::read);
    png_user_struct_free(scaled_image_ptr, struct_type::write);
  
	
    return 0;
}

void set_result_file_name(scaling_strategy strategy, const char*& file_name_result)
{
    switch (strategy)
    {
    case scaling_strategy::seq: file_name_result = "table_scaled_seq.png"; break;
    case scaling_strategy::cuda_seq_kernel: file_name_result = "table_scaled_cuda_seq.png"; break;
    case scaling_strategy::cuda_streams: file_name_result = "table_scaled_cuda_streams.png"; break;
    default: file_name_result = "table_scaled.png";
    }
}

png_user_struct* scale_image(png_user_struct* src_image_ptr, const double scaling_factor, const enum class scaling_strategy scaling_strategy) {

    png_user_struct* image_to_scale_ptr = create_result_image_container(src_image_ptr, scaling_factor);

    if (scaling_factor < 1) {
        std::cout << "scaling factor < 1" << std::endl;
        return image_to_scale_ptr;
    }
	
    sw.Start();
    switch (scaling_strategy) {
    case scaling_strategy::seq:
	    seq::scale_image_apply_sobel(src_image_ptr, image_to_scale_ptr);
    	break;
    case scaling_strategy::cuda_seq_kernel:
        cuda_seq::scale_image_apply_sobel(src_image_ptr, image_to_scale_ptr);
    	break;
    case scaling_strategy::cuda_streams:
        cuda_streams_example::scale_image_apply_sobel(src_image_ptr, image_to_scale_ptr);
    	break;
    default: ;
    }
    sw.Stop();
	
    return image_to_scale_ptr;
}
