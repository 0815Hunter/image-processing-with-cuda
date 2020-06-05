#include <iostream>
#include <vector>

#include "Stopwatch.h"
#include "imageutil.h"
#include "cuda_includes.h"

#include "bilinear_nn_sobel.h"
#include "cuda_scale_image.cuh"
#include "cuda_streams_scale_image.cuh"

#include <windows.h>

enum class execution_strategy {
   seq, cuda_seq_kernel, cuda_streams
};


png_user_struct* scale_image(png_user_struct* src_image_ptr, double scaling_factor, enum class execution_strategy scaling_strategy, kernel_mode
                             kernel_mode);

win32::Stopwatch sw;

void set_result_file_name(std::string& result_image_file_name, execution_strategy strategy, kernel_mode kernel_mode);


void set_test_name(std::string& test_name, execution_strategy strategy, kernel_mode mode);


int main() {
    std::string file_name("czp.png");
	
	auto* source_image_ptr = get_image(file_name.data());
	
    const double scaling_factor = 2;

    std::vector<execution_strategy> strategies;
    std::vector<kernel_mode> modes;
	
    strategies.reserve(3);
    modes.reserve(3);

   
    strategies.push_back(execution_strategy::cuda_seq_kernel);
    strategies.push_back(execution_strategy::seq);
    strategies.push_back(execution_strategy::cuda_streams);

    modes.push_back(kernel_mode::bilinear_nn);
    modes.push_back(kernel_mode::bilinear_nn_sobel);
    modes.push_back(kernel_mode::branch_bilinear_nn_dynamic_sobel);

	
	
    std::vector<double> time_measures;
	
    auto measurement_count = 21;
	time_measures.reserve(measurement_count);

    auto* console_handle = GetStdHandle(STD_OUTPUT_HANDLE);
	

    COORD f;
    f.X = 1000;
    f.Y = 200;
	
    
    SetConsoleScreenBufferSize(console_handle, f);
	
    f.X = 0;
    f.Y = 0;

    unsigned last_y = 0;
    for(execution_strategy s : strategies)
    {
	    for(kernel_mode m : modes)
	    {
	    	if(s == execution_strategy::seq && (m == kernel_mode::bilinear_nn || m == kernel_mode::branch_bilinear_nn_dynamic_sobel))
	    	{
                continue;
	    	}
            if(s == execution_strategy::cuda_streams && m == kernel_mode::branch_bilinear_nn_dynamic_sobel)
            {
                continue;
            }
	    	
            std::string result_image_file_name(file_name);
            set_result_file_name(result_image_file_name, s, m);

            std::string test_name;
            set_test_name(test_name, s, m);

	    	
            std::cout << test_name;
            f.Y++;
            SetConsoleCursorPosition(console_handle, f);

	    	if(s == execution_strategy::seq && scaling_factor > 5)
	    	{
                std::vector<double>().swap(time_measures);
                time_measures.reserve(5);
	    	}
	    	
            for (int i = 0; i < time_measures.capacity(); i++)
            {
                png_user_struct* scaled_image_ptr = scale_image(source_image_ptr, scaling_factor, s, m);
            	
                cudaDeviceReset();

                auto elapsed_milliseconds = sw.ElapsedMilliseconds();
            	
                std::cout << "   " << elapsed_milliseconds;
                f.Y++;
                SetConsoleCursorPosition(console_handle, f);
            	
                time_measures.push_back(elapsed_milliseconds);

                if (i == time_measures.capacity() - 1 && scaling_factor < 10) {
                    write_image(scaled_image_ptr, result_image_file_name.data());
                }
                png_user_struct_free(scaled_image_ptr, struct_type::write);
            }

            double sum = 0, avg = 0;

            time_measures[0] = 0.0;
	    	
            for (auto measure_point : time_measures)
            {
                sum += measure_point;
            }

            avg = sum / (time_measures.capacity() - 1);
            std::cout << "-------------";
            f.Y++;
            SetConsoleCursorPosition(console_handle, f);
            std::cout << "   " << avg;

	    	
            time_measures.clear();
            time_measures.reserve(measurement_count);

            f.X += 50;
	    	if(f.X >= 200)
	    	{
                last_y += measurement_count + 5;
                f.X = 0;
	    	}
            f.Y = last_y;
           
	    	
            SetConsoleCursorPosition(console_handle, f);
	    }
    }

    std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
    
    png_user_struct_free(source_image_ptr, struct_type::read);
  
	
    return 0;
}

void set_result_file_name(std::string& result_image_file_name, execution_strategy strategy, kernel_mode kernel_mode)
{
	auto point_pos = result_image_file_name.find_last_of('.');
    switch (strategy)
    {
    case execution_strategy::seq: result_image_file_name.insert(point_pos, "_scaled_seq"); return;
    case execution_strategy::cuda_seq_kernel: result_image_file_name.insert(point_pos, "_scaled_cuda_seq"); break;
    case execution_strategy::cuda_streams: result_image_file_name.insert(point_pos, "_scaled_cuda_streams"); break;
    default: result_image_file_name.insert(point_pos, "_1");
    }
	
    point_pos = result_image_file_name.find_last_of('.');

	switch (kernel_mode) {
		case kernel_mode::bilinear_nn:  result_image_file_name.insert(point_pos, "_bilinear_nn"); break;

		case kernel_mode::bilinear_nn_sobel: result_image_file_name.insert(point_pos, "_bilinear_nn_sobel"); break;

		case kernel_mode::branch_bilinear_nn_dynamic_sobel: result_image_file_name.insert(point_pos, "_branch_dynamic"); break;
	default: ;
	}
}

void set_test_name(std::string& test_name, execution_strategy strategy, kernel_mode mode)
{
    switch (strategy)
    {
    case execution_strategy::seq: test_name.append( "seq"); return;
    case execution_strategy::cuda_seq_kernel: test_name.append( "cuda_seq"); break;
    case execution_strategy::cuda_streams: test_name.append( "cuda_streams"); break;
    default: test_name.append("1");
    }

    switch (mode) {
    case kernel_mode::bilinear_nn:  test_name.append( "_bilinear_nn"); break;

    case kernel_mode::bilinear_nn_sobel: test_name.append( "_bilinear_nn_sobel"); break;

    case kernel_mode::branch_bilinear_nn_dynamic_sobel: test_name.append( "_branch_dynamic"); break;
    default:;
    }
}

png_user_struct* scale_image(png_user_struct* src_image_ptr, const double scaling_factor, const enum class execution_strategy scaling_strategy, kernel_mode
                             kernel_mode) {

    png_user_struct* image_to_scale_ptr = create_result_image_container(src_image_ptr, scaling_factor);

    if (scaling_factor < 1) {
        std::cout << "scaling factor < 1" << std::endl;
        return image_to_scale_ptr;
    }
	
    sw.Start();
    switch (scaling_strategy) {
    case execution_strategy::seq:
	    seq::scale_image_apply_sobel(src_image_ptr, image_to_scale_ptr);
    	break;
    case execution_strategy::cuda_seq_kernel:
        cuda_seq::scale_image_apply_sobel(src_image_ptr, image_to_scale_ptr, kernel_mode);
    	break;
    case execution_strategy::cuda_streams:
        cuda_streams_example::scale_image_apply_sobel(src_image_ptr, image_to_scale_ptr, kernel_mode);
    	break;
    default: ;
    }
    sw.Stop();
	
    return image_to_scale_ptr;
}
