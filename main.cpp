// this is out command line argument library
#include <tclap/CmdLine.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <fstream>
#include <random>
#include <map>
#include <chrono>

// bitmap header-only library
#include "bitmap_image.hpp"

cl_device_id get_device_id(int platform_index, int device_index)
{
    // store and check for OpenCL errors
    cl_int err;

    // get number of platforms
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(
            0,
            nullptr,
            &num_platforms);
    if (err != CL_SUCCESS || !num_platforms)
    {
        std::cout << "No OpenCL platforms available" << std::endl;
        exit(0);
    }
    if ((cl_uint)platform_index >= num_platforms)
    {
        std::cout
            << "Platform index is invalid: only " << num_platforms << " available" << std::endl;
        exit(0);
    }

    // get all platform IDs
    std::vector<cl_platform_id> platform_ids(num_platforms);
    err = clGetPlatformIDs(
            num_platforms,
            platform_ids.data(),
            nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clGetPlatformIDs with error code " << err << std::endl;
        exit(0);
    }

    // select our platform
    cl_platform_id platform_id = platform_ids[platform_index];

    // get all device ids for the selected platform
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(
            platform_id,
            CL_DEVICE_TYPE_GPU,
            0,
            nullptr,
            &num_devices);
    if (err != CL_SUCCESS)
    {
        std::cout << "No OpenCL GPU devices available on this platform" << std::endl;
        exit(0);
    }
    if ((cl_uint)device_index >= num_devices)
    {
        std::cout
            << "Device index is invalid: only " << num_devices << " available" << std::endl;
        exit(0);
    }

    // get all device ids
    std::vector<cl_device_id> device_ids(num_devices);
    err = clGetDeviceIDs(
            platform_id,
            CL_DEVICE_TYPE_GPU,
            num_devices,
            device_ids.data(),
            nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clGetDeviceIDs with error code " << err << std::endl;
        exit(0);
    }

    // select our id
    cl_device_id device_id = device_ids[device_index];
    
    // print some info to the screen
    size_t name_length = 0;
    err = clGetDeviceInfo(
            device_id,
            CL_DEVICE_NAME,
            0,
            nullptr,
            &name_length);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clGetDeviceInfo with error code " << err << std::endl;
        exit(0);
    }

    std::vector<char> dev_name(name_length);
    err = clGetDeviceInfo(
            device_id,
            CL_DEVICE_NAME,
            name_length,
            (void*)dev_name.data(),
            nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clGetDeviceInfo with error code " << err << std::endl;
        exit(0);
    } dev_name.push_back('\0');
    //std::cout << "Using device: " << dev_name.data() << std::endl;

    // return our device id
    return device_id;
}

int main(int argc, const char ** argv)
{
    // command line arguments for algorithm
    TCLAP::CmdLine cmdline(
            "GPUDEP accelerates simulation of sputter deposition using OpenCL",
            ' ',
            "0.0.1");

    TCLAP::ValueArg<std::string> output(
            "o",
            "output",
            "filename prefix of output data",
            false,
            "output",
            "filename");
    TCLAP::ValueArg<std::string> input(
            "i",
            "input",
            "OpenCL file with our kernels",
            false,
            "kernels.cl",
            "filename");
    TCLAP::ValueArg<int> gridsize(
            "g",
            "gridsize",
            "number of sizes along single dimension",
            false,
            128,
            "integer");
    TCLAP::ValueArg<int> timesteps(
            "t",
            "timesteps",
            "number of timesteps to execute per layer",
            false,
            7000,
            "integer");
    TCLAP::ValueArg<int> filmsteps(
            "f",
            "filmsteps",
            "number of layers",
            false,
            20,
            "integer");
    TCLAP::ValueArg<int> seed(
            "",
            "seed",
            "seed for the number generator",
            false,
            994373,
            "integer");

    // command line arguments for OpenCL
    TCLAP::ValueArg<int> platform(
            "",
            "platform",
            "index of OpenCL platform to use",
            false,
            0,
            "integer [0,N)");
    TCLAP::ValueArg<int> device(
            "",
            "device",
            "index of OpenCL device to use",
            false,
            0,
            "integer [0,N)");

    TCLAP::SwitchArg bitmaps(
            "b",
            "bitmaps",
            "also create bitmaps of layers",
            false);

    // add and parse the command line arguments
    cmdline.add(output);
    cmdline.add(input);
    cmdline.add(gridsize);
    cmdline.add(timesteps);
    cmdline.add(filmsteps);
    cmdline.add(seed);
    cmdline.add(platform);
    cmdline.add(device);
    cmdline.add(bitmaps);
    cmdline.parse(argc, argv);

    // get our device id
    cl_device_id device_id = get_device_id(platform.getValue(), device.getValue());

    // create context
    cl_int err;
    cl_context context = clCreateContext(
            NULL,
            1,
            &device_id,
            nullptr,
            nullptr,
            &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clCreateContext with error code " << err << std::endl;
        exit(0);
    }

    // load kernel code
    std::ifstream file(input.getValue(), std::ios::binary);
    if (file.bad() )
    {
        std::cout << "Failed to open kernels.cl file" << std::endl;

        // release context
        clReleaseContext(context);
        exit(0);
    }

    file.seekg(0, std::ios::end); // seek to end
    size_t num_bytes = size_t(file.tellg()); // get number of bytes skipped
    file.seekg(0, std::ios::beg); // seek back to beginning

    std::vector<char> source; // space for bytes
    source.resize(num_bytes);
    file.read(source.data(), num_bytes); // read all bytes
    char * kernel_code = source.data();

    // build program
    cl_program program = clCreateProgramWithSource(
            context,
            1, // number of c-strings
            (const char**)(&kernel_code), // c-strings
            (const size_t*)(&num_bytes), // array of c-strings lengths
            &err);
    if (err != CL_SUCCESS)
    {
            std::cout << "Failed clCreateProgramWithSource with error code " << err << std::endl;

            // release context
            clReleaseContext(context);
            exit(0);
    }
    
    // prepare build options string
    std::ostringstream build_options;
    build_options
        << " -D GRIDSIZE=" << gridsize.getValue()
        << " -D TIMESTEPS=" << timesteps.getValue()
        << " -D FILMSTEPS=" << filmsteps.getValue();
    
    err = clBuildProgram(
            program,
            1, // number of devices to build for
            &device_id, // array of devices
            build_options.str().c_str(), // preprocessor, optimizations etc
            nullptr, // callback routine
            nullptr); // callback data
    if (err != CL_SUCCESS)
    {
        // get the compilation errors
        std::cout << "error building program:" << std::endl;
        clGetProgramBuildInfo(
                program,
                device_id,
                CL_PROGRAM_BUILD_LOG,
                0, // number of bytes to get
                nullptr, // where to put them
                &num_bytes); // copy number of bytes available

        std::vector<char> build_log(num_bytes);
        clGetProgramBuildInfo(
                program,
                device_id,
                CL_PROGRAM_BUILD_LOG,
                num_bytes,
                build_log.data(),
                nullptr);

        // print to screen
        std::cout << build_log.data() << std::endl;
        std::cout.flush();
        
        // release context
        clReleaseProgram(program);
        clReleaseContext(context);
        exit(0);
    }

    // create kernels
    cl_kernel initialize_kernel = clCreateKernel(
            program,
            "initialize",
            &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clCreateKernel with error code " << err << std::endl;
        
        // release context
        clReleaseProgram(program);
        clReleaseContext(context);
        exit(0);
    }
    cl_kernel timestep_kernel_one = clCreateKernel(
            program,
            "timestep",
            &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clCreateKernel with error code " << err << std::endl;
        
        // release context
        clReleaseKernel(initialize_kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        exit(0);
    }
    
    // create kernels
    cl_kernel timestep_kernel_two = clCreateKernel(
            program,
            "timestep",
            &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clCreateKernel with error code " << err << std::endl;
        
        // release context
        clReleaseKernel(timestep_kernel_one);
        clReleaseKernel(initialize_kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        exit(0);
    }
    
    // create kernels
    cl_kernel underlayer_kernel = clCreateKernel(
            program,
            "underlayer",
            &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clCreateKernel with error code " << err << std::endl;
        
        // release context
        clReleaseKernel(timestep_kernel_two);
        clReleaseKernel(timestep_kernel_one);
        clReleaseKernel(initialize_kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        exit(0);
    }

    // create command queue
    cl_command_queue command_queue = clCreateCommandQueue(
            context,
            device_id,
            0,//CL_QUEUE_PROFILING_ENABLE,
            &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clCreateCommandQueue with error code " << err << std::endl;
        
        // release context
        clReleaseKernel(underlayer_kernel);
        clReleaseKernel(timestep_kernel_two);
        clReleaseKernel(timestep_kernel_one);
        clReleaseKernel(initialize_kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        exit(0);
    }

    // create our buffers
    size_t layer_elements = gridsize.getValue() * gridsize.getValue();
    std::map<std::string, cl_mem> buffers;
    // these are used for alternating between next and last
    buffers["u_one"] = nullptr;
    buffers["u_two"] = nullptr;
    buffers["v_one"] = nullptr;
    buffers["v_two"] = nullptr;
    buffers["underlayer"] = nullptr;
    buffers["keys"] = nullptr;
    buffers["ctrs"] = nullptr;
    for (auto it = buffers.begin(); it != buffers.end(); ++it)
    {
        if (it->first.compare("keys") == 0)
        {
            it->second = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                layer_elements*sizeof(cl_uint2),
                nullptr,
                &err);
        }
        else
        if (it->first.compare("ctrs") == 0)
        {
            it->second = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                layer_elements*sizeof(cl_uint4),
                nullptr,
                &err);
        }
        else
        {
            it->second = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                layer_elements*sizeof(cl_float),
                nullptr,
                &err);
        }

        if (err != CL_SUCCESS)
        {
            std::cout << "Failed clCreateBuffer with error code " << err << std::endl;

            // cycle through and release those allocated buffers
            for (auto it_two = buffers.begin(); it_two != it; ++it_two)
                clReleaseMemObject(it_two->second);

            // release our command queue and context
            clReleaseCommandQueue(command_queue);
            clReleaseKernel(underlayer_kernel);
            clReleaseKernel(timestep_kernel_two);
            clReleaseKernel(timestep_kernel_one);
            clReleaseKernel(initialize_kernel);
            clReleaseProgram(program);
            clReleaseContext(context);
            exit(0);
        }
    }

    // initialize our host random number generator
    std::mt19937_64 gen(seed.getValue());
    std::uniform_int_distribution<cl_uint> rand_uint(0, std::numeric_limits<cl_uint>::max());
    
    // allocate host buffers (going to keep all layers in memory until end)
    std::vector<std::vector<cl_float>> host_buffers(filmsteps.getValue() + 1);
    for (auto & x: host_buffers)
        x.resize(layer_elements);
    std::vector<cl_uint2> keys(layer_elements);
    std::vector<cl_uint4> ctrs(layer_elements);
    
    // initialize our GPU's random number generator(s)
    for (size_t i = 0; i < layer_elements; ++i)
    {
        keys[i].s[0] = (cl_uint)0x9E3779B9;
        keys[i].s[1] = (cl_uint)0xBB67AE85;

        for (size_t j = 0; j < 4; ++j)
            ctrs[i].s[j] = rand_uint(gen);
    }
    err = clEnqueueWriteBuffer(
            command_queue,
            buffers["keys"], // device buffer
            CL_TRUE, // wait for finish
            0, // offset
            layer_elements*sizeof(cl_uint2), // bytes to write
            keys.data(), // host buffer
            0,
            nullptr,
            nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clEnqueueWriteBuffer(1) with error code " << err << std::endl;
        goto end;
    }
    err = clEnqueueWriteBuffer(
            command_queue,
            buffers["ctrs"], // device buffer
            CL_TRUE, // wait for finish
            0, // offset
            layer_elements*sizeof(cl_uint4), // bytes to write
            ctrs.data(), // host buffer
            0,
            nullptr,
            nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clEnqueueWriteBuffer(2) with error code " << err << std::endl;
        goto end;
    }
    
    // initialize first layer on host and device
    for (auto & x: host_buffers[0])
        x = 1;
    err = clEnqueueWriteBuffer(
            command_queue,
            buffers["underlayer"], // device buffer
            CL_TRUE, // wait for finish
            0, // offset
            layer_elements*sizeof(cl_float), // bytes to write
            host_buffers[0].data(), // host buffer
            0,
            nullptr,
            nullptr);
    if (err != CL_SUCCESS)
    {
        std::cout << "Failed clEnqueueWriteBuffer(3) with error code " << err << std::endl;
        goto end;
    }

    // set arguments for our kernels (one set for each swapped state)
    // our initialization kernel always works on u_one and v_one
    err = clSetKernelArg(
            initialize_kernel,
            0, // argument index
            sizeof(cl_mem),
            &buffers["keys"]);
    err |= clSetKernelArg(
            initialize_kernel,
            1, // argument index
            sizeof(cl_mem),
            &buffers["ctrs"]);
    err |= clSetKernelArg(
            initialize_kernel,
            2, // argument index
            sizeof(cl_mem),
            &buffers["u_one"]);
    err |= clSetKernelArg(
            initialize_kernel,
            3, // argument index
            sizeof(cl_mem),
            &buffers["v_one"]);

    // our timestep kernel has two representations: one where the last timestep
    // worked on buffers u_one and v_one, another were the last buffers worked
    // on were u_two and v_two
    err |= clSetKernelArg(
            timestep_kernel_one,
            2, // argument index
            sizeof(cl_mem),
            &buffers["underlayer"]);
    err |= clSetKernelArg(
            timestep_kernel_one,
            3, // argument index
            sizeof(cl_mem),
            &buffers["u_one"]);
    err |= clSetKernelArg(
            timestep_kernel_one,
            4, // argument index
            sizeof(cl_mem),
            &buffers["v_one"]);
    err |= clSetKernelArg(
            timestep_kernel_one,
            5, // argument index
            sizeof(cl_mem),
            &buffers["u_two"]);
    err |= clSetKernelArg(
            timestep_kernel_one,
            6, // argument index
            sizeof(cl_mem),
            &buffers["v_two"]);
    err |= clSetKernelArg(
            timestep_kernel_one,
            7, // argument index
            sizeof(cl_mem),
            &buffers["keys"]);
    err |= clSetKernelArg(
            timestep_kernel_one,
            8, // argument index
            sizeof(cl_mem),
            &buffers["ctrs"]);
    
    err |= clSetKernelArg(
            timestep_kernel_two,
            2, // argument index
            sizeof(cl_mem),
            &buffers["underlayer"]);
    err |= clSetKernelArg(
            timestep_kernel_two,
            3, // argument index
            sizeof(cl_mem),
            &buffers["u_two"]);
    err |= clSetKernelArg(
            timestep_kernel_two,
            4, // argument index
            sizeof(cl_mem),
            &buffers["v_two"]);
    err |= clSetKernelArg(
            timestep_kernel_two,
            5, // argument index
            sizeof(cl_mem),
            &buffers["u_one"]);
    err |= clSetKernelArg(
            timestep_kernel_two,
            6, // argument index
            sizeof(cl_mem),
            &buffers["v_one"]);
    err |= clSetKernelArg(
            timestep_kernel_two,
            7, // argument index
            sizeof(cl_mem),
            &buffers["keys"]);
    err |= clSetKernelArg(
            timestep_kernel_two,
            8, // argument index
            sizeof(cl_mem),
            &buffers["ctrs"]);
   
    // our underlayer kernel creates an underlayer from u_two and v_two 
    err |= clSetKernelArg(
            underlayer_kernel,
            2, // argument index
            sizeof(cl_mem),
            &buffers["underlayer"]);
    err |= clSetKernelArg(
            underlayer_kernel,
            3, // argument index
            sizeof(cl_mem),
            &buffers["keys"]);
    err |= clSetKernelArg(
            underlayer_kernel,
            4, // argument index
            sizeof(cl_mem),
            &buffers["ctrs"]);
    
    if (err != CL_SUCCESS)
    {
        // have fun tracking it down!
        std::cout << "Failed clSetKernelArg with error code " << err << std::endl;
        goto end;
    }

    // some useful information now that we know the run will proceed
    //std::cout
    //    << "Seed     : " << seed.getValue() << std::endl
    //    << "Input    : " << input.getValue() << std::endl
    //    << "Gridsize : " << gridsize.getValue() << std::endl
    //    << "Timesteps: " << timesteps.getValue() << std::endl
    //    << "Filmsteps: " << filmsteps.getValue() << std::endl << std::endl
    //    << "Output   : " << ".\\OUTPUT\\" << output.getValue() << "_XXXXXX.dat" << std::endl
    //    << std::endl;
    

    // finally we can get to the actual calculation: currently we have only 5 things to
    // worry about: timestep_kernel_one/two, underlayer_kernel, initialize_kernel
    // and writting the results to host memory
    {
        // measure elapsed time
        auto start = std::chrono::system_clock::now();
        
        for (cl_int filmstep = 1; filmstep <= filmsteps.getValue(); ++filmstep)
        {
            // set global and local sizes for clEnqueueNDRangeKernel
            size_t global_size [] =
            {(size_t)gridsize.getValue(), (size_t)gridsize.getValue()};

            err = clEnqueueNDRangeKernel(
                    command_queue,
                    initialize_kernel,
                    2, // number of dimensions
                    nullptr, // global id offsets per dimension
                    global_size, // global size per dimension
                    nullptr, // local size (auto if null)
                    0, // number of wait events
                    nullptr, // wait events
                    nullptr); // event for this enqueue

            // perform so many timesteps on current buffers
            cl_int timestep;
            for (timestep = 1; timestep <= timesteps.getValue(); ++timestep)
            {
                // swap kernel execution each cycle
                if (timestep % 2 == 1)
                {
                    clSetKernelArg(
                            timestep_kernel_one,
                            0, // argument index
                            sizeof(cl_int),
                            &timestep);
                    
                    clSetKernelArg(
                            timestep_kernel_one,
                            1, // argument index
                            sizeof(cl_int),
                            &filmstep);
                    
                    // execute timestep kernel one
                    clEnqueueNDRangeKernel(
                            command_queue,
                            timestep_kernel_one,
                            2, // number of dimensions
                            nullptr, // global id offsets per dimension
                            global_size, // global size per dimension
                            nullptr, // local size (auto if null)
                            0, // number of wait events
                            nullptr, // wait events
                            nullptr); // event for this enqueue
                    
                }
                else
                {
                    clSetKernelArg(
                            timestep_kernel_two,
                            0, // argument index
                            sizeof(cl_int),
                            &timestep);
                    
                    clSetKernelArg(
                            timestep_kernel_two,
                            1, // argument index
                            sizeof(cl_int),
                            &filmstep);
                    
                    // execute timestep kernel two
                    clEnqueueNDRangeKernel(
                            command_queue,
                            timestep_kernel_two,
                            2, // number of dimensions
                            nullptr, // global id offsets per dimension
                            global_size, // global size per dimension
                            nullptr, // local size (auto if null)
                            0, // number of wait events
                            nullptr, // wait events
                            nullptr); // event for this enqueue
                }
            }

            // do the underlayer calculation and buffer swaps
            int last_timestep = timestep - 1;
            if (last_timestep % 2 == 1)
            {
                // set our arguments
                clSetKernelArg(
                        underlayer_kernel,
                        0, // argument index
                        sizeof(cl_mem),
                        &buffers["u_two"]);
                clSetKernelArg(
                        underlayer_kernel,
                        1, // argument index
                        sizeof(cl_mem),
                        &buffers["v_two"]);
            }
            else
            {
                // set our arguments
                clSetKernelArg(
                        underlayer_kernel,
                        0, // argument index
                        sizeof(cl_mem),
                        &buffers["u_one"]);
                clSetKernelArg(
                        underlayer_kernel,
                        1, // argument index
                        sizeof(cl_mem),
                        &buffers["v_one"]);
            }

            // execute underlayer kernel one
            err = clEnqueueNDRangeKernel(
                    command_queue,
                    underlayer_kernel,
                    2, // number of dimensions
                    nullptr, // global id offsets per dimension
                    global_size, // global size per dimension
                    nullptr, // local size (auto if null)
                    0, // number of wait events
                    nullptr, // wait events
                    nullptr); // event for this enqueue
                
            // queue up read of underlayer to host buffers
            clEnqueueReadBuffer(
                command_queue,
                buffers["underlayer"],
                CL_FALSE,
                0,
                layer_elements*sizeof(cl_float),
                host_buffers[filmstep].data(),
                0,
                nullptr,
                nullptr);
        }
        // wait for the queue to empty
        clFinish(command_queue);

        // save results to disk
        const int max_cols = gridsize.getValue();
        const int max_rows = gridsize.getValue();
        const int max_layers = (int)host_buffers.size();
        
        // generate the ascii files
        for (int layer = 0; layer < max_layers; ++layer)
        {
            std::ostringstream filename;
            filename
                << "./OUTPUT/"
                << output.getValue()
                << "_"
                << std::setw(6) << std::setfill('0') << std::fixed << layer
                << ".dat";

            // open our file for (over) writing
            std::ofstream asciifile(filename.str(), std::ios::binary);
           
            // write our ascii file
            for (int row = 0; row < max_rows; ++row)
            {
                for (int col = 0; col < max_cols; ++col)
                {
                    asciifile << host_buffers[layer].at(row*max_cols + col);
                    if (col != max_cols)
                        asciifile << " ";
                }
                asciifile << std::endl;
            }
        }

        // generate a complete set of bitmaps with cross sections
        if (bitmaps.getValue())
        {
            // this is our full image
            bitmap_image full_image (max_cols + max_layers, max_rows + max_layers);

            bitmap_image xy_image(max_cols, max_rows); // cols.rows plane, a layer
            bitmap_image xz_image(max_cols, max_layers); // cols.layers plane, a cross section
            bitmap_image yz_image(max_layers, max_rows); // layers.rows plane, a cross section

            // set these to grey while we cycle through layers
            full_image.clear(0);
            xz_image.clear(127);
            yz_image.clear(127);

            // track bitmap number
            int image_num = 0;

            // cycle through layers portion
            for (int z = 0; z < max_layers; ++z, ++image_num)
            {
                for (int y = 0; y < max_rows; ++y)
                {
                    for (int x = 0; x < max_cols; ++x)
                    {
                        const unsigned char intensity = (const unsigned char)(
                                255*host_buffers[z].at(y*max_cols + x));
                        xy_image.set_pixel(x, y, intensity, intensity, intensity);
                    }
                }

                // fill in the full image
                full_image.copy_from(xy_image, max_layers, 0);
                full_image.copy_from(xz_image, max_layers, max_rows);
                full_image.copy_from(yz_image, 0, 0);

                // come up with filename
                std::ostringstream filename;
                filename
                    << "./OUTPUT/"
                    << output.getValue()
                    << "_"
                    << std::setw(6) << std::setfill('0') << std::fixed << image_num
                    << ".bmp";

                // save full image
                full_image.save_image(filename.str());
            }

            // cycle through the xz plane portion
            for (int y = 0; y < max_rows; ++y, ++image_num)
            {
                for (int z = 0; z < max_layers; ++z)
                {
                    for (int x = 0; x < max_cols; ++x)
                    {
                        const unsigned char intensity = (const unsigned char)(
                                255*host_buffers[max_layers - z - 1].at(y*max_cols + x));

                        xz_image.set_pixel(x, z, intensity, intensity, intensity);
                    }
                }
                
                // fill in the full image
                full_image.copy_from(xy_image, max_layers, 0);
                full_image.copy_from(xz_image, max_layers, max_rows);
                full_image.copy_from(yz_image, 0, 0);

                // draw a line to indicate which row we are rendering
                for (int x = 0; x < max_cols; ++x)
                    full_image.set_pixel(max_layers + x, y, 127, 0, 0);
                
                // come up with filename
                std::ostringstream filename;
                filename
                    << "./OUTPUT/"
                    << output.getValue()
                    << "_"
                    << std::setw(6) << std::setfill('0') << std::fixed << image_num
                    << ".bmp";

                // save full image
                full_image.save_image(filename.str());
            }
            
            // draw a final line in the xy_images
            for (int x = 0; x < max_cols; ++x)
                xy_image.set_pixel(x, max_rows - 1, 127, 0, 0);
            
            // cycle through the yz plane portion
            for (int x = max_cols - 1; x >=  0; --x, ++image_num)
            {
                for (int z = 0; z < max_layers; ++z)
                {
                    for (int y = 0; y < max_rows; ++y)
                    {
                        const unsigned char intensity = (const unsigned char)(
                                255*host_buffers[z].at(y*max_cols + x));

                        yz_image.set_pixel(z, y, intensity, intensity, intensity);
                    }
                }
                
                // fill in the full image
                full_image.copy_from(xy_image, max_layers, 0);
                full_image.copy_from(xz_image, max_layers, max_rows);
                full_image.copy_from(yz_image, 0, 0);

                // draw a line to indicate which row we are rendering
                for (int y = 0; y < max_rows; ++y)
                    full_image.set_pixel(max_layers + x, y, 127, 0, 0);
                
                // come up with filename
                std::ostringstream filename;
                filename
                    << "./OUTPUT/"
                    << output.getValue()
                    << "_"
                    << std::setw(6) << std::setfill('0') << std::fixed << image_num
                    << ".bmp";

                // save full image
                full_image.save_image(filename.str());
            }
        }

        // measure elapsed time
        auto finish = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = finish-start;
        
        // tell us how long it took
        //std::cout << "Finished in " << diff.count() << " seconds"  << std::endl;
    }



end:
    // cycle through and release all buffers
    for (auto & x: buffers)
        clReleaseMemObject(x.second);

    // release remainder
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(underlayer_kernel);
    clReleaseKernel(timestep_kernel_two);
    clReleaseKernel(timestep_kernel_one);
    clReleaseKernel(initialize_kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
