
#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convolution(
	const unsigned char* input_image,
	unsigned char* output_image,
	int width,
	int height,
	int channels,
	const float* filter_matrix,
	int filter_dim)
{
	// filter 3*3 => offset from -1 to +1
	int offset = filter_dim / 2;

	// input image (y rows, x cols, c channels)
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			for (int c = 0; c < channels; ++c) {
				float sum = 0.0f;

				//filter (fy rows, fx cols)
				for (int fy = -offset; fy <= offset; ++fy) {
					for (int fx = -offset; fx <= offset; ++fx) {

						int ix = x + fx;
						int iy = y + fy;

						float pixel_value = 0.0f; // default value for out-of-bounds pixels

						if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
							// get the pixel value from the input image array (flattened 3D array)
							int image_index = (iy * width + ix) * channels + c;
							pixel_value = static_cast<float>(input_image[image_index]);
						}

						int filter_index = (fy + offset) * filter_dim + (fx + offset);
						float filter_value = filter_matrix[filter_index];

						sum += pixel_value * filter_value;

					}
				}

				// Clamp the result to [0, 255] and assign it to the output image
				if (sum < 0.0f) sum = 0.0f;
				if (sum > 255.0f) sum = 255.0f;

				// Assign the computed value to the output image array (flattened 3D array)
				int output_index = (y * width + x) * channels + c;
				output_image[output_index] = static_cast<unsigned char>(sum); //cast float to unsigned char
			}
		}
	}
}

__global__ void convolution_kernel(
	unsigned char* input_image,
	unsigned char* output_image,
	int width, int height, int channels,
	const float* filter_matrix, int filter_dim)
{
	// Pixel coordinates for the current thread
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// filter 3*3 => offset from -1 to +1
	int offset = filter_dim / 2;

	// guard against out-of-bounds threads
	if (x < width && y < height) {
		for (int c = 0; c < channels; ++c) {
			float sum = 0.0f;

			//filter (fy rows, fx cols)
			for (int fy = -offset; fy <= offset; ++fy) {
				for (int fx = -offset; fx <= offset; ++fx) {

					int ix = x + fx;
					int iy = y + fy;

					float pixel_value = 0.0f; // default value for out-of-bounds pixels

					if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
						// get the pixel value from the input image array (flattened 3D array)
						int image_index = (iy * width + ix) * channels + c;
						pixel_value = static_cast<float>(input_image[image_index]);
					}

					int filter_index = (fy + offset) * filter_dim + (fx + offset);
					float filter_value = filter_matrix[filter_index];

					sum += pixel_value * filter_value;

				}
			}

			// Clamp the result to [0, 255] and assign it to the output image
			if (sum < 0.0f) sum = 0.0f;
			if (sum > 255.0f) sum = 255.0f;

			// Assign the computed value to the output image array (flattened 3D array)
			int output_index = (y * width + x) * channels + c;
			output_image[output_index] = static_cast<unsigned char>(sum); //cast float to unsigned char
		}
	
	}
}

int main() {

	//FILTERS
	float blur_filter[9] = {
		1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
		1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
		1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f
	};

	float edge_filter[9] = {
		-1.0f, -1.0f, -1.0f,
		-1.0f,  8.0f, -1.0f,
		-1.0f, -1.0f, -1.0f
	};

	float sharpen_filter[9] = {
	 0.0f, -1.0f,  0.0f,
	-1.0f,  5.0f, -1.0f,
	 0.0f, -1.0f,  0.0f
	};

	int filter_dim = 3; // 3x3 filter
	float* filter_matrix = edge_filter; // Choose the filter to apply

	// =====================================================================
	// SERIAL CPU IMPLEMENTATION OF 2D CONVOLUTION
	// =====================================================================
	
	// LOADING IMAGE
	const char* input_image_path = "test_image.jpg";
	const char* output_image_path_serial = "output_image_serial.jpg";

	int width, height, input_channels;
	int output_channels = 3;

	std::cout << "Loading image: " << input_image_path << std::endl;

	unsigned char* input_image = stbi_load(input_image_path, &width, &height, &input_channels, output_channels);

	if (input_image == nullptr) {
		std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
		return 1;
	}

	std::cout << "Image loaded: " << width << "x" << height << ", Channels: " << input_channels << "\n" << std::endl;

	// CONVOLUTION
	// Allocate memory for the output image
	unsigned char* output_image = new unsigned char[width * height * output_channels];

	std::cout << "Applying convolution filter (CPU)..." << std::endl;

	// Measure the time taken for convolution
	auto start_time_cpu = std::chrono::high_resolution_clock::now();

	convolution(input_image, output_image, width, height, output_channels, filter_matrix, filter_dim);

	auto end_time_cpu = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed_time_cpu = end_time_cpu - start_time_cpu;
	std::cout << "Serial Convolution (CPU) completed in " << elapsed_time_cpu.count() << " ms" << std::endl;

	// Save the output image
	int quality = 100;
	stbi_write_jpg(output_image_path_serial, width, height, output_channels, output_image, quality);

	std::cout << "Output image saved: " << output_image_path_serial << "\n" << std::endl;

	// =====================================================================
	// PARALLEL GPU IMPLEMENTATION OF 2D CONVOLUTION (CUDA)
	// =====================================================================

	const char* output_image_path_parallel = "output_image_parallel.jpg";

	std::cout << "GPU memory allocation and transfering..." << std::endl;

	// GPU memory pointers
	unsigned char* d_input_image = nullptr;
	unsigned char* d_output_image = nullptr;
	float* d_filter = nullptr;

	// Allocate GPU memory for input image, output image, and filter
	size_t image_size = width * height * output_channels * sizeof(unsigned char);
	size_t filter_size = 9 * sizeof(float); // 3x3 filter

	cudaMalloc((void**)&d_input_image, image_size);
	cudaMalloc((void**)&d_output_image, image_size);
	cudaMalloc((void**)&d_filter, filter_size);

	// Copy input image and filter to GPU memory
	cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter_matrix, filter_size, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16); // 16 x 16 = 256 threads per block
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y); // Round up to the nearest block

	auto start_time_gpu = std::chrono::high_resolution_clock::now();

	convolution_kernel <<<numBlocks, threadsPerBlock >>> (d_input_image, d_output_image, width, height, output_channels, d_filter, filter_dim);
	cudaDeviceSynchronize();

	auto end_time_gpu  = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed_time_gpu = end_time_gpu - start_time_gpu;
	std::cout << "Parallel Convolution (GPU) completed in " << elapsed_time_gpu.count() << " ms" << std::endl;

	cudaMemcpy(output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);

	// Save the output image
	stbi_write_jpg(output_image_path_parallel, width, height, output_channels, output_image, quality);

	std::cout << "Output image saved: " << output_image_path_parallel << "\n" << std::endl;

	float speedup = elapsed_time_cpu.count() / elapsed_time_gpu.count();
	std::cout << "Speedup (CPU time / GPU time): " << speedup << "x" << std::endl;

	// Free allocated memory
	delete[] output_image;
	stbi_image_free(input_image);
	cudaFree(d_input_image);
	cudaFree(d_output_image);
	cudaFree(d_filter);

	return 0;
}

