#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <string>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define FILTER_DIM 3
#define FILTER_RADIUS 1
#define TILE_SIZE (BLOCK_SIZE + 2 * FILTER_RADIUS)

// Channel management macros
#define RGB_CHANNELS 3
#define RGBA_CHANNELS 4 

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__constant__ float c_filter[FILTER_DIM * FILTER_DIM];	// Constant memory for filter coefficients

// =====================================================================
// SERIAL CPU IMPLEMENTATION
// =====================================================================
void convolution_cpu(
	const unsigned char* input_image,
	unsigned char* output_image,
	int width,
	int height,
	const float* filter)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			for (int c = 0; c < RGB_CHANNELS; ++c) {
				
				float sum = 0.0f;

				for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
					for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {

						int ix = x + fx;
						int iy = y + fy;

						float pixel_value = 0.0f;	// Default to 0 for out-of-bounds pixels

						if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
							int pixel_index = (iy * width + ix) * RGBA_CHANNELS + c;
							pixel_value = static_cast<float>(input_image[pixel_index]);
						}

						int filter_index = (fy + FILTER_RADIUS) * FILTER_DIM + (fx + FILTER_RADIUS);
						float filter_value = filter[filter_index];

						sum += pixel_value * filter_value;	// convolution operation
					}
				}
				// Clamp the result to [0, 255]
				if (sum < 0.0f) sum = 0.0f;
				if (sum > 255.0f) sum = 255.0f;

				int output_index = (y * width + x) * RGBA_CHANNELS + c;
				output_image[output_index] = static_cast<unsigned char>(sum);
			}

			// Set the alpha channel to 255 (fully opaque) for RGBA output
			int alpha_index = (y * width + x) *	RGBA_CHANNELS + RGB_CHANNELS;
			output_image[alpha_index] = 255;
		}
	}
}

// =====================================================================
// PARALLEL GPU IMPLEMENTATION (NAIVE)
// =====================================================================
__global__ void convolution_kernel_naive(
	unsigned char* input_image,
	unsigned char* output_image,
	int width, int height,
	const float* filter_matrix)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		for (int c = 0; c < RGB_CHANNELS; ++c) {
			float sum = 0.0f;

			for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
				for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {

					int ix = x + fx;
					int iy = y + fy;

					float pixel_value = 0.0f;

					if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
						int image_index = (iy * width + ix) * RGBA_CHANNELS + c;
						pixel_value = static_cast<float>(input_image[image_index]);
					}

					int filter_index = (fy + FILTER_RADIUS) * FILTER_DIM + (fx + FILTER_RADIUS);
					float filter_value = filter_matrix[filter_index];

					sum += pixel_value * filter_value;
				}
			}

			if (sum < 0.0f) sum = 0.0f;
			if (sum > 255.0f) sum = 255.0f;

			int output_index = (y * width + x) * RGBA_CHANNELS + c;
			output_image[output_index] = static_cast<unsigned char>(sum);
		}

		// Set alpha channel
		int alpha_index = (y * width + x) * RGBA_CHANNELS + RGB_CHANNELS;
		output_image[alpha_index] = 255;
	}
}

// =====================================================================
// PARALLEL GPU IMPLEMENTATION (SHARED MEMORY - BASIC)
// =====================================================================
__global__ void convolution_kernel_shared(
	unsigned char* input_image,
	unsigned char* output_image,
	int width, int height,
	const float* filter_matrix)
{
	__shared__ float shared_tile[TILE_SIZE][TILE_SIZE][RGB_CHANNELS];

	int t_id = threadIdx.y * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * blockDim.y;
	int num_pixels = TILE_SIZE * TILE_SIZE;

	int origin_x = blockIdx.x * BLOCK_SIZE - FILTER_RADIUS;
	int origin_y = blockIdx.y * BLOCK_SIZE - FILTER_RADIUS;

	for (int i = t_id; i < num_pixels; i += num_threads) {
		int tile_y = i / TILE_SIZE;
		int tile_x = i % TILE_SIZE;

		int image_x = origin_x + tile_x;
		int image_y = origin_y + tile_y;

		for (int c = 0; c < RGB_CHANNELS; ++c) {
			shared_tile[tile_y][tile_x][c] = 0.0f;

			if (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height) {
				int image_index = (image_y * width + image_x) * RGBA_CHANNELS + c;
				shared_tile[tile_y][tile_x][c] = input_image[image_index];
			}
		}
	}

	__syncthreads();

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		for (int c = 0; c < RGB_CHANNELS; ++c) {
			float sum = 0.0f;

			for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
				for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {

					float pixel_value = shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS][c];

					int filter_index = (fy + FILTER_RADIUS) * FILTER_DIM + (fx + FILTER_RADIUS);
					float filter_value = filter_matrix[filter_index];

					sum += pixel_value * filter_value;
				}
			}

			if (sum < 0.0f) sum = 0.0f;
			if (sum > 255.0f) sum = 255.0f;

			int output_index = (y * width + x) * RGBA_CHANNELS + c;
			output_image[output_index] = static_cast<unsigned char>(sum);
		}

		// Set alpha channel
		int alpha_index = (y * width + x) * RGBA_CHANNELS + RGB_CHANNELS;
		output_image[alpha_index] = 255;
	}
}

// =====================================================================
// PARALLEL GPU IMPLEMENTATION (SHARED MEMORY + CONSTANT MEMORY)
// =====================================================================
__global__ void convolution_kernel_shared_constant(
	unsigned char* input_image,
	unsigned char* output_image,
	int width, int height)
{
	__shared__ float shared_tile[TILE_SIZE][TILE_SIZE][RGB_CHANNELS];

	int t_id = threadIdx.y * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * blockDim.y;
	int num_pixels = TILE_SIZE * TILE_SIZE;

	int origin_x = blockIdx.x * BLOCK_SIZE - FILTER_RADIUS;
	int origin_y = blockIdx.y * BLOCK_SIZE - FILTER_RADIUS;

	for (int i = t_id; i < num_pixels; i += num_threads) {
		int tile_y = i / TILE_SIZE;
		int tile_x = i % TILE_SIZE;

		int image_x = origin_x + tile_x;
		int image_y = origin_y + tile_y;

		for (int c = 0; c < RGB_CHANNELS; ++c) {
			shared_tile[tile_y][tile_x][c] = 0.0f;

			if (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height) {
				int image_index = (image_y * width + image_x) * RGBA_CHANNELS + c;
				shared_tile[tile_y][tile_x][c] = input_image[image_index];
			}
		}
	}

	__syncthreads();

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		for (int c = 0; c < RGB_CHANNELS; ++c) {
			float sum = 0.0f;

			for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
				for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {

					float pixel_value = shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS][c];

					float filter_value = c_filter[(fy + FILTER_RADIUS) * FILTER_DIM + (fx + FILTER_RADIUS)];

					sum += pixel_value * filter_value;
				}
			}

			if (sum < 0.0f) sum = 0.0f;
			if (sum > 255.0f) sum = 255.0f;

			int output_index = (y * width + x) * RGBA_CHANNELS + c;
			output_image[output_index] = static_cast<unsigned char>(sum);
		}

		// Set alpha channel
		int alpha_index = (y * width + x) * RGBA_CHANNELS + RGB_CHANNELS;
		output_image[alpha_index] = 255;
	}
}
// =====================================================================
// PARALLEL GPU IMPLEMENTATION (SHARED MEMORY + CONSTANT MEMORY + VECTORIZED)
// =====================================================================
__global__ void convolution_kernel_constant_vector(
	uchar4* __restrict__ input_image,
	uchar4* __restrict__ output_image,
	int width, int height)
{
	__shared__ float4 shared_tile[TILE_SIZE][TILE_SIZE];

	int t_id = threadIdx.y * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * blockDim.y;
	int num_pixels = TILE_SIZE * TILE_SIZE;

	int origin_x = blockIdx.x * BLOCK_SIZE - FILTER_RADIUS;
	int origin_y = blockIdx.y * BLOCK_SIZE - FILTER_RADIUS;

	for (int i = t_id; i < num_pixels; i += num_threads) {
		int tile_y = i / TILE_SIZE;
		int tile_x = i % TILE_SIZE;

		int image_x = origin_x + tile_x;
		int image_y = origin_y + tile_y;
		int image_index = image_y * width + image_x;

		uchar4 global_pixel = make_uchar4(0, 0, 0, 255);

		if (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height) {
			global_pixel = input_image[image_index];
		}

		shared_tile[tile_y][tile_x] = make_float4(
			(float)global_pixel.x,
			(float)global_pixel.y,
			(float)global_pixel.z,
			255
		);
	}

	__syncthreads();

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		float sum_r = 0.0f;
		float sum_g = 0.0f;
		float sum_b = 0.0f;

		for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
			for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {

				float filter_value = c_filter[(fy + FILTER_RADIUS) * FILTER_DIM + (fx + FILTER_RADIUS)];

				sum_r += shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS].x * filter_value;
				sum_g += shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS].y * filter_value;
				sum_b += shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS].z * filter_value;
			}
		}

		sum_r = fminf(fmaxf(sum_r, 0.0f), 255.0f);
		sum_g = fminf(fmaxf(sum_g, 0.0f), 255.0f);
		sum_b = fminf(fmaxf(sum_b, 0.0f), 255.0f);

		output_image[y * width + x] = make_uchar4(
			static_cast<unsigned char>(sum_r),
			static_cast<unsigned char>(sum_g),
			static_cast<unsigned char>(sum_b),
			255
		);
	}
}
// =====================================================================
// PARALLEL GPU IMPLEMENTATION (SHARED MEMORY + CONSTANT MEMORY + VECTORIZED + UNROLL)
// =====================================================================
__global__ void convolution_kernel_optimized(
	uchar4* __restrict__ input_image,
	uchar4* __restrict__ output_image,
	int width, int height)
{
	__shared__ float4 shared_tile[TILE_SIZE][TILE_SIZE];

	int t_id = threadIdx.y * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * blockDim.y;
	int num_pixels = TILE_SIZE * TILE_SIZE;

	int origin_x = blockIdx.x * BLOCK_SIZE - FILTER_RADIUS;
	int origin_y = blockIdx.y * BLOCK_SIZE - FILTER_RADIUS;

	for (int i = t_id; i < num_pixels; i += num_threads) {
		int tile_y = i / TILE_SIZE;
		int tile_x = i % TILE_SIZE;

		int image_x = origin_x + tile_x;
		int image_y = origin_y + tile_y;
		int image_index = image_y * width + image_x;

		uchar4 global_pixel = make_uchar4(0, 0, 0, 255);

		if (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height) {
			global_pixel = input_image[image_index];
		}

		shared_tile[tile_y][tile_x] = make_float4(
			(float)global_pixel.x,
			(float)global_pixel.y,
			(float)global_pixel.z,
			255
		);
	}

	__syncthreads();

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		float sum_r = 0.0f;
		float sum_g = 0.0f;
		float sum_b = 0.0f;

#pragma unroll
		for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; ++fy) {
#pragma unroll
			for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {

				float filter_value = c_filter[(fy + FILTER_RADIUS) * FILTER_DIM + (fx + FILTER_RADIUS)];

				sum_r += shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS].x * filter_value;
				sum_g += shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS].y * filter_value;
				sum_b += shared_tile[threadIdx.y + fy + FILTER_RADIUS][threadIdx.x + fx + FILTER_RADIUS].z * filter_value;
			}
		}

		sum_r = fminf(fmaxf(sum_r, 0.0f), 255.0f);
		sum_g = fminf(fmaxf(sum_g, 0.0f), 255.0f);
		sum_b = fminf(fmaxf(sum_b, 0.0f), 255.0f);

		output_image[y * width + x] = make_uchar4(
			static_cast<unsigned char>(sum_r),
			static_cast<unsigned char>(sum_g),
			static_cast<unsigned char>(sum_b),
			255
		);
	}
}

// =====================================================================
// UTILITIES
// =====================================================================
int validation(unsigned char* output_image_cpu, unsigned char* output_image_gpu, int width, int height, int tollerance) {
	bool is_valid = true;
	int total_pixels = width * height * RGBA_CHANNELS;

	for (int i = 0; i < total_pixels; ++i) {
		int diff = std::abs(static_cast<int>(output_image_cpu[i]) - static_cast<int>(output_image_gpu[i]));

		if (diff > tollerance) {
			is_valid = false;
			std::cerr << "Validation failed at index " << i << ": CPU value = "
				<< static_cast<int>(output_image_cpu[i]) << ", GPU value = "
				<< static_cast<int>(output_image_gpu[i]) << ", difference = " << diff << std::endl;
			// Stop after first few errors to avoid flooding console
			break;
		}
	}

	if (!is_valid) {
		std::cerr << " -> CPU and GPU results do not match within the specified tolerance of " << tollerance << std::endl;
		return 1;
	}
	return 0;
}

double get_throughput_mpixels(int width, int height, double time_ms) {
	double time_sec = time_ms / 1000.0;
	double total_pixels = static_cast<double>(width) * height;

	return (total_pixels / 1e6) / time_sec;
}

int main(int argc, char** argv) {

	if (argc < 5) {
		std::cerr << "Error: missing arguments" << std::endl;
		std::cerr << "Usage: " << argv[0] << " <filter_type> <iterations> <input_image_path> <output_image_path>" << std::endl;
		std::cerr << "filter_type: blur, edge, sharpen" << std::endl;
		return 1;
	}

	const int iterations = std::stoi(argv[2]);

	if (iterations <= 0 || iterations > 100) {
		std::cerr << "Error: iterations must be between 0 and 100" << std::endl;
		return 1;
	}

	std::string filter_type = argv[1];
	const char* input_image_path = argv[3];
	const char* output_image_path = argv[4];

	const float* active_filter = nullptr;

	float blur_filter[FILTER_DIM * FILTER_DIM] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,	1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
	float edge_filter[FILTER_DIM * FILTER_DIM] = { -1.0f, -1.0f, -1.0f, -1.0f, 8.0f, -1.0f, -1.0f, -1.0f, -1.0f };
	float sharpen_filter[FILTER_DIM * FILTER_DIM] = { 0.0f, -1.0f, 0.0f, -1.0f, 5.0f, -1.0f, 0.0f, -1.0f, 0.0f };

	if (filter_type == "blur") { active_filter = blur_filter; }
	else if (filter_type == "edge") { active_filter = edge_filter; }
	else if (filter_type == "sharpen") { active_filter = sharpen_filter; }
	else {
		std::cerr << "Error: unknown filter_type '" << filter_type << "'" << std::endl;
		return 1;
	}

	// LOADING IMAGE
	int width, height, input_channels;
	unsigned char* input_image = stbi_load(input_image_path, &width, &height, &input_channels, RGBA_CHANNELS);

	if (input_image == nullptr) {
		std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
		return 1;
	}

	std::cout << "================================================================" << std::endl;
	std::cout << "2D CONVOLUTION PERFORMANCE COMPARISON" << std::endl;
	std::cout << input_image_path << " Loaded: " << width << "x" << height << ", Channels: " << input_channels << " -> " << RGBA_CHANNELS << " (RGBA)" << std::endl;
	std::cout << "================================================================\n" << std::endl;

	size_t image_size = width * height * RGBA_CHANNELS * sizeof(unsigned char);
	size_t filter_size = FILTER_DIM * FILTER_DIM * sizeof(float);

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	// Host Output Buffers for Validation
	unsigned char* output_image_cpu = new unsigned char[width * height * RGBA_CHANNELS];
	unsigned char* output_image_naive = new unsigned char[width * height * RGBA_CHANNELS];
	unsigned char* output_image_shared = new unsigned char[width * height * RGBA_CHANNELS];
	unsigned char* output_image_shared_constant = new unsigned char[width * height * RGBA_CHANNELS];
	unsigned char* output_image_shared_constant_vector = new unsigned char[width * height * RGBA_CHANNELS];
	unsigned char* output_image_opt = new unsigned char[width * height * RGBA_CHANNELS];


	// ===================== CPU BASELINE =====================
	
		std::cout << "SERIAL CONVOLUTION (CPU)" << std::endl;
		// Warmup
		convolution_cpu(input_image, output_image_cpu, width, height, active_filter);
		auto start_time_cpu = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < iterations; ++i) {
			convolution_cpu(input_image, output_image_cpu, width, height, active_filter);
		}
		auto end_time_cpu = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_time_cpu = end_time_cpu - start_time_cpu;
		double time_cpu = elapsed_time_cpu.count()/iterations;

		std::cout << " - Time:\t" << time_cpu << " ms" << std::endl;
		std::cout << " - Performance:\t" << get_throughput_mpixels(width, height, time_cpu) << " MPixels/s" << std::endl;
		std::cout << "________________________________________________________________\n" << std::endl;
	

	// ===================== 1. GPU NAIVE =====================
	{
		std::cout << "PARALLEL CONVOLUTION (GPU NAIVE)" << std::endl;
		unsigned char* d_input_image = nullptr;
		unsigned char* d_output_image = nullptr;
		float* d_filter = nullptr;

		CUDA_CHECK(cudaMalloc((void**)&d_input_image, image_size));
		CUDA_CHECK(cudaMalloc((void**)&d_output_image, image_size));
		CUDA_CHECK(cudaMalloc((void**)&d_filter, filter_size));

		CUDA_CHECK(cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_filter, active_filter, filter_size, cudaMemcpyHostToDevice));

		// Warmup
		convolution_kernel_naive << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height, d_filter);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(start));
		for (int i = 0; i < iterations; ++i) {
			convolution_kernel_naive << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height, d_filter);
		}
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));

		float time_naive = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&time_naive, start, stop));
		time_naive /= iterations;

		CUDA_CHECK(cudaMemcpy(output_image_naive, d_output_image, image_size, cudaMemcpyDeviceToHost));

		std::cout << " - Avg Time:\t" << time_naive << " ms" << std::endl;
		std::cout << " - Performance:\t" << get_throughput_mpixels(width, height, time_naive) << " MPixels/s" << std::endl;
		std::cout << " - Speedup:\t" << time_cpu / time_naive << "x" << std::endl;
		std::cout << "________________________________________________________________\n" << std::endl;

		CUDA_CHECK(cudaFree(d_input_image));
		CUDA_CHECK(cudaFree(d_output_image));
		CUDA_CHECK(cudaFree(d_filter));
	}


	// ===================== 2. GPU SHARED BASIC =====================
	{
		std::cout << "PARALLEL CONVOLUTION (GPU SHARED - BASIC)" << std::endl;
		unsigned char* d_input_image = nullptr;
		unsigned char* d_output_image = nullptr;
		float* d_filter = nullptr;

		CUDA_CHECK(cudaMalloc((void**)&d_input_image, image_size));
		CUDA_CHECK(cudaMalloc((void**)&d_output_image, image_size));
		CUDA_CHECK(cudaMalloc((void**)&d_filter, filter_size));

		CUDA_CHECK(cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_filter, active_filter, filter_size, cudaMemcpyHostToDevice));

		convolution_kernel_shared << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height, d_filter);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(start));
		for (int i = 0; i < iterations; ++i) {
			convolution_kernel_shared << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height, d_filter);
		}
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));

		float time_shared = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&time_shared, start, stop));
		time_shared /= iterations;

		CUDA_CHECK(cudaMemcpy(output_image_shared, d_output_image, image_size, cudaMemcpyDeviceToHost));

		std::cout << " - Avg Time:\t" << time_shared << " ms" << std::endl;
		std::cout << " - Performance:\t" << get_throughput_mpixels(width, height, time_shared) << " MPixels/s" << std::endl;
		std::cout << " - Speedup:\t" << time_cpu / time_shared << "x" << std::endl;
		std::cout << "________________________________________________________________\n" << std::endl;

		CUDA_CHECK(cudaFree(d_input_image));
		CUDA_CHECK(cudaFree(d_output_image));
		CUDA_CHECK(cudaFree(d_filter));
	}

	// ===================== 3. GPU SHARED + CONSTANT MEMORY =====================
	{
		std::cout << "PARALLEL CONVOLUTION (GPU SHARED + CONSTANT MEMORY)" << std::endl;
		unsigned char* d_input_image = nullptr;
		unsigned char* d_output_image = nullptr;

		CUDA_CHECK(cudaMalloc((void**)&d_input_image, image_size));
		CUDA_CHECK(cudaMalloc((void**)&d_output_image, image_size));

		CUDA_CHECK(cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol(c_filter, active_filter, filter_size));

		convolution_kernel_shared_constant << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(start));
		for (int i = 0; i < iterations; ++i) {
			convolution_kernel_shared_constant << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height);
		}
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));

		float time_shared = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&time_shared, start, stop));
		time_shared /= iterations;

		CUDA_CHECK(cudaMemcpy(output_image_shared_constant, d_output_image, image_size, cudaMemcpyDeviceToHost));

		std::cout << " - Avg Time:\t" << time_shared << " ms" << std::endl;
		std::cout << " - Performance:\t" << get_throughput_mpixels(width, height, time_shared) << " MPixels/s" << std::endl;
		std::cout << " - Speedup:\t" << time_cpu / time_shared << "x" << std::endl;
		std::cout << "________________________________________________________________\n" << std::endl;

		CUDA_CHECK(cudaFree(d_input_image));
		CUDA_CHECK(cudaFree(d_output_image));
	}

	// ===================== 4. GPU SHARED + CONSTANT MEMORY + VECTORIZED =====================
	{
		std::cout << "PARALLEL CONVOLUTION (GPU SHARED + CONSTANT MEMORY + VECTORIZED)" << std::endl;
		uchar4* d_input_image = nullptr;
		uchar4* d_output_image = nullptr;

		CUDA_CHECK(cudaMalloc((void**)&d_input_image, image_size));
		CUDA_CHECK(cudaMalloc((void**)&d_output_image, image_size));

		CUDA_CHECK(cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol(c_filter, active_filter, filter_size));

		convolution_kernel_constant_vector << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(start));
		for (int i = 0; i < iterations; ++i) {
			convolution_kernel_constant_vector << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height);
		}
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));

		float time_opt = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&time_opt, start, stop));
		time_opt /= iterations;

		CUDA_CHECK(cudaMemcpy(output_image_shared_constant_vector, d_output_image, image_size, cudaMemcpyDeviceToHost));

		std::cout << " - Avg Time:\t" << time_opt << " ms" << std::endl;
		std::cout << " - Performance:\t" << get_throughput_mpixels(width, height, time_opt) << " MPixels/s" << std::endl;
		std::cout << " - Speedup:\t" << time_cpu / time_opt << "x" << std::endl;
		std::cout << "________________________________________________________________\n" << std::endl;

		CUDA_CHECK(cudaFree(d_input_image));
		CUDA_CHECK(cudaFree(d_output_image));
	}

	// ===================== 5. GPU SHARED + CONSTANT MEMORY + VECTORIZED + UNROLLED =====================
	{
		std::cout << "PARALLEL CONVOLUTION (GPU SHARED + CONSTANT MEMORY + VECTORIZED + UNROLLED)" << std::endl;
		uchar4* d_input_image = nullptr;
		uchar4* d_output_image = nullptr;

		CUDA_CHECK(cudaMalloc((void**)&d_input_image, image_size));
		CUDA_CHECK(cudaMalloc((void**)&d_output_image, image_size));

		CUDA_CHECK(cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol(c_filter, active_filter, filter_size));

		convolution_kernel_optimized << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(start));
		for (int i = 0; i < iterations; ++i) {
			convolution_kernel_optimized << <numBlocks, threadsPerBlock >> > (d_input_image, d_output_image, width, height);
		}
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));

		float time_opt = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&time_opt, start, stop));
		time_opt /= iterations;

		CUDA_CHECK(cudaMemcpy(output_image_opt, d_output_image, image_size, cudaMemcpyDeviceToHost));

		std::cout << " - Avg Time:\t" << time_opt << " ms" << std::endl;
		std::cout << " - Performance:\t" << get_throughput_mpixels(width, height, time_opt) << " MPixels/s" << std::endl;
		std::cout << " - Speedup:\t" << time_cpu / time_opt << "x" << std::endl;
		std::cout << "________________________________________________________________\n" << std::endl;

		CUDA_CHECK(cudaFree(d_input_image));
		CUDA_CHECK(cudaFree(d_output_image));
	}

	// VALIDATION AND EXPORT
	bool is_valid = true;
	int quality = 100;

	if (validation(output_image_cpu, output_image_naive, width, height, 1) != 0) { std::cerr << "Naive GPU Failed!" << std::endl; is_valid = false; }
	if (validation(output_image_cpu, output_image_shared, width, height, 1) != 0) { std::cerr << "Shared Basic GPU Failed!" << std::endl; is_valid = false; }
	if (validation(output_image_cpu, output_image_shared_constant, width, height, 1) != 0) { std::cerr << "Shared Constant GPU Failed!" << std::endl; is_valid = false; }
	if (validation(output_image_cpu, output_image_shared_constant_vector, width, height, 1) != 0) { std::cerr << "Shared Constant Vectorized GPU Failed!" << std::endl; is_valid = false; }
	if (validation(output_image_cpu, output_image_opt, width, height, 1) != 0) { std::cerr << "Shared Optimized GPU Failed!" << std::endl; is_valid = false; }

	if (is_valid) {
		
		std::cout << "Success! All GPU versions match CPU baseline." << std::endl;
		std::cout << "Final image saved to: " << output_image_path << std::endl;

		if (!stbi_write_jpg(output_image_path, width, height, RGBA_CHANNELS, output_image_opt, quality)) {
			std::cerr << "Failed to save optimized GPU image!" << std::endl;
		}
		
	}
	else {

		std::cerr << "\nValidation failed! Debug images have been saved separately:" << std::endl;
		std::cout << "0. output_image_serial.jpg" << std::endl;
		std::cout << "1. output_image_naive.jpg" << std::endl;
		std::cout << "2. output_image_shared.jpg" << std::endl;
		std::cout << "3. output_image_shared_constant.jpg" << std::endl;
		std::cout << "4. output_image_shared_constant_vector.jpg" << std::endl;
		std::cout << "5. output_image_opt.jpg" << std::endl;

		if (!stbi_write_jpg("output_image_serial.jpg", width, height, RGBA_CHANNELS, output_image_cpu, quality)) {
			std::cerr << "Failed to save serial CPU image!" << std::endl;
		}
		if (!stbi_write_jpg("output_image_naive.jpg", width, height, RGBA_CHANNELS, output_image_naive, quality)) {
			std::cerr << "Failed to save naive GPU image!" << std::endl;
		}
		if (!stbi_write_jpg("output_image_shared.jpg", width, height, RGBA_CHANNELS, output_image_shared, quality)) {
			std::cerr << "Failed to save shared GPU image!" << std::endl;
		}
		if (!stbi_write_jpg("output_image_shared_constant.jpg", width, height, RGBA_CHANNELS, output_image_shared_constant, quality)) {
			std::cerr << "Failed to save shared constant GPU image!" << std::endl;
		}
		if (!stbi_write_jpg("output_image_shared_constant_vector.jpg", width, height, RGBA_CHANNELS, output_image_shared_constant_vector, quality)) {
			std::cerr << "Failed to save shared constant vectorized GPU image!" << std::endl;
		}
		if (!stbi_write_jpg("output_image_opt.jpg", width, height, RGBA_CHANNELS, output_image_opt, quality)) {
			std::cerr << "Failed to save optimized GPU image!" << std::endl;
		}
	}

	// FREE MEMORY
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	delete[] output_image_cpu;
	delete[] output_image_naive;
	delete[] output_image_shared;
	delete[] output_image_shared_constant;
	delete[] output_image_shared_constant_vector;
	delete[] output_image_opt;

	stbi_image_free(input_image);

	return 0;
}
