# 2D Convolution: CPU vs GPU (CUDA)

A CUDA C++ project that applies 3x3 convolution filters (Blur, Edge Detection, Sharpen) to images, comparing a serial CPU implementation with an optimized parallel GPU implementation.

## Features

- Serial CPU convolution for baseline comparison
- CUDA GPU kernel optimized with:
  - **Shared memory tiling** (16x16 blocks with halo cells)
  - **Constant memory** for the filter coefficients
  - **Loop unrolling** on the convolution inner loops
- Benchmarking with GPU warm-up and averaged timing over 100 iterations using `cudaEvent_t`
- Supports JPG/PNG input via [stb_image](https://github.com/nothings/stb)

## Build

Requires the CUDA Toolkit and a CUDA-capable GPU.

```bash
nvcc -O3 main.cu -o convolution
```

## Usage

```bash
./convolution <filter_type> <image_path> <iterations>
```

`filter_type` can be one of: `blur`, `edge`, `sharpen`

Example:

```bash
./convolution edge test_image.jpg 100
```

The program prints the CPU time, the average GPU time, and the resulting speedup. It saves two output images:

- `output_image_serial.jpg` (CPU result)
- `output_image_parallel.jpg` (GPU result)

## How it works

### CPU implementation

A straightforward triple-nested loop over image rows, columns, and channels, applying the 3x3 filter with zero-padding at the borders.

### GPU implementation

Each thread block loads a 16x16 tile of the image, plus a 1-pixel border, into shared memory. This avoids redundant reads from global memory, since neighboring threads need overlapping pixels for the convolution.

The filter coefficients are stored in `__constant__` memory, which is broadcast to all threads in a warp at low latency.

### Benchmark methodology

1. A warm-up kernel launch (with `cudaDeviceSynchronize()`) to exclude one-time initialization costs.
2. The kernel is launched in a loop, timed with `cudaEvent_t`; The number of iterations is decided by the user from command line.
3. The average time per iteration is used to compute the speedup over the CPU implementation.

## Project history

An earlier, non-optimized (naive) version of the GPU kernel is kept as a separate text file for reference.

## Notes

This is a learning project built while getting familiar with CUDA. Feedback and suggestions are welcome.
