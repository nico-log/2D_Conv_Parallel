# 2D Convolution — CUDA Optimization

This project implements a CUDA kernel that applies a 3×3 convolutional filter to an image, comparing a serial CPU baseline with five progressively optimized GPU implementations.

## Overview

The goal is to build a 2D convolution from scratch and show, step by step, how different CUDA optimization techniques affect performance. Starting from a naive GPU kernel, each version adds one optimization on top of the previous one:

0. **CPU implementation** (baseline)
1. **GPU naive** — global memory only
2. **GPU + shared memory**
3. **GPU + shared memory + constant memory**
4. **GPU + shared memory + constant memory + vectorized accesses**
5. **GPU + shared memory + constant memory + vectorized accesses + loop unrolling**

Even though optimized convolution libraries already exist, building this kernel from scratch has educational value and allows customizing or combining filters without depending on large external libraries — useful for embedded systems and real-time computer vision applications.

## How convolution works

A 3×3 filter (kernel) slides over the input image. At each position, the filter coefficients are multiplied with the corresponding pixel values, and the results are summed to produce one output pixel. The same spatial filter is applied independently to each color channel (R, G, B).

Since sliding a k×k filter over an image normally shrinks the output, this project uses **zero-padding** (padding = 1 for a 3×3 filter) to keep the output the same size as the input. Note that zero-padding creates an artificial high-contrast edge at the image border, which edge/sharpen filters pick up as a bright frame around the output.

## Filters

Three fixed 3×3 filters are implemented:

- **Blur** — removes noise and detail
- **Sharpen** — boosts edge contrast
- **Edge** — highlights object boundaries

## Usage

```bash
./convolution <filter> <iterations> <input_image_path> <output_image_path>
```

- `filter`: `edge`, `sharpen`, or `blur`
- `iterations`: number of GPU benchmarking iterations
- `input_image_path`: path to the input image
- `output_image_path`: path to save the output image

The program prints the CPU execution time and throughput (in MPixel/s), then for each GPU implementation the average execution time, throughput, and speed-up relative to the CPU.

Image I/O uses the header-only libraries [`stb_image.h` and `stb_image_write.h`](https://github.com/nothings/stb). Images are loaded as RGBA (an alpha channel is added even though only RGB is used for the convolution), which enables the vectorized memory access optimization described below.

A validation function compares each GPU output against the CPU output pixel by pixel, with a tolerance of 1 intensity level to account for floating-point rounding differences. If all GPU outputs pass validation, only one output image is saved; otherwise, all outputs are saved for inspection.

## CPU implementation

A straightforward serial implementation: nested loops over width, height, and channel, with two more nested loops over the 3×3 filter for each output pixel. Out-of-bounds pixels are treated as zero (implicit zero-padding). The result is clamped to [0, 255].

## GPU implementation

All kernels use 16×16 thread blocks (256 threads per block), a natural fit for 2D image data. The grid size is computed so that all pixels are covered even when the image dimensions aren't multiples of the block size.

### Naive
Each thread computes one output pixel, reading directly from global memory. This alone gives a large speed-up over the CPU, but the same input pixels get re-read redundantly by many threads, and global memory has relatively high latency.

### Shared memory
Each block first cooperatively loads its 16×16 region of pixels — plus a 1-pixel halo on each side, since border threads need neighboring pixels — into an 18×18 shared memory tile. After a `__syncthreads()` barrier, the convolution is computed from this fast, on-chip memory instead of global memory.

### Constant memory
The filter matrix (only 36 bytes) is moved from global to constant memory. Since every thread in a warp reads the same filter coefficient at the same time, constant memory can broadcast that value to the whole warp in a single request, avoiding redundant global memory traffic.

### Vectorized accesses
Instead of three separate 1-byte reads for R, G, and B, each thread reads one full RGBA pixel at once using CUDA's `uchar4` type. This improves access granularity and memory coalescing, at the cost of loading one extra (unused) alpha byte per pixel.

### Loop unrolling
The inner loop over the 3×3 filter is unrolled with `#pragma unroll`. Since the loop bounds are compile-time constants and the loop is very small, `nvcc` likely already performs much of this optimization on its own, so the expected additional gain is small.

## Benchmarking methodology

- **Execution time** is measured with `std::chrono` on the CPU and with CUDA Events (`cudaEventRecord`) on the GPU, since kernel launches are asynchronous with respect to the host.
- **Throughput** (MPixel/s) is also reported, since it normalizes for image resolution and is a meaningful metric for memory-bound workloads.
- A **warm-up run** (not measured) is performed before timing, to avoid counting driver initialization and first-launch overhead.
- GPU kernels are run for multiple iterations (100) and averaged, since individual kernel runs are very short and sensitive to system noise; the CPU is timed with a single iteration, since its execution time is long enough that this isn't a concern.

Reported GPU times cover kernel execution only — they exclude host-device transfers, memory allocation, and other setup costs.

## Results

Test setup: NVIDIA GeForce GT 1030 (2 GB GDDR5, 64-bit bus), driver 560.64, CUDA 12.6. Input image: 4344×2896, RGBA.

**Edge filter**

| Kernel | Time (ms) | Throughput (MP/s) | Speedup |
|---|---|---|---|
| CPU (baseline) | 297.55 | 42.28 | 1.00x |
| GPU naive | 15.42 | 815.98 | 19.30x |
| GPU + shared mem | 10.11 | 1244.06 | 29.42x |
| GPU + constant | 6.98 | 1802.84 | 42.64x |
| GPU + vectorized | 4.83 | 2605.57 | 61.63x |
| GPU + unrolled | 4.81 | 2613.04 | **61.80x** |

The blur and sharpen filters show almost identical numbers, since all three filters share the same computational structure (same filter size, channel count, and memory access pattern — only the coefficients differ).

The fully optimized kernel is roughly **61× faster** than the serial CPU baseline.

## Roofline analysis

An analytical roofline model was used to check whether the kernel is compute-bound or memory-bound. Based on estimated data transferred per block (an 18×18 RGBA tile read, a 16×16 RGBA tile written) and the operation count per pixel, the kernel's arithmetic intensity is approximately:

```
AI_kernel ≈ 6 FLOP/Byte
```

The device's ridge point (peak compute ÷ peak bandwidth) is about 23.5 FLOP/Byte. Since 6 < 23.5, the kernel is **memory-bound**, which matches the observed benefit of the shared/constant memory and vectorization optimizations.

Using the kernel's measured execution time, the estimated achieved performance is about 142 GFLOP/s, against a theoretical roof of 288 GFLOP/s (~49%). The gap is mainly explained by the unused alpha channel being transferred, possible cache hits reducing actual DRAM traffic below the model's estimate, and non-arithmetic instructions (address calculations, boundary checks, clamping) that aren't counted as FLOPs.

Note: this is an analytical estimate, not a profiler-based measurement — see "Future work" below.

## Future work

- Validate the analytical roofline model with a real profiler (e.g. NVIDIA Nsight Compute).
- Re-run the benchmark on more modern GPU hardware.
- Compare against an established convolution/image-processing library instead of only a custom CPU baseline.
- Test with larger filter sizes (5×5, 7×7) — larger filters increase the ratio of halo pixels to active pixels in shared memory, which could hurt performance.
- Test different block sizes — larger blocks reduce the halo-to-active ratio but increase shared memory usage per block.
- Test on a range of image resolutions.

## References

- Notes on Parallel Programming — Francesco Leporati
- Notes on Advanced Computer Architecture — Emanuele Torti
- Notes on Machine Learning — Claudio Cusano
- CUDA Programming Guide, NVIDIA Corporation
- [NVIDIA GeForce GT 1030 documentation](https://www.nvidia.com/en-us/geforce/graphics-cards/gt-1030/)
- [TechPowerUp GPU Database — GeForce GT 1030](https://www.techpowerup.com/gpu-specs/geforce-gt-1030.c2954)
- Afif, Said, Atri — "Efficient 2D Convolution Filters Implementations on Graphics Processing Unit Using NVIDIA CUDA", IJIGSP, Vol.10, No.8, 2018
- [stb (image I/O library)](https://github.com/nothings/stb)
- Test image: Sebastian Pena Lambarri — Whaleshark, Maldives

---

*This is a learning project built while getting familiar with CUDA, parallel computing and memory optimization. Feedback and suggestions are welcome.*
