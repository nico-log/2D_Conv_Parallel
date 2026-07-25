# 2D Convolution — CUDA Optimization

Advanced Computer Architecture project: a CUDA kernel that applies a 3×3 convolutional filter to an image, built from scratch to explore progressive GPU optimizations and measure their impact on performance.

## Why build it from scratch?

- Learn the basics of CUDA programming and GPU architecture
- Fully customized solution
- No external heavy dependencies (suitable for embedded system applications)

## Features

- Serial CPU baseline implementation
- 5 progressive CUDA GPU implementations (naive → shared memory → constant memory → vectorized → loop unrolled)
- 3x3 filters: **blur**, **sharpen**, **edge**
- Automatic validation of every GPU output against the CPU baseline
- Only external dependency: [`stb_image`](https://github.com/nothings/stb) / `stb_image_write` (header-only, embedded in the project)

## Usage

```bash
./convolution <filter> <iterations> <input_image> <output_image>
```

| Argument | Description |
|---|---|
| `filter` | `blur`, `sharpen`, or `edge` |
| `iterations` | number of benchmark iterations (1–100) |
| `input_image` | path to the input JPEG |
| `output_image` | path to the output JPEG |

## Assumptions

- Fixed 3×3 filter size
- Images loaded as RGB, expanded internally to RGBA (alpha fixed at 255)
- Only JPEG format supported

## Optimization steps and results

| Kernel | Time (ms) | Throughput (MPixel/s) | Speedup |
|---|---|---|---|
| CPU (Baseline) | 297.55 | 42.28 | 1.00x |
| GPU (Naive) | 15.42 | 815.98 | 19.30x |
| GPU (+ Shared Memory) | 10.11 | 1244.06 | 29.42x |
| GPU (+ Constant Memory) | 6.98 | 1802.84 | 42.64x |
| GPU (+ Vectorization) | 4.83 | 2605.57 | 61.63x |
| GPU (+ Loop Unrolling) | 4.81 | 2613.04 | 61.80x |

*Results measured on an NVIDIA GeForce GT 1030 (Pascal, 2GB), edge filter, 100 GPU iterations after warm-up.*

## Future development

- Support for variable filter sizes, and the memory-management implications of larger filter dimensions.

## References

- Notes on Parallel Programming (2025) — Francesco Leporati
- Notes on Machine Learning (2025) — Claudio Cusano
- NVIDIA CUDA C Programming Guide (v4.2)
- [NVIDIA GeForce GT 1030 documentation](https://www.nvidia.com/en-us/geforce/graphics-cards/gt-1030/)

## Notes

This is a learning project built while getting familiar with CUDA, parallel computing and memory optimization. Feedback and suggestions are welcome.
