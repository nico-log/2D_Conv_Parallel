# 2D Convolution: CPU vs GPU (CUDA)

A CUDA C++ project that applies 3x3 convolution filters (Blur, Edge Detection, Sharpen) to images, comparing a serial CPU implementation with an optimized parallel GPU implementation.

## Features

- Serial CPU convolution for baseline comparison
- CUDA GPU kernel
- Benchmarking with GPU warm-up and averaged timing
- Output validation 
- Supports JPG input via [stb_image](https://github.com/nothings/stb)

The program prints the CPU time, the average GPU time, and the resulting speedup.

## Project history

An earlier, non-optimized version of the GPU kernel is kept as a separate text file for reference.

## Notes

This is a learning project built while getting familiar with CUDA, parallel computing and memory optimization. Feedback and suggestions are welcome.
The project is currently a Work in Progress and is still in the active development phase.
