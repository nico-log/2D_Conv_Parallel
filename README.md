# 2D_Conv_Parallel

## output (T4 GPU Google Colab)
Loading image: test_image.jpg
Image loaded: 3856x2565, Channels: 3

Applying convolution filter (CPU)...
Serial Convolution (CPU) completed in 802.91 ms
Output image saved: output_image_serial.jpg

GPU memory allocation and transfering...
Parallel Convolution (GPU) completed in 1.83092 ms
Output image saved: output_image_parallel.jpg

Speedup (CPU time / GPU time): 438.528x

