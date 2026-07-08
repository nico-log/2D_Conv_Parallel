# 2D_Conv_Parallel

## output (T4 GPU Google Colab)
Image loaded: 3856x2565, Channels: 3

1. Serial Convolution (CPU) completed in 720.524 ms
2. Parallel Convolution (GPU) completed in 2.5464 ms
3. Parallel Convolution (GPU - Shared Memory) completed in 1.76333 ms

Speedup (CPU time / GPU time): 282.958x
Speedup (GPU time / GPU time shared): 1.44409x
Speedup (CPU time / GPU time shared): 408.616x

