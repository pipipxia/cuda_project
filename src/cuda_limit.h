#ifndef CUDA_LIMITS_H
#define CUDA_LIMITS_H

#include <cuda_runtime.h>

/* -------------------- 线程块维度 -------------------- */
#define CUDA_MAX_BLOCK_DIM_X   1024
#define CUDA_MAX_BLOCK_DIM_Y   1024
#define CUDA_MAX_BLOCK_DIM_Z   64
#define CUDA_MAX_THREADS_PER_BLOCK 1024            // 乘积上限

/* -------------------- 网格维度 -------------------- */
#define CUDA_MAX_GRID_DIM_X    2147483647          // 2^31-1
#define CUDA_MAX_GRID_DIM_Y    65535
#define CUDA_MAX_GRID_DIM_Z    65535

/* -------------------- 共享内存 -------------------- */
#define CUDA_MAX_SHARED_PER_BLOCK 49152            // 48 KB (Volta+ 可配 96 KB)

/* -------------------- 纹理/表面内存 -------------------- */
#define CUDA_MAX_TEXTURE_1D  65536                 // 线性 1D 纹理
#define CUDA_MAX_TEXTURE_2D  65536                 // 2D 纹理单边最大
#define CUDA_MAX_TEXTURE_3D  2048                  // 3D 纹理单边最大

/* -------------------- 常量内存 -------------------- */
#define CUDA_MAX_CONSTANT_BYTES 65536              // 64 KB

/* -------------------- 运行时查询宏 -------------------- */
#define CUDA_GET_ATTR(attr, dev) [&]{ int v; cudaDeviceGetAttribute(&v, attr, dev); return v; }()

#endif // CUDA_LIMITS_H