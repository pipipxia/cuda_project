#ifndef CUDA_MACRO_H
#define CUDA_MACRO_H

#include <iostream>
// cuda
#include <cuda_runtime.h>
#include "cuda_limit.h"

// 断言 blockDim 合法
#define CHECK_BLOCK_DIM()                                                   \
    do {                                                                    \
        int bx = blockDim.x;int by = blockDim.y;int bz = blockDim.z;        \
        if ((bx) <= 0 || (by) <= 0 || (bz) <= 0)                            \
            throw std::invalid_argument("blockDim contains 0");             \
        if ((bx) > 1024 || (by) > 1024 || (bz) > 64)                        \
            throw std::invalid_argument("blockDim exceeds hardware limit"); \
        if ((bx) * (by) * (bz) > 1024)                                      \
            throw std::invalid_argument("total threads > 1024");            \
    } while (0);

// 运行时查询设备实际上限
#define CHECK_BLOCK_DIM_RUNTIME(bx, by, bz, dev)                                         \
    do {                                                                                 \
        int bx = blockDim.x;int by = blockDim.y;int bz = blockDim.z;                     \
        int maxT = 0;                                                                    \
        cudaDeviceGetAttribute(&maxT, cudaDevAttrMaxThreadsPerBlock, dev);               \
        if ((bx) * (by) * (bz) > maxT)                                                   \
            throw std::invalid_argument("total threads exceeds device limit");           \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////
class CudaTimer
{
public:
    explicit CudaTimer(const char* name = "kernel") : m_name(name)
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void tic()  { cudaEventRecord(start_, 0); }
    void toc()  { cudaEventRecord(stop_, 0); }

    // 同步并打印时间
    void print()
    {
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        std::cout << "[" << m_name << "] " << ms << " ms\n";
    }

    // 仅返回毫秒，不打印
    float elapsed()
    {
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
    const char* m_name;
};

// 宏：一行调用即可测 kernel
#define CUDA_TIMER_SCOPE(name) CudaTimer _timer(name); _timer.tic();
#define CUDA_TIMER_TOC()       _timer.toc(); _timer.print();

////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
static int get_1d_idx()
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = z_idx * blockDim.y * blockDim.x + y_idx * blockDim.x + x_idx;
    return idx;
}



#endif