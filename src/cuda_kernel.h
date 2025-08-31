#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include <iostream>
// cuda
#include <cuda_runtime.h>

template <typename T>
__global__ void add_kernel(T* out, const T* in0, const T* in1, int num);



#endif