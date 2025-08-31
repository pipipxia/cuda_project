#include "cuda_kernel.h"
#include "cuda_macro.h"

template <typename T>
__global__ void add_kernel(T* out, const T* in0, const T* in1, int num) {
    CHECK_BLOCK_DIM();
    int idx = get_1d_idx();
    if (idx < num) {
        out[idx] = in0[idx] + in1[idx];
    }
}
// 必须显示实例化，不然运行会报错找不到add_kernel符号
template __global__ void add_kernel<float>(float*, const float*, const float*, int);
template __global__ void add_kernel<double>(double*, const double*, const double*, int);