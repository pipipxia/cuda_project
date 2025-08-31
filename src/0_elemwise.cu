#include "include.h"
#include "cuda_kernel.h"

namespace py = pybind11;

template <typename T>
class ElemWise {
private:
// Python ndarray ──> PyArrayObject* ──> data buffer
// py::array_t<T> 值对象 py::array_t<double> 值对象值拷贝：包装内部会 Py_INCREF，引用计数+1 → 内存安全 值对象 = 浅拷贝（指针+引用计数）
// py::array_t<T>& 引用成员 引用成员：只保存裸指针 → 没有引用计数保护 → Python 随时可释放。
    py::array_t<T> in0_, in1_;
    py::array_t<T> out_;
public:
    ElemWise(py::array_t<T> out, py::array_t<T> in0, py::array_t<T> in1);

    // 业务函数：逐元素加
    void add();
};

// 构造函数：只做形状检查 + 保存视图, 构造函数初始化是 C++ 类里初始化 const 成员、引用成员的唯一方式。
template <typename T>
ElemWise<T>::ElemWise(py::array_t<T> out, py::array_t<T> in0, py::array_t<T> in1)
    : out_(std::move(out)), in0_(std::move(in0)), in1_(std::move(in1)) { 
    
    auto in0_info  = in0_.request();
    auto in1_info  = in1_.request();
    auto out_info = out_.request();
    if (in0_info.shape != out_info.shape || in0_info.shape != in1_info.shape){
        throw std::invalid_argument("Shape mismatch");
        printf("bbbb");}
}

template <typename T>
void ElemWise<T>::add() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto in0_info = in0_.request();
    auto in1_info = in1_.request();
    auto out_info = out_.request();
    size_t num = in0_info.size;
    size_t bytes = num * sizeof(T);

    // CPU内存
    T* h_in0 = static_cast<T*>(in0_info.ptr);
    T* h_in1 = static_cast<T*>(in1_info.ptr);
    T* h_out = static_cast<T*>(out_info.ptr);

    // GPU内存
    T* d_in0, *d_in1, *d_out;
    cudaMalloc((T **)&d_in0, bytes);
    cudaMalloc((T **)&d_in1, bytes);
    cudaMalloc((T **)&d_out, bytes);

    // 数据copy to gpu内存
    cudaMemcpy(d_in0, h_in0, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in1, h_in1, bytes, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);

    // 初始化grid block size
    dim3 block_dim(1, 1, num);// x, y, z, 一个线程块最大值是1024，超出计算结果就是0
    dim3 grid_dim(1, 1, 1);
    add_kernel<T><<<grid_dim, block_dim>>>(d_out, d_in0, d_in1, num);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    // 同步，等待kernel执行完毕
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);

    // 同步版的memcpy，内部会自动等待device执行完毕
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in0);
    cudaFree(d_in1);
    cudaFree(d_out);
    // // cpu计算
    // for (size_t i = 0; i < num; ++i) {
    //     h_out[i] = h_in0[i] + h_in1[i];
    // }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 实例化类模板，如果自己没写，编译器会补上
template class ElemWise<double>;
template class ElemWise<float>;

// 类模板注册到python中
PYBIND11_MODULE(PYTHON_MODULE_NAME, m)
{
    py::class_<ElemWise<double>>(m, "ElemWiseFloat64")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def("add", &ElemWise<double>::add);

    py::class_<ElemWise<float>>(m, "ElemWiseFloat32")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>>())
        .def("add", &ElemWise<float>::add);
}


