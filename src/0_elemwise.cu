#include "include.h"

namespace py = pybind11;

<<<<<<< HEAD
void elemwise_add(py::array_t<double> &output, py::array_t<double> input0, py::array_t<double> input1) {
    printf("this is cuda elemwise add function\n");
    output = input0 + input1;
}

PYBIND11_MODULE(PYTHON_MODULE_NAME, m) {
    m.doc() = "pybind11 python plugin";
    m.def("elemwise_add", &elemwise_add, "A function which adds two numbers");
}
=======
class ElemWise {
private:
// Python ndarray ──> PyArrayObject* ──> data buffer
// py::array_t<double> 值对象 py::array_t<double> 值对象值拷贝：包装内部会 Py_INCREF，引用计数+1 → 内存安全 值对象 = 浅拷贝（指针+引用计数）
// py::array_t<double>& 引用成员 引用成员：只保存裸指针 → 没有引用计数保护 → Python 随时可释放。
    py::array_t<double> in0_, in1_;
    py::array_t<double> out_;
public:
    ElemWise(py::array_t<double> out, py::array_t<double> in0, py::array_t<double> in1);

    // 业务函数：逐元素加
    void add();
};

// 构造函数：只做形状检查 + 保存视图, 构造函数初始化是 C++ 类里初始化 const 成员、引用成员的唯一方式。
ElemWise::ElemWise(py::array_t<double> out, py::array_t<double> in0, py::array_t<double> in1)
    : out_(std::move(out)), in0_(std::move(in0)), in1_(std::move(in1)) { 
    
    auto in0_info  = in0_.request();
    auto in1_info  = in1_.request();
    auto out_info = out_.request();
    if (in0_info.shape != out_info.shape || in0_info.shape != in1_info.shape)
        throw std::invalid_argument("Shape mismatch");
}

void ElemWise::add() {
    auto in0_info = in0_.request();
    auto in1_info = in1_.request();
    auto out_info = out_.request();

    const   double* in0_ptr = static_cast<double*>(in0_info.ptr);
    const   double* in1_ptr = static_cast<double*>(in1_info.ptr);
    double* out_ptr         = static_cast<double*>(out_info.ptr);

    size_t n = in0_info.size;
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = in0_ptr[i] + in1_ptr[i];
    }  
}

PYBIND11_MODULE(PYTHON_MODULE_NAME, m) {
    py::class_<ElemWise>(m, "ElemWise")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def("add", &ElemWise::add);
}
>>>>>>> 261b8c4 (cuda project init)
