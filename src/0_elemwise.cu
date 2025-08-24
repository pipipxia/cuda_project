#include "include.h"

namespace py = pybind11;

void elemwise_add(py::array_t<double> &output, py::array_t<double> input0, py::array_t<double> input1) {
    printf("this is cuda elemwise add function\n");
    output = input0 + input1;
}

PYBIND11_MODULE(PYTHON_MODULE_NAME, m) {
    m.doc() = "pybind11 python plugin";
    m.def("elemwise_add", &elemwise_add, "A function which adds two numbers");
}
