# =============== 基本信息 ===============
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_op LANGUAGES CXX CUDA)

# =============== 语言标准 & PIC ===============
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置为debug编译模式
set(CMAKE_BUILD_TYPE Debug)

# # =============== GPU 架构 ===============
# # 本地 GPU + 常见兜底，开发/单机足够
# set(CMAKE_CUDA_ARCHITECTURES "native") #"native;75;86;89"

# # =============== Python & LibTorch ===============
# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# execute_process(
#   COMMAND "${Python3_EXECUTABLE}" -c
#           "import torch; print(torch.utils.cmake_prefix_path)"
#   OUTPUT_STRIP_TRAILING_WHITESPACE
#   OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
# )
# list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
# find_package(Torch REQUIRED)

# =============== pybind11 ===============
add_subdirectory(3rdparty/pybind11)

# =============== 自动收集源文件 ===============
file(GLOB_RECURSE SRC_FILES
     "${CMAKE_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_SOURCE_DIR}/src/*.cu")

message(SRC_FILES ${SRC_FILES})

# =============== 创建 Python 扩展 ===============
pybind11_add_module(cuda_op MODULE ${SRC_FILES})

# =============== 链接 & 头文件 ===============
# target_link_libraries(cuda_op PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(cuda_op PRIVATE "${CMAKE_SOURCE_DIR}/src")

# =============== 调试路径映射（可选） ===============
target_compile_options(cuda_op PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")