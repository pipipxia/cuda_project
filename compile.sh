#!/bin/bash
# BUILD_DIR="build"
# if [[ -d "$BUILD_DIR" ]]; then
#     echo "检测到 build 目录，正在删除..."
#     rm -rf "$BUILD_DIR"
# fi
# echo "重新创建 build 目录..."
# mkdir -p "$BUILD_DIR"

BUILD_SO_NAME="cuda_op.cpython-312-x86_64-linux-gnu.so"
BUILD_DIR="/home/lihuixiang/cuda/cuda_project/build"

[ -f $BUILD_SO_NAME ] && rm -f $BUILD_SO_NAME
[ -d $BUILD_DIR ] || mkdir -p $BUILD_DIR
cd $BUILD_DIR
echo "构建项目"
cmake .. || exit 1
echo "编译cuda project....."
make -j12 || exit 1
cp ${BUILD_DIR}/${BUILD_SO_NAME} ${BUILD_DIR}/../
# cd ../
# echo "允许非root用户附加进程....."
# sudo tee /proc/sys/kernel/yama/ptrace_scope <<< "0"
