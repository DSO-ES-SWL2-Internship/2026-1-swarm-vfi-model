# Hello world, to check the cross toolchain works before adding TFLite.
#
# Build for the iMX8 with the Yocto SDK (the environment script points CMake at the aarch64 compiler):
#   source ~/sdk/imx8mp/environment-setup-armv8a-poky-linux
#   cmake -B build && cmake --build build
#   file build/hello   -> should say ARM aarch64
# Without the SDK script, the same commands build an x86 version for the PC.

cmake_minimum_required(VERSION 3.16)
project(hello CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hello hello.cpp)
