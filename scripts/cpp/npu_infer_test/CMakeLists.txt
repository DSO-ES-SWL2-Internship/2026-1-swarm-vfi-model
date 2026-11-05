# npu_infer_test: TFLite inference on the NPU through the VX delegate (see npu_infer_test.cpp).
#
# Build for the iMX8 with the Yocto SDK (the environment script points CMake at the aarch64 compiler
# and the board's sysroot, which has the TFLite headers and library):
#   source ~/sdk/imx8mp/environment-setup-armv8a-poky-linux
#   cmake -B build && cmake --build build
#   file build/npu_infer_test   -> should say ARM aarch64
# The PC has no TFLite, so this one only builds with the SDK.

cmake_minimum_required(VERSION 3.16)
project(npu_infer_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TFLite installs a pkg-config file (tensorflow2-lite.pc) with its include and link flags.
# The VX delegate isn't linked: the program loads /usr/lib/libvx_delegate.so at runtime.
find_package(PkgConfig REQUIRED)
pkg_check_modules(TFLITE REQUIRED IMPORTED_TARGET tensorflow2-lite)

add_executable(npu_infer_test npu_infer_test.cpp)
target_link_libraries(npu_infer_test PRIVATE PkgConfig::TFLITE)
