# Developing on the board

There are three ways to put your own code on the i.MX8, from quickest to most permanent:

| Method | Best for | Cost of a change |
| --- | --- | --- |
| **Python script** | Prototyping. This is how the relay works | `scp` the file and run it |
| **C++ with the SDK** | Anything that must live in a C++ codebase | Cross-compile on your PC, `scp` the binary |
| **A Yocto recipe** | A final program that ships inside the image | Rebuild the image, reflash |

For day-to-day work, use Python or the SDK. The recipe route is for release builds. It is covered in
[The Yocto image](yocto.md).

## Running a model on the NPU

Both languages do the same four things:

| Step | What happens |
| --- | --- |
| **Set up** | Load the model, and attach NXP's VX delegate (`/usr/lib/libvx_delegate.so`), which sends the work to the NPU |
| **Warm up** | Run one inference. The NPU compiles the model's graph on the first run, which is slow. It is cached afterwards |
| **Run** | Copy the input in, invoke, read the output |
| **Check** | Compare the time against your budget |

The VX delegate is not linked into your program. It is loaded **by path at run time**.

=== "Python"

    ```python
    import numpy as np
    from tflite_runtime.interpreter import Interpreter, load_delegate

    vx = load_delegate("/usr/lib/libvx_delegate.so")
    interp = Interpreter(model_path="vfi_unet.tflite", experimental_delegates=[vx])
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # Warm up: the first run compiles the graph for the NPU
    interp.set_tensor(inp["index"], np.zeros(inp["shape"], dtype=inp["dtype"]))
    interp.invoke()

    # Run (frames: int8, shape (1, 256, 448, 2))
    interp.set_tensor(inp["index"], frames)
    interp.invoke()
    result = interp.get_tensor(out["index"])
    ```

=== "C++"

    ```cpp
    #include "tensorflow/lite/model.h"
    #include "tensorflow/lite/interpreter.h"
    #include "tensorflow/lite/kernels/register.h"
    #include "tensorflow/lite/delegates/external/external_delegate.h"

    auto model = tflite::FlatBufferModel::BuildFromFile("vfi_unet.tflite");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp;
    tflite::InterpreterBuilder(*model, resolver)(&interp);

    auto opts = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
    interp->ModifyGraphWithDelegate(TfLiteExternalDelegateCreate(&opts));
    interp->AllocateTensors();

    // Warm up
    TfLiteTensor* t = interp->input_tensor(0);
    std::memset(t->data.raw, 0, t->bytes);
    interp->Invoke();

    // Run
    std::memcpy(interp->typed_input_tensor<int8_t>(0), frames, n_bytes);
    interp->Invoke();
    int8_t* result = interp->typed_output_tensor<int8_t>(0);
    ```

!!! warning "If the NPU is not used, nothing tells you"
    A float operation, or a layer the NPU cannot compile, makes the delegate quietly fall back to the CPU. Your program
    still works, just far slower. Always check the delegate's log line, or time the inference. See the
    [rules for the NPU](ml-model.md#export-it-for-the-npu).

## Measuring a model: benchmark_model

TensorFlow Lite ships a benchmark tool on the board. It is the quickest way to time a model, with or without the NPU:

```bash
B=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model

$B --graph=vfi_unet.tflite --num_runs=100                                        # CPU, one thread
$B --graph=vfi_unet.tflite --num_runs=100 --num_threads=4                        # CPU, four threads
$B --graph=vfi_unet.tflite --num_runs=100 --warmup_runs=5 \
   --external_delegate_path=/usr/lib/libvx_delegate.so                          # NPU
USE_GPU_INFERENCE=1 $B --graph=vfi_unet.tflite --num_runs=3 --warmup_runs=1 \
   --external_delegate_path=/usr/lib/libvx_delegate.so                          # GPU (slow)
```

Read the `Inference (avg)` figure, in microseconds. The tool feeds random input, which is fine for timing because a
convolutional network does the same work whatever the pixel values. `model/experiments/benchmarks/benchmark_backends.sh`
runs the three backends over several models and writes a table.

## C++ with the SDK

You compile on your PC with a cross-compiler, then copy the binary to the board. The **SDK** packages the
cross-compiler together with a copy of the board's libraries and headers (TFLite 2.19, the VX delegate), so what you
build links against exactly what is on the board.

### 1. Build the SDK (once)

Inside the Yocto build environment (see [The Yocto image](yocto.md#build-the-image)):

```bash
bitbake -k core-image-base -c populate_sdk
```

This produces an installer of about 2 GB in `tmp/deploy/sdk/`, named like
`fsl-imx-xwayland-glibc-x86_64-core-image-base-armv8a-imx8mpevk-toolchain-6.18-wrynose.sh`. It is built from the same
image recipe, so its libraries match the board's.

### 2. Install it (once)

```bash
./fsl-imx-xwayland-glibc-x86_64-core-image-base-armv8a-imx8mpevk-toolchain-6.18-wrynose.sh -d ~/sdk/imx8mp
```

Then, **in every new terminal**:

```bash
source ~/sdk/imx8mp/environment-setup-armv8a-poky-linux
echo $CXX      # aarch64-poky-linux-g++ ... --sysroot=...
which cmake    # a path under ~/sdk/imx8mp
```

### 3. Build

The example in `model/cpp/npu_infer_test` finds TFLite through its pkg-config file:

```cmake
cmake_minimum_required(VERSION 3.16)
project(npu_infer_test CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TFLITE REQUIRED IMPORTED_TARGET tensorflow2-lite)
add_executable(npu_infer_test npu_infer_test.cpp)
target_link_libraries(npu_infer_test PRIVATE PkgConfig::TFLITE)
```

```bash
cmake -B build && cmake --build build
file build/npu_infer_test     # ELF 64-bit LSB pie executable, ARM aarch64
```

The `file` check is the quickest way to confirm you built for the board and not for your PC.

### 4. Copy and run

```bash
scp build/npu_infer_test root@<board ip>:~/
ssh root@<board ip> ./npu_infer_test vfi_unet.tflite
```

The binary is dynamically linked and small (about 215 KB). It uses the board's `libtensorflow-lite.so`, which the image
already has. The model is a separate file, read at run time.

## Watching the hardware

`hw_monitor.py` shows live load next to any pipeline. Run it in a second window:

```bash
mount -t debugfs none /sys/kernel/debug      # once per boot
python3 hw_monitor.py
```

For a quick look at the four cores, `htop` is in the image too. The GPU and NPU rows in `hw_monitor.py` count idle
cycles, so use them to see *whether* a unit is working, not as a percentage.

## Moving files

| What | How |
| --- | --- |
| A file to the board | `scp file root@<board ip>:~/` |
| A folder, only the changes | `rsync -av folder/ root@<board ip>:~/folder/` (`rsync` is in the image) |
| A look at the board's screen | Serial console: `/dev/ttyUSB2`, 115200 baud, user `root`, no password |

!!! tip "Close PuTTY before running a script that uses the serial port"
    Two programs reading the same port garble each other's output.
