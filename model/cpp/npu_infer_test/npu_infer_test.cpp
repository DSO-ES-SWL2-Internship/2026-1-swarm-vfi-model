// Checks the C++ code on the "NPU Usage for ML Model" slide on the iMX8: sets up the interpreter with the VX
// (NPU) delegate, warms up, then times inference on a fixed test input. The input and the output are saved
// so that compare_with_python.py can run the Python version of the slide on the same input and compare.
//
// Build: cross-compiled on the PC with the Yocto SDK and CMake (see CMakeLists.txt); the board has no compiler.
// Run on the iMX8: npu_infer_test MODEL.tflite [--cpu] [--runs N] [--out DIR]
//   --cpu   no delegate, 4 CPU threads (to compare the NPU against)
//   --runs  timed inferences after the warm-up (default 50)
//   --out   where input.bin and output_cpp_<npu|cpu>.bin go (default /tmp/npu_infer_test)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

static const char* VX_DELEGATE = "/usr/lib/libvx_delegate.so";
static const int CPU_THREADS = 4;

static double ms_since(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static void fail(const std::string& what) {
	std::fprintf(stderr, "npu_infer_test: %s\n", what.c_str());
	std::exit(1);
}

static void save(const std::string& path, const void* data, size_t bytes) {
	std::ofstream f(path, std::ios::binary);
	f.write(static_cast<const char*>(data), bytes);
	if (!f) fail("could not write " + path);
}

static std::string describe(const TfLiteTensor* t) {
	// e.g. "int8 [1, 256, 448, 2], scale 0.00392, zero point -128"
	std::string s = std::string(TfLiteTypeGetName(t->type)) + " [";
	for (int i = 0; i < t->dims->size; i++) s += (i ? ", " : "") + std::to_string(t->dims->data[i]);
	char q[80];
	std::snprintf(q, sizeof q, "], scale %.6g, zero point %d", t->params.scale, t->params.zero_point);
	return s + q;
}

int main(int argc, char** argv) {
	if (argc < 2) fail("usage: npu_infer_test MODEL.tflite [--cpu] [--runs N] [--out DIR]");
	std::string model_path = argv[1], out_dir = "/tmp/npu_infer_test";
	bool use_cpu = false;
	int runs = 50;
	for (int i = 2; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--cpu") use_cpu = true;
		else if (arg == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
		else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
		else fail("unknown argument " + arg);
	}
	mkdir(out_dir.c_str(), 0755);
	const char* target = use_cpu ? "cpu" : "npu";

	// Slide row 1 -- set up: the interpreter with the model and the NPU delegate
	auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
	if (!model) fail("could not load " + model_path);
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interp;
	if (tflite::InterpreterBuilder(*model, resolver)(&interp) != kTfLiteOk || !interp) fail("InterpreterBuilder failed");

	TfLiteDelegate* npu = nullptr;
	if (use_cpu) {
		interp->SetNumThreads(CPU_THREADS);
	} else {
		auto o = TfLiteExternalDelegateOptionsDefault(VX_DELEGATE);
		npu = TfLiteExternalDelegateCreate(&o);
		if (!npu) fail(std::string("could not load ") + VX_DELEGATE);
		if (interp->ModifyGraphWithDelegate(npu) != kTfLiteOk) fail("ModifyGraphWithDelegate failed");
	}
	if (interp->AllocateTensors() != kTfLiteOk) fail("AllocateTensors failed");

	// How much of the model the delegate took: it replaces the ops it runs with one node per part it takes,
	// so a fully delegated model has an execution plan of 1 node, run by the delegate.
	int delegated = 0;
	for (int node : interp->execution_plan())
		if (interp->node_and_registration(node)->first.delegate) delegated++;
	std::printf("model %s on the %s\n", model_path.c_str(), target);
	std::printf("execution plan: %zu nodes, %d run by the delegate\n", interp->execution_plan().size(), delegated);

	TfLiteTensor* in = interp->input_tensor(0);
	const TfLiteTensor* out = interp->output_tensor(0);
	std::printf("input  %s\noutput %s\n", describe(in).c_str(), describe(out).c_str());
	if (in->type != kTfLiteInt8 || out->type != kTfLiteInt8) fail("expected an INT8 model (int8 input and output)");

	// Slide row 2 -- warm up: one inference so the NPU graph is compiled and cached
	auto start = std::chrono::steady_clock::now();
	std::memset(in->data.raw, 0, in->bytes);
	if (interp->Invoke() != kTfLiteOk) fail("warm-up Invoke failed");
	std::printf("warm-up: %.1f ms\n", ms_since(start));

	// A fixed test input: pseudo-random bytes, the same on every run. The values don't need to look like
	// frames; the point is that the C++ and Python versions get the same input and should give the same output.
	std::vector<int8_t> frames(in->bytes);
	for (size_t i = 0; i < frames.size(); i++) frames[i] = static_cast<int8_t>((i * 2654435761u) >> 24);
	size_t n_bytes = frames.size();

	// Slide row 3 -- run: load the inputs -> run inference -> read the outputs
	double total = 0;
	for (int r = 0; r < runs; r++) {
		start = std::chrono::steady_clock::now();
		std::memcpy(interp->typed_input_tensor<int8_t>(0), frames.data(), n_bytes);
		if (interp->Invoke() != kTfLiteOk) fail("Invoke failed");
		total += ms_since(start);
	}
	int8_t* result = interp->typed_output_tensor<int8_t>(0);

	long sum = 0;
	int lo = 127, hi = -128;
	for (size_t i = 0; i < out->bytes; i++) {
		sum += result[i];
		lo = std::min<int>(lo, result[i]);
		hi = std::max<int>(hi, result[i]);
	}
	std::printf("inference: %.2f ms mean over %d runs\n", total / runs, runs);
	std::printf("output values: min %d, max %d, mean %.2f\n", lo, hi, double(sum) / out->bytes);

	save(out_dir + "/input.bin", frames.data(), n_bytes);
	save(out_dir + "/output_cpp_" + target + ".bin", result, out->bytes);
	std::printf("saved %s/input.bin and output_cpp_%s.bin\n", out_dir.c_str(), target);

	interp.reset();  // the interpreter must go before the delegate it uses
	if (npu) TfLiteExternalDelegateDelete(npu);
	return 0;
}
