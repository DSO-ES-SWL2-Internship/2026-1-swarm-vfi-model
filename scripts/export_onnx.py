# Exports the trained toy U-Net to a fully INT8-quantized ONNX model, as an
# alternative to export_tflite.py's TFLite export -- lets the model be served
# by ONNX Runtime (targeting GPU/NPU via onnxruntime-openvino on the Windows
# host) instead of only tf.lite.Interpreter's CPU-only path there.
#
# Converts straight from the original float32 Keras model, not from the
# already-quantized .tflite file -- avoids porting TFLite's INT8 quantization
# representation across formats (which has known op-mapping gaps), and lets
# ONNX Runtime's own quantizer calibrate independently. The resulting INT8
# ONNX model is NOT expected to be numerically identical to toy_unet_int8.tflite
# -- it's a fresh quantization via a different toolchain, which is fine since
# the toy U-Net is a placeholder model, not something needing cross-format
# bit-exact reproduction.

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

import interpolate
import training
import vimeo_dataset

REP_DATASET_SIZE = 100  # same size as export_tflite.py's calibration set, for consistency

FP32_ONNX_PATH = training.ARTIFACTS_DIR / "toy_unet_fp32.onnx"


def main():
    convert_to_onnx()
    quantize()

    fp32_size = FP32_ONNX_PATH.stat().st_size
    int8_size = training.ONNX_MODEL_PATH.stat().st_size
    print(f"saved {training.ONNX_MODEL_PATH} ({int8_size / 1024:.0f} KB, "
          f"vs {fp32_size / 1024:.0f} KB fp32 .onnx -- {fp32_size / int8_size:.1f}x smaller)\n")
    sanity_check()


# Step 1: Keras -> ONNX (float32), straight from the original model, no TFLite involved
def convert_to_onnx():
    model = interpolate.load_model()
    input_signature = [tf.TensorSpec((1, training.IMG_HEIGHT, training.IMG_WIDTH, 2), tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

    training.ARTIFACTS_DIR.mkdir(exist_ok=True)
    onnx.save(onnx_model, FP32_ONNX_PATH)


# Calibration data reader, mirroring export_tflite.py's representative_dataset_gen
# -- same Vimeo train split, same sample count, so the two quantized models are
# calibrated on comparable (though not identically toolchain-processed) data.
class VimeoCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name):
        ds = vimeo_dataset.make_dataset("train.txt", batch_size=1, shuffle=True)
        self.input_name = input_name
        self.iterator = iter(ds.take(REP_DATASET_SIZE))

    def get_next(self):
        try:
            x, _ = next(self.iterator)
            return {self.input_name: x.numpy()}
        except StopIteration:
            return None


# Step 2: ONNX float32 -> ONNX INT8, using ONNX Runtime's own static quantizer
def quantize():
    session = ort.InferenceSession(str(FP32_ONNX_PATH))
    input_name = session.get_inputs()[0].name

    quantize_static(
        model_input=str(FP32_ONNX_PATH),
        model_output=str(training.ONNX_MODEL_PATH),
        calibration_data_reader=VimeoCalibrationReader(input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )


# Runs one real example to sanity check for proper inference -- mirrors
# export_tflite.py's sanity_check(), same eval sample, for a direct comparison
def sanity_check():
    session = ort.InferenceSession(str(training.ONNX_MODEL_PATH))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    ds = vimeo_dataset.make_dataset("eval.txt", batch_size=1, shuffle=False)
    x, y = next(iter(ds.take(1)))

    out = session.run([output_name], {input_name: x.numpy()})[0]

    print(f"input:  shape={x.shape} dtype={x.numpy().dtype}")
    print(f"output: shape={out.shape} dtype={out.dtype}")
    print(f"sample output range: [{out.min():.4f}, {out.max():.4f}] "
          f"(ground truth range: [{y.numpy().min():.4f}, {y.numpy().max():.4f}])")


if __name__ == "__main__":
    main()
