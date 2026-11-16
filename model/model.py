"""The VFI U-Net: architecture, loss and the paths of the trained model.

The network takes the frames before and after (2 channels, 256x448, grey, video range) and predicts the frame between
them. It is a small U-Net (two poolings, 16/32/64 channels) whose output is the AVERAGE of the two input frames plus a
learned correction: sharp where nothing moves, and the network only has to describe the difference. The average is a
fixed 1x1 convolution (no channel slicing), so the whole graph stays on the iMX8's NPU.

    input (2) -> encoder -> bottleneck -> decoder (with skip connections) -> delta (1)
    output = clip( (A + B) / 2 + delta, 0, 1 )

The correction layer starts at zero, so training starts from a plain blend. The trained model is artifacts/vfi_unet.keras
(export_tflite.py makes the INT8 file artifacts/vfi_unet.tflite that the relay loads).
"""
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_HEIGHT, IMG_WIDTH = 256, 448  # the size the model works at (the relay scales every frame to it)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"  # models of the ablation runs (train.py --name)
MODEL_PATH = ARTIFACTS_DIR / "vfi_unet.keras"
TFLITE_PATH = ARTIFACTS_DIR / "vfi_unet.tflite"


def build_model(width=1.0, residual=True, h=IMG_HEIGHT, w=IMG_WIDTH):
	"""The U-Net. width scales the channels (16/32/64 at 1.0); residual=False gives a plain sigmoid output instead of
	the blend plus correction (kept only for the ablation runs). Fully convolutional: h and w can be any multiple of 4."""
	c1, c2, c3 = (round(c * width) for c in (16, 32, 64))
	inputs = keras.Input(shape=(h, w, 2))
	e1 = layers.Conv2D(c1, 3, padding="same", activation="relu")(inputs)
	e2 = layers.Conv2D(c2, 3, padding="same", activation="relu")(layers.MaxPooling2D(2)(e1))
	b = layers.Conv2D(c3, 3, padding="same", activation="relu")(layers.MaxPooling2D(2)(e2))
	u2 = layers.Conv2DTranspose(c2, 3, strides=2, padding="same", activation="relu")(b)
	u2 = layers.Conv2D(c2, 3, padding="same", activation="relu")(layers.Concatenate()([u2, e2]))
	u1 = layers.Conv2DTranspose(c1, 3, strides=2, padding="same", activation="relu")(u2)
	u1 = layers.Conv2D(c1, 3, padding="same", activation="relu")(layers.Concatenate()([u1, e1]))
	if not residual:
		return keras.Model(inputs, layers.Conv2D(1, 3, padding="same", activation="sigmoid")(u1), name="unet")
	average = layers.Conv2D(1, 1, use_bias=False, trainable=False, kernel_initializer=keras.initializers.Constant(0.5),
	                        name="average")(inputs)
	delta = layers.Conv2D(1, 3, padding="same", kernel_initializer="zeros", name="delta")(u1)
	out = layers.ReLU(max_value=1.0, name="clip")(layers.Add(name="sum")([average, delta]))
	return keras.Model(inputs, out, name="unet_residual")


def make_loss(grad_weight=0.0):
	"""Charbonnier loss (a smooth L1 on the pixels), plus grad_weight x an L1 loss on the image gradients, which
	punishes blur directly (an ablation: it did not help LPIPS)."""
	def loss(y, p):
		l = tf.reduce_mean(tf.sqrt(tf.square(y - p) + 1e-6))
		if grad_weight:
			gy, gp = tf.image.image_gradients(y), tf.image.image_gradients(p)
			l += grad_weight * sum(tf.reduce_mean(tf.abs(a - b)) for a, b in zip(gy, gp))
		return l
	return loss


def load_model(path=MODEL_PATH):
	return keras.models.load_model(path, compile=False)
