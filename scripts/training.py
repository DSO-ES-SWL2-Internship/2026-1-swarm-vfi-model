"""
Trains a U-Net VFI Model on Vimeo-90K dataset
Exports as INT8 TFLite file for deployment on iMX8M-Plus
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

IMG_HEIGHT = 256 # Vimeo-90K
IMG_WIDTH = 448 # Vimeo-90K
RADIUS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "toy_unet.keras"
SAMPLE_PREDICTION_PATH = ARTIFACTS_DIR / "sample_prediction.png"
TFLITE_MODEL_PATH = ARTIFACTS_DIR / "toy_unet_int8.tflite"
ONNX_MODEL_PATH = ARTIFACTS_DIR / "toy_unet_int8.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the toy VFI U-Net")
    parser.add_argument(
        "--data", choices=["synthetic", "vimeo"], default="vimeo",
        help="synthetic: moving-circle generator. vimeo: real Vimeo-90K subset (see vimeo_dataset.py)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Dataset
    if args.data == "vimeo":
        import vimeo_dataset
        train_ds = vimeo_dataset.make_dataset("train.txt", batch_size=8)
        val_ds = vimeo_dataset.make_dataset("val.txt", batch_size=8, shuffle=False)
        loss_fn = charbonnier_loss  # weighted_mae is tuned for the synthetic bright-circle data only
        steps_per_epoch = max(1, len(vimeo_dataset.load_manifest("train.txt")) // 8)
        validation_steps = max(1, len(vimeo_dataset.load_manifest("val.txt")) // 8)
    else:
        train_ds = make_dataset(batch_size=8)
        val_ds = make_dataset(batch_size=8)
        loss_fn = weighted_mae
        steps_per_epoch = 300
        validation_steps = 30

    # Model architecture definition and compilation
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn)
    model.summary() # Prints out model architecture

    # Train
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=val_ds,
        validation_steps=validation_steps,
    )

    # Save and export model, and sample image
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    for x_batch, _ in val_ds.take(1):
        pred = model.predict(x_batch[:1], verbose=0)
        pred_img = (pred[0, ..., 0] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_img, mode="L").save(SAMPLE_PREDICTION_PATH)

# Draws a circle on a 2D array representing an image
def render_circle(cx, cy, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, radius=RADIUS):
    yy, xx = np.mgrid[0:img_h, 0:img_w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(radius + 1 - dist, 0, 1).astype(np.float32)

# Creates a triplet of images, each with a circle at a different position
def make_triplet(img_h=IMG_HEIGHT, img_w=IMG_WIDTH, margin=40, displacement_range=(10, 150)):
    displacement = np.random.uniform(*displacement_range)
    angle = np.random.uniform(0, 2 * np.pi)
    cy0 = img_h / 2 + np.random.uniform(-40, 40)
    cx0 = margin
    cx2 = np.clip(cx0 + displacement * np.cos(angle), margin, img_w - margin)
    cy2 = np.clip(cy0 + displacement * np.sin(angle), margin, img_h - margin)
    cx1, cy1 = (cx0 + cx2) / 2, (cy0 + cy2) / 2
 
    f0 = render_circle(cx0, cy0)
    f1 = render_circle(cx1, cy1)
    f2 = render_circle(cx2, cy2)
    return f0[..., None], f1[..., None], f2[..., None]  # add channel dim

# Generator function for building dataset
def data_generator():
    while True:
        f0, f1, f2 = make_triplet()
        x = np.concatenate([f0, f2], axis=-1)  # (H, W, 2)
        yield x, f1

# Creates synthethic dataset, consisting of triplets of images of circles in different locations along the image
def make_dataset(batch_size=8):
    ds = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Defines U-Net model architecture (using functions API)
def build_model():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 2))

    # Model layers
    encode1 = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    downsample1 = layers.MaxPooling2D(pool_size=2)(encode1)

    encode2 = layers.Conv2D(32, 3, padding="same", activation="relu")(downsample1)
    downsample2 = layers.MaxPooling2D(pool_size=2)(encode2)
    
    bottleneck = layers.Conv2D(64, 3, padding="same", activation="relu")(downsample2)

    upsample2 = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(bottleneck)
    upsample2 = layers.Concatenate()([upsample2, encode2])
    upsample2 = layers.Conv2D(32, 3, padding="same", activation="relu")(upsample2)

    upsample1 = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(upsample2)
    upsample1 = layers.Concatenate()([upsample1, encode1])
    upsample1 = layers.Conv2D(16, 3, padding="same", activation="relu")(upsample1)

    outputs = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(upsample1)

    return keras.Model(inputs, outputs, name="toy_vfi_unet")

# Loss function for synthetic test
# Weight mean absolute error
# Upweighted for the white circles which cover a small area only to counteract imbalanced dataset
# Reminder to change to charbonnier_loss for real video data (e.g. Vimeo-90K)
def weighted_mae(y_true, y_pred):
    weight = 1.0 + 15.0 * y_true
    return tf.reduce_mean(weight * tf.abs(y_true - y_pred))

# Loss function for real video data
def charbonnier_loss(y_true, y_pred, epsilon=1e-3):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + epsilon**2)) 

if __name__ == '__main__':
    main()
