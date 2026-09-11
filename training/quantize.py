"""
Quantization and TFLite conversion pipeline for Swara (Production-Ready Infrastructure).

Target: FULL INTEGER INT8 TFLite for TensorFlow Lite Micro.
Requirements:
- input tensor: INT8 (shape [1, 49, 10, 1])
- output tensor: INT8 (shape [1, 3])
- weights & activations: INT8
- representative dataset MUST come from real training features (never all-zero dummy data)
- strict failure if real training dataset is absent
"""

import os
import sys
import argparse
from typing import Generator, List
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from dataset import SwaraDataset


def create_representative_dataset_generator(
    data_dir: str,
    max_samples: int = 150,
) -> Generator[List[np.ndarray], None, None]:
    """
    Builds a calibration generator yielding real MFCC features from the training dataset split.
    Fails explicitly if real audio data is absent to avoid invalid zero/dummy calibration.
    """
    dataset_loader = SwaraDataset(data_dir=data_dir)
    splits = dataset_loader.scan_dataset()
    train_files = splits.get("train", [])

    if len(train_files) == 0:
        raise FileNotFoundError(
            f"\n[ERROR] Cannot calibrate INT8 quantization: no real training data found in '{os.path.abspath(data_dir)}'.\n"
            "Full INT8 quantization requires REAL representative audio features to determine\n"
            "proper dynamic activation ranges, scale factors, and zero points.\n"
            "Using fake, random, or zero arrays is strictly prohibited."
        )

    # Use up to max_samples for representative calibration
    selected_files = train_files[:max_samples]
    print(f"Calibrating INT8 quantization with {len(selected_files)} real training samples...")

    X, _ = dataset_loader.load_tensors_from_file_list(selected_files)

    for i in range(len(X)):
        # Model input shape is (1, 49, 10, 1) float32
        sample = np.expand_dims(X[i], axis=0).astype(np.float32)
        yield [sample]


def convert_to_float32_tflite(model_path: str, output_path: str):
    """Convert Keras model / SavedModel to baseline Float32 TFLite."""
    if tf is None:
        raise RuntimeError("TensorFlow is required for TFLite conversion.")

    print(f"Converting {model_path} to Float32 TFLite: {output_path}...")
    if os.path.isdir(model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Float32 model saved to {output_path} ({len(tflite_model)} bytes)")
    return tflite_model


def convert_to_int8_tflite(
    model_path: str,
    data_dir: str,
    output_path: str,
    max_calibration_samples: int = 150,
):
    """
    Convert Keras model to FULL INTEGER INT8 TFLite using real dataset calibration.
    Enforces INT8 input and INT8 output for direct MCU memory mapping.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for TFLite quantization.")

    print(f"Converting {model_path} to FULL INT8 TFLite: {output_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Enforce representative dataset from real data
    representative_gen = lambda: create_representative_dataset_generator(
        data_dir=data_dir,
        max_samples=max_calibration_samples,
    )

    if os.path.isdir(model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Full integer INT8 model saved to {output_path} ({len(tflite_model)} bytes)")
    return tflite_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Swara Model to Full Integer INT8")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained Keras model or SavedModel")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to real dataset for INT8 calibration")
    parser.add_argument("--output_int8", type=str, default="models/swara_int8.tflite", help="Destination path for INT8 TFLite")
    parser.add_argument("--output_float32", type=str, default="models/swara_float32.tflite", help="Optional Float32 TFLite output")
    args = parser.parse_args()

    if args.output_float32:
        convert_to_float32_tflite(args.model_path, args.output_float32)

    convert_to_int8_tflite(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_path=args.output_int8,
    )
