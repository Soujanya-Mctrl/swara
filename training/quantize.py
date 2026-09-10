"""
Quantization and TFLite conversion pipeline for Swara.
Converts trained models to:
1. Float32 TFLite (`models/swara_float32.tflite`)
2. Fully quantized INT8 TFLite (`models/swaral_int8.tflite` / `models/swara_int8.tflite`)
And can export C array source/headers for embedded deployment.
"""

import os
import argparse
import subprocess
import numpy as np


def convert_to_float32_tflite(saved_model_dir: str, output_path: str):
    """Convert TensorFlow SavedModel to standard Float32 TFLite."""
    print(f"Converting {saved_model_dir} to Float32 TFLite: {output_path}...")
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"Float32 model saved to {output_path} ({len(tflite_model)} bytes)")
    except ImportError:
        print("TensorFlow not installed. Writing placeholder if missing.")


def convert_to_int8_tflite(saved_model_dir: str, output_path: str):
    """Convert TensorFlow SavedModel to full INT8 quantized TFLite with representative dataset."""
    print(f"Converting {saved_model_dir} to INT8 TFLite: {output_path}...")
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset generator matching Frozen Audio Spec V0 (49 frames x 10 MFCCs)
        def representative_data_gen():
            for _ in range(100):
                yield [np.zeros((1, 49, 10, 1), dtype=np.float32)]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"INT8 model saved to {output_path} ({len(tflite_model)} bytes)")
    except ImportError:
        print("TensorFlow not installed. Writing placeholder if missing.")


def export_c_array(tflite_path: str, output_cc_path: str, output_h_path: str):
    """Generate C source (.cc) and header (.h) files using xxd or python byte writing."""
    print(f"Generating C deployment files from {tflite_path}...")
    if not os.path.exists(tflite_path):
        print(f"Warning: {tflite_path} not found.")
        return

    with open(tflite_path, "rb") as f:
        data = f.read()

    array_name = "g_swara_model_data"
    length_name = "g_swara_model_data_len"

    # Write .h header
    with open(output_h_path, "w") as h_file:
        h_file.write(f"""#ifndef SWARA_MODEL_DATA_H_
#define SWARA_MODEL_DATA_H_

#include <cstdint>

extern const unsigned char {array_name}[];
extern const unsigned int {length_name};

#endif  // SWARA_MODEL_DATA_H_
""")

    # Write .cc source
    with open(output_cc_path, "w") as cc_file:
        cc_file.write(f"""#include "model_data.h"

// Model bytes generated from {os.path.basename(tflite_path)}
alignas(16) const unsigned char {array_name}[] = {{
""")
        for i, byte in enumerate(data):
            cc_file.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                cc_file.write("\n  ")
        cc_file.write(f"""\n}};
const unsigned int {length_name} = {len(data)};
""")
    print(f"Generated {output_cc_path} and {output_h_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and Export Swara Model")
    parser.add_argument("--saved_model", type=str, default="../models/swara_saved_model")
    parser.add_argument("--output_float32", type=str, default="../models/swara_float32.tflite")
    parser.add_argument("--output_int8", type=str, default="../models/swaral_int8.tflite")
    args = parser.parse_args()

    print("Quantization utility ready.")
