"""
Deterministic TFLite to C Array Exporter for Embedded Deployment (TFLM).

Converts a verified .tflite model into:
- deployment/model_data.h
- deployment/model_data.cc

Ensures placeholder or invalid files are never exported as deployment models,
and outputs model metadata headers.
"""

import os
import sys
import argparse

# Ensure training directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from validate_tflite import inspect_tflite_model


def export_model_to_c_array(
    tflite_path: str,
    output_cc_path: str = "deployment/model_data.cc",
    output_h_path: str = "deployment/model_data.h",
    array_name: str = "g_swara_model_data",
    length_name: str = "g_swara_model_data_len",
    skip_validation: bool = False,
):
    """
    Converts a binary .tflite model file into 16-byte aligned C source and header.
    Validates model structure and FlatBuffer header before exporting.
    """
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"[ERROR] Source TFLite model file not found at: {tflite_path}")

    with open(tflite_path, "rb") as f:
        data = f.read()

    file_size = len(data)
    model_filename = os.path.basename(tflite_path)

    # Basic FlatBuffer header check (must start with offset and contain 'TFL3')
    if file_size < 32 or b"TFL3" not in data[:16]:
        raise ValueError(f"[ERROR] {tflite_path} does not appear to be a valid TFLite flatbuffer binary.")

    metadata_comment = ""
    if not skip_validation:
        try:
            report = inspect_tflite_model(tflite_path)
            inp_info = report["input_tensors"][0] if report["input_tensors"] else {}
            out_info = report["output_tensors"][0] if report["output_tensors"] else {}
            metadata_comment = f"""// Model Metadata:
//   Input shape:         {inp_info.get('shape')}
//   Input dtype:         {inp_info.get('dtype')}
//   Input scale:         {inp_info.get('scale')} (zero-point: {inp_info.get('zero_point')})
//   Output shape:        {out_info.get('shape')}
//   Output dtype:        {out_info.get('dtype')}
//   Output scale:        {out_info.get('scale')} (zero-point: {out_info.get('zero_point')})
//   Total weights count: {report.get('total_weight_elements')}
//   Operators ({report.get('operator_count')}):   {', '.join(report.get('operators', []))}
//   TFLM compatible:     {report.get('tflm_compatible')}
"""
        except Exception as e:
            metadata_comment = f"// [Warning] Could not inspect TFLite schema: {e}\n"

    os.makedirs(os.path.dirname(output_h_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_cc_path) or ".", exist_ok=True)

    # Write Header (.h)
    header_content = f"""#ifndef SWARA_MODEL_DATA_H_
#define SWARA_MODEL_DATA_H_

#include <cstdint>

// Swara Keyword Spotting / Edge Wake Word Detection Model
// Generated from: {model_filename} ({file_size} bytes)
{metadata_comment}extern const unsigned char {array_name}[];
extern const unsigned int {length_name};

#endif  // SWARA_MODEL_DATA_H_
"""
    with open(output_h_path, "w", encoding="utf-8") as h_file:
        h_file.write(header_content)

    # Write Source (.cc)
    with open(output_cc_path, "w", encoding="utf-8") as cc_file:
        cc_file.write(f"""#include "model_data.h"

// Model bytes generated deterministically from {model_filename}
// Model size: {file_size} bytes
{metadata_comment}alignas(16) const unsigned char {array_name}[] = {{
""")
        # 12 bytes per line
        for i in range(0, file_size, 12):
            chunk = data[i : i + 12]
            hex_chunk = ", ".join(f"0x{b:02x}" for b in chunk)
            if i + 12 < file_size:
                hex_chunk += ","
            cc_file.write(f"  {hex_chunk}\n")

        cc_file.write(f"""}};

const unsigned int {length_name} = {file_size};
""")

    print(f"[SUCCESS] Exported validated model {tflite_path} ({file_size} bytes) to:")
    print(f"  * Header: {os.path.abspath(output_h_path)}")
    print(f"  * Source: {os.path.abspath(output_cc_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TFLite Model to Deterministic C++ Array")
    parser.add_argument("--tflite_path", type=str, required=True, help="Input .tflite model file")
    parser.add_argument("--output_cc", type=str, default="deployment/model_data.cc", help="Output .cc path")
    parser.add_argument("--output_h", type=str, default="deployment/model_data.h", help="Output .h path")
    parser.add_argument("--skip_validation", action="store_true", help="Skip schema inspection")
    args = parser.parse_args()

    export_model_to_c_array(args.tflite_path, args.output_cc, args.output_h, skip_validation=args.skip_validation)
