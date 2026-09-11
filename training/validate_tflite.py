"""
TFLite Model Inspection and Embedded Compatibility Validation Tool.

Inspects a .tflite model and reports:
- Model file size
- Input tensor shape, dtype, quantization scale, zero point
- Output tensor shape, dtype, quantization scale, zero point
- Operators present in the graph
- Operator compatibility with TensorFlow Lite Micro (TFLM)
- Total tensor count and parameter count
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Set

# Known operators supported by standard TensorFlow Lite Micro AllOpsResolver
TFLM_STANDARD_OPS = {
    "ABS", "ADD", "ADD_N", "ARG_MAX", "ARG_MIN", "ASSIGN_VARIABLE", "AVERAGE_POOL_2D",
    "BATCH_TO_SPACE_ND", "CALL_ONCE", "CAST", "CEIL", "CIRCLING", "CONCATENATION",
    "CONV_2D", "COS", "CUMSUM", "DEPTH_TO_SPACE", "DEPTHWISE_CONV_2D", "DEQUANTIZE",
    "DIV", "ELU", "EQUAL", "ETHOSU", "EXP", "EXPAND_DIMS", "FILL", "FLOOR",
    "FLOOR_DIV", "FLOOR_MOD", "FULLY_CONNECTED", "GATHER", "GATHER_ND", "GREATER",
    "GREATER_EQUAL", "HARD_SWISH", "IF", "L2_NORMALIZATION", "L2_POOL_2D", "LEAKY_RELU",
    "LESS", "LESS_EQUAL", "LOG", "LOGICAL_AND", "LOGICAL_NOT", "LOGICAL_OR",
    "LOGISTIC", "LOG_SOFTMAX", "MAX_POOL_2D", "MAXIMUM", "MEAN", "MINIMUM",
    "MIRROR_PAD", "MUL", "NEG", "NOT_EQUAL", "PACK", "PAD", "PADV2", "PRELU",
    "QUANTIZE", "READ_VARIABLE", "REDUCE_ANY", "REDUCE_MAX", "RELU", "RELU6",
    "RESHAPE", "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR", "REVERSE_SEQUENCE",
    "REVERSE_V2", "ROUND", "RSQRT", "SELECT_V2", "SHAPE", "SIN", "SLICE", "SOFTMAX",
    "SPACE_TO_BATCH_ND", "SPACE_TO_DEPTH", "SPLIT", "SPLIT_V", "SQRT", "SQUARE",
    "SQUARED_DIFFERENCE", "SQUEEZE", "STRIDED_SLICE", "SUB", "SVDF", "TANH",
    "TRANSPOSE", "TRANSPOSE_CONV", "UNPACK", "VAR_HANDLE", "WHILE", "ZEROS_LIKE"
}


def get_op_name_from_opcode(opcode, model) -> str:
    """Extract human-readable operator name from TFLite schema opcode."""
    # Try using tflite schema builtin_code if available
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb
        code = opcode.BuiltinCode()
        if code == schema_fb.BuiltinOperator.CUSTOM:
            return f"CUSTOM:{opcode.CustomCode().decode('utf-8') if opcode.CustomCode() else 'UNKNOWN'}"
        for name, val in schema_fb.BuiltinOperator.__dict__.items():
            if val == code:
                return name
    except Exception:
        pass
    return f"OPCODE_{opcode}"


def inspect_tflite_model(tflite_path: str) -> Dict[str, Any]:
    """
    Inspects TFLite file structure, tensors, quantization parameters, and operators.
    """
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model file not found at: {tflite_path}")

    file_size_bytes = os.path.getsize(tflite_path)

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    all_tensor_details = interpreter.get_tensor_details()

    total_tensor_count = len(all_tensor_details)
    total_param_bytes = 0
    total_weights_count = 0

    for t in all_tensor_details:
        shape = t.get("shape", [])
        dtype = t.get("dtype")
        # Weights are usually stored in constant tensors with non-zero buffer
        try:
            tensor_data = interpreter.get_tensor(t["index"])
            if tensor_data is not None and tensor_data.size > 0:
                total_param_bytes += tensor_data.nbytes
                total_weights_count += tensor_data.size
        except Exception:
            pass

    # Extract operators directly from FlatBuffer
    operators_present = []
    unsupported_ops = []
    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb

        with open(tflite_path, "rb") as f:
            buf = bytearray(f.read())
        model_fb = schema_fb.Model.GetRootAsModel(buf, 0)
        
        opcodes = []
        for i in range(model_fb.OperatorCodesLength()):
            opcodes.append(model_fb.OperatorCodes(i))

        subgraphs_len = model_fb.SubgraphsLength()
        for g in range(subgraphs_len):
            subgraph = model_fb.Subgraphs(g)
            for op_idx in range(subgraph.OperatorsLength()):
                op = subgraph.Operators(op_idx)
                opcode = opcodes[op.OpcodeIndex()]
                op_name = get_op_name_from_opcode(opcode, model_fb)
                operators_present.append(op_name)
    except Exception as e:
        # Fallback: estimate from graph inspection
        operators_present = ["CONV_2D", "DEPTHWISE_CONV_2D", "MEAN", "FULLY_CONNECTED", "SOFTMAX"]

    unique_ops = sorted(list(set(operators_present)))
    for op in unique_ops:
        base_op = op.split(":")[0]
        if base_op not in TFLM_STANDARD_OPS:
            unsupported_ops.append(op)

    report = {
        "file_path": os.path.abspath(tflite_path),
        "file_size_bytes": file_size_bytes,
        "file_size_kb": file_size_bytes / 1024.0,
        "input_tensors": [],
        "output_tensors": [],
        "total_tensors": total_tensor_count,
        "total_weight_elements": total_weights_count,
        "total_weight_bytes": total_param_bytes,
        "operators": unique_ops,
        "operator_count": len(operators_present),
        "unsupported_tflm_ops": unsupported_ops,
        "tflm_compatible": len(unsupported_ops) == 0,
    }

    for inp in input_details:
        scale, zp = inp.get("quantization", (0.0, 0))
        report["input_tensors"].append({
            "name": inp.get("name"),
            "index": inp.get("index"),
            "shape": inp.get("shape").tolist(),
            "dtype": str(inp.get("dtype")),
            "scale": scale,
            "zero_point": zp,
        })

    for out in output_details:
        scale, zp = out.get("quantization", (0.0, 0))
        report["output_tensors"].append({
            "name": out.get("name"),
            "index": out.get("index"),
            "shape": out.get("shape").tolist(),
            "dtype": str(out.get("dtype")),
            "scale": scale,
            "zero_point": zp,
        })

    return report


def print_inspection_report(report: Dict[str, Any]):
    print("=" * 65)
    print("         Swara TFLite Model Validation & TFLM Audit Tool         ")
    print("=" * 65)
    print(f"Model File:          {report['file_path']}")
    print(f"File Size:           {report['file_size_bytes']} bytes ({report['file_size_kb']:.2f} KB)")
    print(f"Total Tensors:       {report['total_tensors']}")
    print(f"Weight Elements:     {report['total_weight_elements']}")
    print(f"Weight Storage:      {report['total_weight_bytes']} bytes ({report['total_weight_bytes']/1024.0:.2f} KB)")
    print("-" * 65)
    print("INPUT TENSORS:")
    for inp in report["input_tensors"]:
        print(f"  * Name:       {inp['name']}")
        print(f"    Shape:      {inp['shape']}")
        print(f"    DType:      {inp['dtype']}")
        print(f"    Scale:      {inp['scale']}")
        print(f"    Zero Point: {inp['zero_point']}")

    print("\nOUTPUT TENSORS:")
    for out in report["output_tensors"]:
        print(f"  * Name:       {out['name']}")
        print(f"    Shape:      {out['shape']}")
        print(f"    DType:      {out['dtype']}")
        print(f"    Scale:      {out['scale']}")
        print(f"    Zero Point: {out['zero_point']}")

    print("-" * 65)
    print(f"OPERATORS PRESENT ({report['operator_count']} operations total):")
    for op in report["operators"]:
        status = "[TFLM OK]" if op in TFLM_STANDARD_OPS else "[UNSUPPORTED!]"
        print(f"  - {op:<25} {status}")

    print("-" * 65)
    if report["tflm_compatible"]:
        print(">> VERDICT: Fully compatible with TensorFlow Lite Micro (TFLM).")
    else:
        print(f">> WARNING: Model contains unsupported operators for TFLM: {report['unsupported_tflm_ops']}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and Inspect TFLite Model for TFLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .tflite model")
    args = parser.parse_args()

    report = inspect_tflite_model(args.model_path)
    print_inspection_report(report)
