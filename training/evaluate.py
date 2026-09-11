"""
Evaluation and benchmark script for Swara models (Production-Ready Infrastructure).
Computes overall accuracy, per-class accuracy, confusion matrix, precision, recall, and F1.

IMPORTANT:
- Distinguishes RECORDING-LEVEL evaluation from speaker-independent evaluation.
- Supports evaluating both Keras SavedModel/H5 models and quantized TFLite models.
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple
import numpy as np

from dataset import SwaraDataset, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes confusion matrix, overall accuracy, per-class accuracy, precision, recall, and F1 score.
    """
    num_classes = len(CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total_samples = len(y_true)
    overall_accuracy = float(np.trace(cm)) / float(total_samples) if total_samples > 0 else 0.0

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        class_name = IDX_TO_CLASS[c]
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = np.sum(cm[c, :])

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = float(tp) / float(support) if support > 0 else 0.0

        per_class_metrics[class_name] = {
            "support": int(support),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "total_samples": total_samples,
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": cm,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
    }


def evaluate_tflite_model(tflite_path: str, X_test: np.ndarray) -> np.ndarray:
    """Run inference over X_test using TFLite interpreter."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))
    is_int8 = input_details["dtype"] == np.int8

    predictions = []
    for i in range(len(X_test)):
        sample = np.expand_dims(X_test[i], axis=0)
        if is_int8 and input_scale > 0.0:
            sample_quant = np.clip(np.round(sample / input_scale) + input_zero_point, -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details["index"], sample_quant)
        else:
            interpreter.set_tensor(input_details["index"], sample.astype(input_details["dtype"]))

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])

        if is_int8 and output_scale > 0.0:
            # Dequantize or argmax
            logits = (output_data.astype(np.float32) - output_zero_point) * output_scale
        else:
            logits = output_data

        pred_class = int(np.argmax(logits, axis=-1)[0])
        predictions.append(pred_class)

    return np.array(predictions, dtype=np.int32)


def evaluate(
    model_path: str,
    data_dir: str = "data/raw",
    split: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate trained model or TFLite model on the specified dataset split.
    """
    print("=" * 65)
    print("           Swara Wake Word Evaluation Pipeline (M2c.1)          ")
    print("=" * 65)
    print(" [CRITICAL AUDIT NOTICE]")
    print(" Evaluation metric is strictly: RECORDING-LEVEL")
    print(" It is NOT speaker-independent because speaker IDs are unavailable.")
    print("=" * 65)
    print(f"Model path:     {os.path.abspath(model_path)}")
    print(f"Data source:    {os.path.abspath(data_dir)}")
    print(f"Target split:   {split}")
    print("=" * 65)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file not found at: {model_path}")

    dataset_loader = SwaraDataset(data_dir=data_dir)
    splits = dataset_loader.scan_dataset()
    total_files = dataset_loader.count_total_files(splits)

    if total_files == 0:
        raise FileNotFoundError(
            f"[ERROR] No valid audio files found in '{data_dir}'.\n"
            "Real evaluation dataset required."
        )

    eval_entries = splits.get(split, [])
    if len(eval_entries) == 0:
        raise ValueError(f"[ERROR] Split '{split}' contains 0 recordings.")

    print(f"Extracting features for {len(eval_entries)} recordings in '{split}' split...")
    X_eval, y_true = dataset_loader.load_tensors_from_file_list(eval_entries)

    # Predict
    if model_path.endswith(".tflite"):
        y_pred = evaluate_tflite_model(model_path, X_eval)
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        probabilities = model.predict(X_eval)
        y_pred = np.argmax(probabilities, axis=-1).astype(np.int32)

    results = compute_classification_metrics(y_true, y_pred)

    print("\n" + "-" * 45)
    print(f"EVALUATION RESULTS (Split: {split.upper()})")
    print("-" * 45)
    print(f"Total Evaluated Samples: {results['total_samples']}")
    print(f"Overall Accuracy:        {results['overall_accuracy'] * 100:.2f}%")
    print(f"Macro Precision:         {results['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:            {results['macro_recall'] * 100:.2f}%")
    print(f"Macro F1 Score:          {results['macro_f1'] * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cls_name in CLASSES:
        m = results["per_class"][cls_name]
        print(f"  [{cls_name.upper():<7}] Support: {m['support']:<4} | Acc: {m['accuracy']*100:.1f}% | Prec: {m['precision']*100:.1f}% | Rec: {m['recall']*100:.1f}% | F1: {m['f1']*100:.1f}%")

    print("\nConfusion Matrix (Rows = True, Cols = Predicted):")
    print(f"            {'silence':<9}{'unknown':<9}{'swara':<9}")
    for i, cls_name in enumerate(CLASSES):
        row_str = "  ".join(f"{results['confusion_matrix'][i, j]:<7}" for j in range(len(CLASSES)))
        print(f"  {cls_name:<8}  {row_str}")
    print("-" * 45)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Swara Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .keras, SavedModel, or .tflite")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset root")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    args = parser.parse_args()

    evaluate(model_path=args.model_path, data_dir=args.data_dir, split=args.split)
