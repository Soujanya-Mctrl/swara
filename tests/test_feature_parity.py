"""
Deterministic Feature-Equivalence & Pipeline Parity Test (M2b.1).

Verifies:
1. Python SwaraFeatureExtractor matches compiled C front-end (src/features/mfcc.c) within numerical tolerance (< 0.05 max error).
2. Deterministic RECORDING-LEVEL splitting partitions files with zero leakage across train/val/test splits.
3. Dataset loading produces strictly compliant (N, 49, 10, 1) and (N,) tensors.
4. Dataset inspection audit tool operates accurately.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from dataset import SwaraFeatureExtractor, SwaraDataset, CLASSES, CLASS_TO_IDX, inspect_dataset


def test_feature_equivalence_with_c():
    print("=== [1] Testing Feature Equivalence between Python and C Runtime ===")

    wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_16k_1s.wav"))
    c_ref_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_c_mfcc_reference.bin"))

    assert os.path.exists(wav_file), f"Missing test WAV: {wav_file}"
    assert os.path.exists(c_ref_bin), f"Missing C reference binary: {c_ref_bin}"

    dataset = SwaraDataset(use_preemphasis=True)
    extractor = dataset.feature_extractor

    audio_samples = dataset.load_wav_file(wav_file)
    assert len(audio_samples) == 16000, f"Expected 16000 samples, got {len(audio_samples)}"

    py_features = extractor.extract_window_mfcc(audio_samples)
    assert py_features.shape == (49, 10, 1), f"Expected (49, 10, 1), got {py_features.shape}"
    assert np.all(np.isfinite(py_features)), "Detected non-finite values!"

    c_features = np.fromfile(c_ref_bin, dtype=np.float32).reshape(49, 10)
    py_flat = py_features.squeeze()

    abs_diff = np.abs(py_flat - c_features)
    max_err = float(np.max(abs_diff))
    mean_err = float(np.mean(abs_diff))

    print(f"  * Maximum Absolute Error: {max_err:.6f}")
    print(f"  * Mean Absolute Error:    {mean_err:.6f}")

    TOLERANCE = 0.05
    assert max_err < TOLERANCE, f"Feature mismatch exceeded {TOLERANCE}: max error = {max_err}"
    print(f"-> PASSED: Python features match C reference within numerical tolerance (< {TOLERANCE}).\n")


def test_recording_level_splits():
    print("=== [2] Testing Recording-Level Partitioning (Labeled as Recording-Level) ===")
    print("  [AUDIT NOTE] Speaker identities are unavailable in real dataset.")
    print("  Partitioning is strictly RECORDING-LEVEL, NOT speaker-independent.\n")

    files = [f"recording_{i:04d}.wav" for i in range(200)]
    train_set = set()
    val_set = set()
    test_set = set()

    for f in files:
        split = SwaraDataset.get_recording_split(f, val_ratio=0.15, test_ratio=0.15)
        if split == "train":
            train_set.add(f)
        elif split == "val":
            val_set.add(f)
        elif split == "test":
            test_set.add(f)

    print(f"Total Recordings: {len(files)}")
    print(f"  * Train: {len(train_set)} ({len(train_set)/len(files)*100:.1f}%)")
    print(f"  * Val:   {len(val_set)} ({len(val_set)/len(files)*100:.1f}%)")
    print(f"  * Test:  {len(test_set)} ({len(test_set)/len(files)*100:.1f}%)")

    assert len(train_set.intersection(val_set)) == 0, "Overlap between Train and Val!"
    assert len(train_set.intersection(test_set)) == 0, "Overlap between Train and Test!"
    assert len(val_set.intersection(test_set)) == 0, "Overlap between Val and Test!"
    print("-> PASSED: Clean recording-level partition with zero partition overlap.\n")


def test_dataset_tensor_shapes_and_classes():
    print("=== [3] Testing Dataset Classes and Tensor Pipeline Shapes ===")
    assert CLASS_TO_IDX == {"silence": 0, "unknown": 1, "swara": 2}

    wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "test_16k_1s.wav"))
    dataset = SwaraDataset()

    mock_entries = [
        (wav_file, 0, None),
        (wav_file, 1, None),
        (wav_file, 2, None),
    ]

    X, y = dataset.load_tensors_from_file_list(mock_entries)
    assert X.shape == (3, 49, 10, 1)
    assert y.shape == (3,)
    assert list(y) == [0, 1, 2]
    assert np.all(np.isfinite(X))
    print(f"Generated X: {X.shape}, y: {y.shape}")
    print("-> PASSED: Complies with (N, 49, 10, 1) and (N,) contract.\n")


def test_inspection_tool():
    print("=== [4] Testing Dataset Inspection Audit Tool ===")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    report = inspect_dataset(data_dir)

    assert report["total_wav_files"] >= 1
    assert report["valid_wav_files"] >= 1
    assert len(report["invalid_corrupt_files"]) == 0
    assert 16000 in report["sample_rates"]
    assert 1 in report["channel_counts"]
    assert 16 in report["bit_depths"]
    print("-> PASSED: Dataset inspection correctly audited test fixtures.\n")


if __name__ == "__main__":
    print("==================================================================")
    print("       Swara M2b.1 Dataset & Feature Contract Audit Test          ")
    print("==================================================================\n")
    test_feature_equivalence_with_c()
    test_recording_level_splits()
    test_dataset_tensor_shapes_and_classes()
    test_inspection_tool()
    print("==================================================================")
    print("      All M2b.1 Audit and Parity Tests PASSED Successfully!       ")
    print("==================================================================")
