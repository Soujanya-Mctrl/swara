"""
CLI tool for Swara dataset inspection and audit.

Usage:
    python training/inspect_dataset.py --data_dir data/raw
    python training/inspect_dataset.py --data_dir /path/to/google_drive_dataset
"""

import sys
import os
import argparse
import numpy as np

# Ensure training directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dataset import inspect_dataset
from config import CLASSES


def main():
    parser = argparse.ArgumentParser(description="Swara Dataset Integrity Inspector & Audit Tool")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory containing audio dataset")
    args = parser.parse_args()

    print("=" * 68)
    print("           Swara Dataset Audit & Integrity Report                ")
    print("=" * 68)
    print(f"Target Directory: {os.path.abspath(args.data_dir)}\n")

    report = inspect_dataset(args.data_dir)

    if "error" in report:
        print(f"[FATAL AUDIT ERROR] {report['error']}")
        sys.exit(1)

    print(f"1. Total WAV Files Found:       {report['total_wav_files']}")
    print(f"2. Valid WAV Files:             {report['valid_wav_files']}")
    print(f"3. Usable Recordings (Contract): {report['usable_recordings']} (16kHz, mono, 16-bit PCM)")
    print(f"4. Invalid / Corrupt Files:     {len(report['invalid_corrupt_files'])}")
    if report["invalid_corrupt_files"]:
        for item in report["invalid_corrupt_files"][:10]:
            print(f"     - {item['file']}: {item['error']}")
        if len(report["invalid_corrupt_files"]) > 10:
            print(f"     ... and {len(report['invalid_corrupt_files']) - 10} more.")

    print("\n5. Class Distribution & Imbalance:")
    for cls_name in CLASSES:
        info = report["class_breakdown"][cls_name]
        total = info["total"]
        valid = info["valid"]
        dur = info["duration_sec"]
        status = "[OK]" if valid > 0 else "[MISSING!]"
        print(f"     * {cls_name:<8} {status:<10} Total: {total:<5} | Valid: {valid:<5} | Duration: {dur/60.0:.2f} min ({dur:.1f} s)")

    print(f"     * Max/Min Imbalance Ratio:  {report['class_imbalance_ratio']:.2f}x")

    print("\n6. Sample Rate Distribution:")
    if report["sample_rates"]:
        for sr, count in sorted(report["sample_rates"].items()):
            compliance = "[COMPLIANT]" if sr == 16000 else "[NON-COMPLIANT]"
            print(f"     * {sr} Hz: {count} files ({count/report['valid_wav_files']*100:.1f}%) {compliance}")
    else:
        print("     (No audio files found)")

    print("\n7. Channel Count Distribution:")
    if report["channel_counts"]:
        for ch, count in sorted(report["channel_counts"].items()):
            ch_name = "Mono (1-ch)" if ch == 1 else ("Stereo (2-ch)" if ch == 2 else f"{ch} channels")
            compliance = "[COMPLIANT]" if ch == 1 else "[NON-COMPLIANT]"
            print(f"     * {ch_name}: {count} files ({count/report['valid_wav_files']*100:.1f}%) {compliance}")
    else:
        print("     (No audio files found)")

    print("\n8. Bit Depth Distribution:")
    if report["bit_depths"]:
        for bd, count in sorted(report["bit_depths"].items()):
            compliance = "[COMPLIANT]" if bd == 16 else "[NON-COMPLIANT]"
            print(f"     * {bd}-bit: {count} files ({count/report['valid_wav_files']*100:.1f}%) {compliance}")
    else:
        print("     (No audio files found)")

    print("\n9. Duration Analysis:")
    durations = report["durations_sec"]
    if durations:
        d_arr = np.array(durations)
        print(f"     * Min Duration:     {np.min(d_arr):.3f} s")
        print(f"     * Max Duration:     {np.max(d_arr):.3f} s")
        print(f"     * Mean Duration:    {np.mean(d_arr):.3f} s")
        print(f"     * Median Duration:  {np.median(d_arr):.3f} s")
        print(f"     * Std Dev Duration: {np.std(d_arr):.3f} s")
        print(f"     * Files < 1.0s:     {report['files_shorter_than_1s']} (padded)")
        print(f"     * Files == 1.0s:    {report['files_exact_1s']}")
        print(f"     * Files > 1.0s:     {report['files_longer_than_1s']} (centered/truncated)")
    else:
        print("     (No audio files found)")

    print("\n10. Duplication Analysis:")
    print(f"     * Duplicate Copies: {report['duplicate_files_count']}")
    if report["duplicate_groups"]:
        for grp in report["duplicate_groups"][:5]:
            print(f"         MD5 {grp['md5']}: {len(grp['copies'])} copies")

    print("\n" + "=" * 68)
    # Final Readiness Verdict
    missing_classes = [c for c in CLASSES if report["class_breakdown"][c]["valid"] == 0]
    if missing_classes:
        print(f">> AUDIT VERDICT: NOT READY FOR TRAINING. Missing required classes: {missing_classes}")
        sys.exit(2)
    elif report["usable_recordings"] == 0:
        print(">> AUDIT VERDICT: NOT READY FOR TRAINING. 0 usable recordings conforming to 16kHz/mono/16-bit.")
        sys.exit(3)
    else:
        print(f">> AUDIT VERDICT: READY. {report['usable_recordings']} compliant audio recordings ready for training.")
    print("=" * 68)


if __name__ == "__main__":
    main()
