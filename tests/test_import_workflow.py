"""
Test Suite for Swara Dataset Import Workflow (M2c).

Validates:
1. Manifest generation and structure (with speaker_id = 'unknown' and label = 'swara')
2. Content-based SHA-256 duplicate detection
3. WAV format inspection and validation
4. Deterministic import and collision handling
5. Strict read-only preservation of source files
"""

import os
import sys
import shutil
import tempfile
import wave
import struct
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from import_drive_dataset import (
    import_dataset,
    inspect_wav_header,
    compute_sha256,
)


def create_dummy_wav(filepath: str, sample_rate=16000, channels=1, num_samples=16000, val_mult=1):
    """Create a compliant dummy WAV file for testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = [int((i * val_mult) % 30000 - 15000) for i in range(num_samples)]
        wf.writeframes(struct.pack(f"<{num_samples}h", *samples))


def test_wav_validation():
    print("\n[TEST 1] Testing WAV Header Inspection & Validation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_wav = os.path.join(tmpdir, "valid.wav")
        create_dummy_wav(valid_wav, 16000, 1, 16000)

        bad_sr_wav = os.path.join(tmpdir, "bad_sr.wav")
        create_dummy_wav(bad_sr_wav, 44100, 1, 44100)

        corrupt_file = os.path.join(tmpdir, "corrupt.wav")
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_WAV_HEADER_CORRUPTED_BYTES")

        hdr_valid = inspect_wav_header(valid_wav)
        assert hdr_valid["valid_wav"] is True
        assert hdr_valid["sample_rate"] == 16000
        assert hdr_valid["channels"] == 1
        assert hdr_valid["bits_per_sample"] == 16
        assert hdr_valid["meets_swara_spec"] is True
        assert abs(hdr_valid["duration_sec"] - 1.0) < 1e-4

        hdr_bad = inspect_wav_header(bad_sr_wav)
        assert hdr_bad["valid_wav"] is True
        assert hdr_bad["meets_swara_spec"] is False  # Fails 16 kHz spec

        hdr_corrupt = inspect_wav_header(corrupt_file)
        assert hdr_corrupt["valid_wav"] is False
        assert hdr_corrupt["error"] is not None
        print("  -> Passed: Header validation accurately checks WAV compliance.")


def test_import_manifest_and_speaker_unknown():
    print("\n[TEST 2] Testing Manifest Generation & Unknown Speaker Preservation...")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        wav1 = os.path.join(src_dir, "recording_001.wav")
        create_dummy_wav(wav1, 16000, 1, 16000, val_mult=5)

        report = import_dataset(source=src_dir, dest_dir=dst_dir, drive_file_id="dummy_drive_id")
        assert report["successfully_imported"] == 1

        manifest_file = os.path.join(dst_dir, "manifest.json")
        assert os.path.exists(manifest_file), "Manifest JSON was not generated!"

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) == 1
        entry = manifest[0]
        assert entry["original_filename"] == "recording_001.wav"
        assert entry["dataset_label"] == "swara"
        assert entry["speaker_id"] == "unknown"  # STRICT REQUIREMENT
        assert entry["drive_file_id"] == "dummy_drive_id"
        assert len(entry["sha256"]) == 64
        assert entry["is_duplicate"] is False
        print("  -> Passed: Manifest correctly cataloged with speaker_id = 'unknown' and label = 'swara'.")


def test_duplicate_detection():
    print("\n[TEST 3] Testing SHA-256 Duplicate Content Detection...")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        # Create original and byte-identical duplicate with different filenames
        orig_file = os.path.join(src_dir, "take_A.wav")
        dup_file = os.path.join(src_dir, "take_B_copy.wav")
        create_dummy_wav(orig_file, 16000, 1, 16000, val_mult=7)
        shutil.copy2(orig_file, dup_file)

        assert compute_sha256(orig_file) == compute_sha256(dup_file)

        report = import_dataset(source=src_dir, dest_dir=dst_dir)
        assert report["total_files_found"] == 2
        assert report["successfully_imported"] == 1
        assert report["duplicate_files"] == 1

        manifest_file = os.path.join(dst_dir, "manifest.json")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # One original, one duplicate entry
        dup_entries = [e for e in manifest if e["is_duplicate"]]
        assert len(dup_entries) == 1
        assert dup_entries[0]["canonical_file"] is not None
        print(f"  -> Passed: Byte-identical copy detected and referenced to canonical: {dup_entries[0]['canonical_file']}")


def test_collision_handling():
    print("\n[TEST 4] Testing Filename Collision Handling (Different Content)...")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        sub1 = os.path.join(src_dir, "folder1")
        sub2 = os.path.join(src_dir, "folder2")
        os.makedirs(sub1)
        os.makedirs(sub2)

        # Same filename, different audio content
        file1 = os.path.join(sub1, "swara_sample.wav")
        file2 = os.path.join(sub2, "swara_sample.wav")
        create_dummy_wav(file1, 16000, 1, 16000, val_mult=3)
        create_dummy_wav(file2, 16000, 1, 16000, val_mult=11)

        assert compute_sha256(file1) != compute_sha256(file2)

        report = import_dataset(source=src_dir, dest_dir=dst_dir)
        assert report["total_files_found"] == 2
        assert report["successfully_imported"] == 2
        assert report["duplicate_files"] == 0

        imported_files = os.listdir(dst_dir)
        wav_files = [f for f in imported_files if f.endswith(".wav")]
        assert len(wav_files) == 2
        print(f"  -> Passed: Both distinct files preserved without overwrite: {wav_files}")


def test_source_preservation():
    print("\n[TEST 5] Testing Read-Only Source File Preservation...")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        test_wav = os.path.join(src_dir, "source_original.wav")
        create_dummy_wav(test_wav, 16000, 1, 16000)

        mtime_before = os.path.getmtime(test_wav)
        sha_before = compute_sha256(test_wav)

        report = import_dataset(source=src_dir, dest_dir=dst_dir)

        mtime_after = os.path.getmtime(test_wav)
        sha_after = compute_sha256(test_wav)

        assert report["source_preservation_verified"] is True
        assert mtime_before == mtime_after, "Source modification timestamp changed!"
        assert sha_before == sha_after, "Source file content altered!"
        print("  -> Passed: Source files remain strictly read-only and unmodified.")


if __name__ == "__main__":
    print("==================================================================")
    print("      Running Swara Dataset Import Workflow Test Suite            ")
    print("==================================================================")
    test_wav_validation()
    test_import_manifest_and_speaker_unknown()
    test_duplicate_detection()
    test_collision_handling()
    test_source_preservation()
    print("==================================================================")
    print("        All Dataset Import Tests PASSED Successfully!             ")
    print("==================================================================")
