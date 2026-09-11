"""
Swara Google Drive & Local Dataset Import Workflow (M2c).

Imports, validates, deduplicates, and catalogs WAV recordings into:
    data/raw/swara/

Features:
- Configurable source: Local folder or Google Drive folder ID/URL (via gdown)
- Strictly READ-ONLY with respect to source (never modifies, deletes, or moves source files)
- Full RIFF/WAVE header inspection and format verification against Swara V0 spec
- SHA-256 hash calculation for content-based duplicate detection
- Collision-resistant deterministic local file naming
- Manifest generation (data/raw/swara/manifest.json) with speaker_id = 'unknown'
- Comprehensive summary audit report
"""

import os
import sys
import shutil
import wave
import struct
import hashlib
import datetime
import json
import re
from typing import Dict, List, Any, Optional, Tuple


SWARA_SPEC = {
    "sample_rate": 16000,
    "channels": 1,
    "bits_per_sample": 16,
    "min_samples": 16000,
    "target_duration_sec": 1.0,
}


def safe_relpath(path: str, start: Optional[str] = None) -> str:
    """Format path as relative if on same drive, otherwise normalized absolute."""
    if path is None:
        return None
    try:
        start_dir = start if start is not None else os.getcwd()
        return os.path.relpath(path, start_dir).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Parse a .env file into key-value pairs without external dependencies."""
    env_vars: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return env_vars

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            env_vars[key] = val
    return env_vars


def inspect_wav_header(filepath: str) -> Dict[str, Any]:
    """
    Examine RIFF/WAVE header and extract technical parameters.
    Does not use heavy third-party audio packages.
    """
    res = {
        "valid_wav": False,
        "is_pcm": False,
        "sample_rate": 0,
        "channels": 0,
        "bits_per_sample": 0,
        "num_samples": 0,
        "duration_sec": 0.0,
        "file_size_bytes": 0,
        "error": None,
        "meets_swara_spec": False,
    }

    try:
        res["file_size_bytes"] = os.path.getsize(filepath)
        with wave.open(filepath, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()

            res["valid_wav"] = True
            res["channels"] = channels
            res["sample_rate"] = sample_rate
            res["bits_per_sample"] = sample_width * 8
            res["num_samples"] = num_frames
            res["duration_sec"] = float(num_frames) / float(sample_rate) if sample_rate > 0 else 0.0
            res["is_pcm"] = True  # wave module in Python standard library only handles PCM format 1

            meets_spec = (
                sample_rate == SWARA_SPEC["sample_rate"]
                and channels == SWARA_SPEC["channels"]
                and (sample_width * 8) == SWARA_SPEC["bits_per_sample"]
            )
            res["meets_swara_spec"] = meets_spec
    except Exception as e:
        res["valid_wav"] = False
        res["error"] = str(e)

    return res


def download_google_drive_folder(folder_id_or_url: str, staging_dir: str) -> str:
    """
    Download a Google Drive folder using gdown into a read-only staging directory.
    Source Drive files are never modified or renamed.
    """
    try:
        import gdown
    except ImportError:
        raise RuntimeError("gdown is required to download from Google Drive URL/ID. Run: pip install gdown")

    os.makedirs(staging_dir, exist_ok=True)
    print(f"[DRIVE] Downloading Google Drive folder to staging area: {staging_dir}...")

    # Extract ID if a full URL was passed
    folder_url = folder_id_or_url
    if "drive.google.com" in folder_id_or_url and "folders/" in folder_id_or_url:
        match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_id_or_url)
        if match:
            folder_id = match.group(1)
            folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    gdown.download_folder(url=folder_url, output=staging_dir, quiet=False, use_cookies=False)
    return staging_dir


def import_dataset(
    source: str,
    dest_dir: str = "data/raw/swara",
    drive_file_id: Optional[str] = None,
    staging_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main dataset import workflow.
    - source: Can be a local folder path OR a Google Drive URL / Folder ID.
    - dest_dir: Destination folder for imported files (default: data/raw/swara).
    - Returns: Import audit report dictionary.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(dest_dir, exist_ok=True)

    # 1. Resolve source location
    source_is_remote_drive = (
        "drive.google.com" in source
        or len(source) > 20 and not os.path.exists(source) and os.sep not in source
    )

    actual_source_dir = source
    if source_is_remote_drive:
        if staging_dir is None:
            staging_dir = os.path.abspath(os.path.join(dest_dir, "..", ".gdrive_staging"))
        actual_source_dir = download_google_drive_folder(source, staging_dir)

    if not os.path.exists(actual_source_dir):
        raise FileNotFoundError(f"Source folder not found: {actual_source_dir}")

    # 2. Discover all WAV files in source (read-only traversal)
    source_files: List[str] = []
    for root, _, files in os.walk(actual_source_dir):
        # Do not recurse into staging inside destination
        if os.path.abspath(root).startswith(os.path.abspath(dest_dir)):
            continue
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                source_files.append(os.path.join(root, f))

    # 3. Process each file into dest_dir
    manifest_entries: List[Dict[str, Any]] = []
    known_hashes: Dict[str, str] = {}  # sha256 -> canonical local path
    allocated_filenames: set = set()

    # Pre-populate known hashes and allocated names from any existing files in dest_dir
    for f in os.listdir(dest_dir):
        if f.lower().endswith(".wav"):
            full_p = os.path.join(dest_dir, f)
            h = compute_sha256(full_p)
            known_hashes[h] = full_p
            allocated_filenames.add(f.lower())

    stats = {
        "total_source_files": len(source_files),
        "successfully_imported": 0,
        "duplicate_files_count": 0,
        "corrupt_files_count": 0,
        "skipped_files_count": 0,
        "sample_rate_dist": {},
        "channel_dist": {},
        "bit_depth_dist": {},
        "durations_sec": [],
        "files_shorter_than_1s": 0,
        "files_longer_than_1s": 0,
        "files_exact_1s": 0,
        "duplicate_records": [],
        "source_read_only_verified": True,
    }

    # Record source file timestamps before import to verify source remains unmodified
    source_mtimes_before = {p: os.path.getmtime(p) for p in source_files}

    for src_path in source_files:
        orig_filename = os.path.basename(src_path)
        sha256_hash = compute_sha256(src_path)
        file_size = os.path.getsize(src_path)
        hdr = inspect_wav_header(src_path)

        if not hdr["valid_wav"]:
            stats["corrupt_files_count"] += 1
            manifest_entries.append({
                "local_path": None,
                "original_filename": orig_filename,
                "source_path": src_path,
                "drive_file_id": drive_file_id,
                "sha256": sha256_hash,
                "file_size_bytes": file_size,
                "import_timestamp": timestamp,
                "dataset_label": "swara",
                "speaker_id": "unknown",
                "is_valid_wav": False,
                "error": hdr["error"],
                "is_duplicate": False,
                "canonical_file": None,
            })
            continue

        # Accumulate statistics
        sr = hdr["sample_rate"]
        ch = hdr["channels"]
        bd = hdr["bits_per_sample"]
        dur = hdr["duration_sec"]

        stats["sample_rate_dist"][sr] = stats["sample_rate_dist"].get(sr, 0) + 1
        stats["channel_dist"][ch] = stats["channel_dist"].get(ch, 0) + 1
        stats["bit_depth_dist"][bd] = stats["bit_depth_dist"].get(bd, 0) + 1
        stats["durations_sec"].append(dur)

        if dur < 1.0:
            stats["files_shorter_than_1s"] += 1
        elif dur > 1.0:
            stats["files_longer_than_1s"] += 1
        else:
            stats["files_exact_1s"] += 1

        # Check for byte-identical duplicate
        is_dup = sha256_hash in known_hashes
        if is_dup:
            canonical_path = known_hashes[sha256_hash]
            stats["duplicate_files_count"] += 1
            stats["duplicate_records"].append({
                "original_filename": orig_filename,
                "sha256": sha256_hash,
                "canonical_file": canonical_path,
            })
            manifest_entries.append({
                "local_path": canonical_path,
                "original_filename": orig_filename,
                "source_path": src_path,
                "drive_file_id": drive_file_id,
                "sha256": sha256_hash,
                "file_size_bytes": file_size,
                "import_timestamp": timestamp,
                "dataset_label": "swara",
                "speaker_id": "unknown",
                "is_valid_wav": True,
                "format": hdr,
                "is_duplicate": True,
                "canonical_file": canonical_path,
            })
            continue

        # Determine target filename avoiding collisions
        target_name = orig_filename
        base_stem, ext = os.path.splitext(orig_filename)
        if target_name.lower() in allocated_filenames:
            # Deterministic collision naming
            target_name = f"{base_stem}_{sha256_hash[:8]}{ext}"

        dest_path = os.path.join(dest_dir, target_name)
        # Copy file cleanly (read-only copy)
        shutil.copy2(src_path, dest_path)

        allocated_filenames.add(target_name.lower())
        known_hashes[sha256_hash] = dest_path
        stats["successfully_imported"] += 1

        manifest_entries.append({
            "local_path": safe_relpath(dest_path),
            "original_filename": orig_filename,
            "source_path": src_path,
            "drive_file_id": drive_file_id,
            "sha256": sha256_hash,
            "file_size_bytes": file_size,
            "import_timestamp": timestamp,
            "dataset_label": "swara",
            "speaker_id": "unknown",  # PRESERVED AS UNKNOWN
            "is_valid_wav": True,
            "format": hdr,
            "is_duplicate": False,
            "canonical_file": safe_relpath(dest_path),
        })

    # 4. Verify source files were untouched
    for p, orig_mtime in source_mtimes_before.items():
        if os.path.getmtime(p) != orig_mtime:
            stats["source_read_only_verified"] = False
            raise RuntimeError(f"CRITICAL: Source file was modified: {p}")

    # 5. Save Manifests
    manifest_path = os.path.join(dest_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    root_manifest_path = os.path.join(os.path.dirname(dest_dir), "swara_manifest.json")
    with open(root_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    # 6. Final report assembly
    durs = stats["durations_sec"]
    report = {
        "source": actual_source_dir,
        "destination": os.path.abspath(dest_dir),
        "manifest_path": os.path.abspath(manifest_path),
        "total_files_found": stats["total_source_files"],
        "successfully_imported": stats["successfully_imported"],
        "duplicate_files": stats["duplicate_files_count"],
        "corrupt_files": stats["corrupt_files_count"],
        "skipped_files": stats["skipped_files_count"],
        "speaker_identity": "unknown (strictly preserved)",
        "sample_rate_distribution": stats["sample_rate_dist"],
        "channel_distribution": stats["channel_dist"],
        "bit_depth_distribution": stats["bit_depth_dist"],
        "duration_stats": {
            "min_sec": min(durs) if durs else 0.0,
            "max_sec": max(durs) if durs else 0.0,
            "mean_sec": float(sum(durs) / len(durs)) if durs else 0.0,
            "files_shorter_than_1s": stats["files_shorter_than_1s"],
            "files_longer_than_1s": stats["files_longer_than_1s"],
            "files_exact_1s": stats["files_exact_1s"],
        },
        "duplicate_details": stats["duplicate_records"],
        "source_preservation_verified": stats["source_read_only_verified"],
    }

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Swara Google Drive Dataset Import Tool")
    parser.add_argument("--source", type=str, default=None, help="Local directory path OR Google Drive folder URL/ID (defaults to .env)")
    parser.add_argument("--dest_dir", type=str, default="data/raw/swara", help="Destination folder (default: data/raw/swara)")
    parser.add_argument("--drive_file_id", type=str, default=None, help="Optional Google Drive folder or file ID for manifest metadata")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file (default: .env)")
    args = parser.parse_args()

    source = args.source
    drive_id = args.drive_file_id

    # If source is not provided via CLI, resolve from .env or system environment
    if not source:
        env_vars = load_env_file(args.env_file)
        for key in ["GOOGLE_DRIVE_LINK", "GDRIVE_DATASET_URL", "SWARA_GDRIVE_URL", "GDRIVE_URL", "GOOGLE_DRIVE_FOLDER_ID", "GDRIVE_FOLDER_ID"]:
            val = env_vars.get(key) or os.environ.get(key)
            if val:
                source = val
                if not drive_id:
                    drive_id = env_vars.get("GDRIVE_FOLDER_ID") or os.environ.get("GDRIVE_FOLDER_ID")
                print(f"[CONFIG] Loaded source from {args.env_file} (key: {key})")
                break

    if not source:
        print("ERROR: No dataset source specified!")
        print(f"Please specify --source on CLI or configure GOOGLE_DRIVE_LINK in {args.env_file}")
        sys.exit(1)

    print("==================================================================")
    print("           Swara Google Drive Dataset Import Workflow             ")
    print("==================================================================")
    print(f"Source:      {source}")
    print(f"Destination: {args.dest_dir}\n")

    report = import_dataset(
        source=source,
        dest_dir=args.dest_dir,
        drive_file_id=drive_id,
    )

    print("\n--- Import Summary ---")
    print(f"1. Total Source Files:        {report['total_files_found']}")
    print(f"2. Successfully Imported:     {report['successfully_imported']}")
    print(f"3. Duplicate Copies Detected: {report['duplicate_files']}")
    print(f"4. Corrupt / Invalid Files:   {report['corrupt_files']}")
    print(f"5. Skipped Files:             {report['skipped_files']}")
    print(f"6. Speaker Identity:          {report['speaker_identity']}")
    print(f"7. Source Unmodified:         {report['source_preservation_verified']}")
    print(f"8. Manifest Written:          {report['manifest_path']}")

    print("\n--- Audio Format Breakdown ---")
    print(f"Sample Rates: {report['sample_rate_distribution']}")
    print(f"Channels:     {report['channel_distribution']}")
    print(f"Bit Depths:   {report['bit_depth_distribution']}")
    print(f"Durations:    Min: {report['duration_stats']['min_sec']:.3f}s | Max: {report['duration_stats']['max_sec']:.3f}s | Mean: {report['duration_stats']['mean_sec']:.3f}s")
    print(f"Files < 1.0s: {report['duration_stats']['files_shorter_than_1s']}")
    print(f"Files == 1.0s:{report['duration_stats']['files_exact_1s']}")
    print(f"Files > 1.0s: {report['duration_stats']['files_longer_than_1s']}")
    print("==================================================================")


if __name__ == "__main__":
    main()
