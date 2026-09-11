# Swara: Edge Voice & Wake Detection Runtime

**Swara** is an ultra-low-power, deterministic edge voice recognition and wake detection system designed for TinyML microcontrollers and embedded processors (ARM Cortex-M, ESP32, RISC-V).

---

## ⚡ Hard Engineering Constraints

The architecture is strictly designed from day one around embedded physical limits:

* **Implementation:** **C/C++ only** for the actual project and runtime. No runtime interpreter.
* **Target Memory Footprint:** **$\le 256\text{ KB}$** total (Flash code + INT8 weights + Tensor Arena + PCM ring buffer + MFCC scratch + stack).
* **CPU Target:** **$\approx 10\%$ CPU usage while idle** via lightweight VAD gating.
* **Deployment Target:** Edge / TinyML (bare-metal, FreeRTOS, Zephyr).
* **Inference Engine:** INT8 Depthwise Separable CNN (DS-CNN).
* **Audio Pipeline:** `PCM → VAD → MFCC → DS-CNN → wake detection`.
* **Zero Dynamic Allocation:** `malloc`/`calloc`/`realloc`/`free` strictly forbidden in runtime audio paths.
* **Role of Python:** **Not part of the deployed system.** Python is strictly optional external tooling for offline dataset curation and training.

---

## 📁 Project Architecture

```text
swara/
│
├── src/                # PRIMARY RUNTIME: Native C embedded implementation
│   ├── audio/
│   │   ├── audio_buffer.c  # Circular ring buffer for 16-bit PCM sliding window
│   │   ├── audio_buffer.h  # Ring buffer API & frozen V0 constants
│   │   ├── vad.c           # Lightweight energy/ZCR voice activity gate
│   │   ├── vad.h           # VAD header, thresholds & hangover state
│   │   ├── wav_reader.c    # Minimal zero-allocation 16kHz 16-bit mono WAV reader
│   │   └── wav_reader.h    # WAV header parser & validation interface
│   │
│   ├── features/
│   │   ├── fft.c           # 512-point Radix-2 Cooley-Tukey FFT & Hamming window
│   │   ├── fft.h           # FFT API, twiddle factors & power spectrum
│   │   ├── mfcc.c          # 20 Mel filterbanks & 10 DCT-II cepstral coefficients
│   │   └── mfcc.h          # MFCC extraction API & full 49x10 window extractor
│   │
│   ├── model/
│   │   ├── classifier.c    # [Upcoming] INT8 TFLM model execution & thresholding
│   │   └── classifier.h    # [Upcoming] Classifier interface
│   │
│   └── main.c              # [Upcoming] Embedded application loop & wake event handler
│
├── tests/              # Native C verification & unit test suites
│   ├── data/
│   │   └── test_16k_1s.wav # Deterministic 1-second 16kHz mono 16-bit test WAV fixture
│   ├── test_audio_buffer.c # Ring buffer capacity, extraction & wrap-around tests
│   ├── test_vad.c          # VAD energy threshold, silence rejection & speech trigger
│   ├── test_wav.c          # WAV header validation, format rejection & sample reading
│   ├── test_fft.c          # FFT twiddle, bit-reversal & sine tone frequency tests
│   ├── test_mfcc.c         # Mel filterbank, log-compression & DCT tests
│   ├── test_pipeline.c     # End-to-end Silence, 1kHz tone, and formant speech tests
│   ├── test_wav_mfcc.c     # Milestone M1b: Full WAV -> 49x10 MFCC pipeline validation
│   └── benchmark_mfcc.c    # Microsecond-accurate latency, throughput & memory profiler
│
├── models/             # Frozen model binaries & deployment C arrays
│   ├── swara_float32.tflite
│   └── swaral_int8.tflite
│
├── deployment/         # TFLite Micro model byte arrays
│   ├── model_data.cc
│   └── model_data.h
│
├── training/           # OPTIONAL external tooling (Python offline development)
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── quantize.py
│   ├── config.py
│   ├── validate_tflite.py
│   ├── export_model_header.py
│   ├── import_drive_dataset.py
│   └── prepare_dataset_splits.py
│
├── tools/              # Visual inspection & CLI exploration tools
│   ├── dashboard.py    # Rich interactive terminal dashboard
│   └── wav_visualizer/ # Windows desktop WAV & MFCC inspection GUI
│
├── data/               # Offline training & validation datasets
│   ├── raw/            # Master immutable archive (raw/swara/ + manifest.json)
│   ├── train/          # Physical training partition (swara: 7 files)
│   ├── val/            # Physical validation partition (swara: 1 file)
│   ├── test/           # Physical test partition (swara: 3 files)
│   ├── processed/
│   └── augmented/
│
├── CMakeLists.txt      # Root build configuration for runtime & test suites
├── architecture.md     # Architecture specifications, hard constraints & change log
└── README.md           # Project documentation and developer guide
```

---

## 🎙️ Frozen Audio Specification (V0)

The audio front-end and feature extraction parameters are frozen for V0:

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Sample Rate** | `16,000 Hz` | 16 kHz acoustic bandwidth |
| **Channels** | `1` | Mono channel |
| **Sample Format** | `signed 16-bit PCM` | Standard signed integer audio (-32768 to 32767) |
| **Window** | `1,000 ms` | 1.0 second duration |
| **Samples / Window** | `16,000` | 16,000 samples @ 16 kHz |
| **Frame Length** | `30 ms` | 480 samples per FFT analysis frame |
| **Frame Step** | `20 ms` | 320 samples hop size (10 ms overlap) |
| **Pre-emphasis** | `0.97` | High-frequency acoustic boost: $y[n] = x[n] - 0.97 \cdot x[n-1]$ |
| **Window Function** | **Hamming** | $w[n] = 0.54 - 0.46 \cos(2\pi n / 479)$ |
| **FFT Size** | `512` | Next power of 2 for 480-sample frame |
| **Mel Filters** | `20` | Triangular Mel filterbank count (20 Hz - 8,000 Hz) |
| **MFCC Coefficients** | `10` | First 10 DCT-II cepstral coefficients |
| **Feature Tensor Shape** | `(49, 10, 1)` | 49 time frames × 10 MFCCs × 1 channel (490 floats) |

---

## 📊 Front-End Performance & Memory Verification (M1b)

Measured via [`benchmark_pipeline`](tests/benchmark_pipeline.c) on host machine:

| Metric | Measured Benchmark | Notes |
| :--- | :--- | :--- |
| **VAD Gating Check (20 ms frame)** | **`0.15 µs`** (0.00015 ms) | **6.62M frames / sec** |
| **VAD Idle CPU Duty Cycle** | **`0.0008%`** | Far below the hard $\approx 10\%$ CPU idle budget |
| **Buffer Ingestion (320-sample hop)** | **`2.54 µs`** | Real-time sliding window write |
| **Single Frame MFCC (30 ms / 480 spl)** | **`10.03 µs`** (0.010 ms) | **2,991× faster than real-time** |
| **1-Second Active Window (49 frames)** | **`0.458 ms`** | **2,183× faster than real-time (RTF 1:2183)** |
| **Active Speech CPU Duty Cycle** | **`0.046%`** | Ultra-low power profile |
| **Streaming Pipeline (50% speech / 50% silence)** | **`0.613 ms` / 1s audio** | 4,900 silence frames bypassed via VAD |
| **Total Front-End Static RAM** | **`60,444 bytes`** (~59.03 KB) | Ring buffer + VAD + Mel + FFT + 49x10 matrix |
| **Peak Stack Scratch Memory** | **`6,084 bytes`** (~5.94 KB) | 512-pt complex FFT + power spectrum |
| **Dynamic Allocations (`malloc`)** | **`0 bytes`** | Zero dynamic heap allocation |

---

## 🚀 Getting Started

### 1. Build & Run Native C Tests (CMake)

Swara's runtime engine and audio feature extraction pipeline are built in native C99 with zero dynamic memory allocation.

```bash
# Configure build with CMake (MinGW or MSVC)
cmake -B build -G "MinGW Makefiles"
cmake --build build

# Run unit test suite (7 tests)
ctest --test-dir build --output-on-failure
```

Included verification test suites:
- `test_audio_buffer`: Circular ring buffer wrap-around, frame extraction (480 samples), and window retrieval (16,000 samples).
- `test_vad`: Voice Activity Detection energy thresholds, silence rejection, and speech gating.
- `test_wav`: RIFF/WAVE header validation, 16kHz mono 16-bit format compliance, and non-conforming file rejection.
- `test_fft`: 512-point Radix-2 Cooley-Tukey FFT, Hamming windowing, and 1000 Hz pure sine wave frequency bin detection.
- `test_mfcc`: 20-channel Mel filterbank creation, log-compression, and 10-coefficient DCT-II calculation.
- `test_pipeline`: End-to-end synthetic silence, tone, and vowel formant speech audio pipelines.
- `test_wav_mfcc`: End-to-end WAV file ingestion to 49×10 MFCC matrix with canary boundary guard checks.

### 2. Run the Performance Benchmarks

Run the complete end-to-end audio pipeline benchmark:
```bash
./build/benchmark_pipeline
```

Or benchmark MFCC feature extraction in isolation:
```bash
./build/benchmark_mfcc
```


---

### 3. Offline Python Tooling (Dataset Ingestion, Training & Quantization)

Install dependencies if developing models or importing audio datasets:
```bash
pip install tensorflow numpy gdown rich matplotlib
```

#### A. Google Drive Dataset Ingestion Workflow
Swara provides a deterministic, read-only dataset importer that reads from `.env` and downloads audio without modifying the remote Google Drive folder:
1. Configure your Google Drive folder link in [`.env`](.env):
   ```ini
   GOOGLE_DRIVE_LINK="https://drive.google.com/drive/folders/<your_folder_id>?usp=sharing"
   ```
2. Run the automated import pipeline:
   ```bash
   py training/import_drive_dataset.py
   ```
   * Downloads and validates audio headers using standard library `wave`.
   * Enforces 16,000 Hz, mono, signed 16-bit PCM.
   * Performs SHA-256 deduplication and generates [`data/raw/swara/manifest.json`](data/raw/swara/manifest.json).
   * **Result:** Real recordings are saved in `data/raw/swara/` (currently 11 valid recordings, 33.5s total duration).

#### B. Physical Dataset Partitioning (`train`, `val`, `test`)
To partition master raw recordings into physical directory splits:
```bash
py training/prepare_dataset_splits.py
```
This materializes the following layout:
```text
data/
├── train/
│   ├── swara/     (7 recordings - 70%)
│   ├── silence/   (directory ready for negative samples)
│   └── unknown/   (directory ready for negative samples)
├── val/
│   ├── swara/     (1 recording - 15%)
│   ├── silence/   (ready)
│   └── unknown/   (ready)
├── test/
│   ├── swara/     (3 recordings - 15%)
│   ├── silence/   (ready)
│   └── unknown/   (ready)
└── raw/
    └── swara/     (11 master recordings + manifest.json)
```

#### C. End-to-End Model Creation Workflow
1. **Dataset Integrity Audit**:
   Inspect all WAV files and verify audio specification compliance:
   ```bash
   py training/inspect_dataset.py --data_dir data
   ```
2. **Model Training Pipeline**:
   Train the DS-CNN keyword spotting network once negative audio samples (`silence`, `unknown`) are placed in `data/train` and `data/val`:
   ```bash
   py training/train.py --data_dir data --epochs 30 --batch_size 16 --lr 0.001 --save_path models/swara_saved_model
   ```
3. **Recording-Level Evaluation**:
   Compute accuracy, confusion matrix, precision, and recall on the test set:
   ```bash
   py training/evaluate.py --model_path models/swara_saved_model --data_dir data --split test
   ```
4. **Full Integer INT8 Quantization**:
   Quantize the trained model using real speech features for calibration:
   ```bash
   py training/quantize.py --model_path models/swara_saved_model --data_dir data --output_int8 models/swara_int8.tflite
   ```
5. **TFLite Model Inspection & TFLM Operator Validation**:
   Audit the generated flatbuffer to ensure 100% of operators are supported by TFLM:
   ```bash
   py training/validate_tflite.py --model_path models/swara_int8.tflite
   ```
6. **Deterministic C Array Export**:
   Export the model into 16-byte aligned C deployment arrays:
   ```bash
   py training/export_model_header.py --tflite_path models/swara_int8.tflite --output_cc deployment/model_data.cc --output_h deployment/model_data.h
   ```

---

### 4. Interactive CLI Dashboard
Swara includes a rich terminal-based interactive dashboard to explore the entire model creation pipeline, compare DS-CNN architecture candidates against memory budgets, run C benchmarks, verify numerical parity, and audit datasets:
```bash
py tools/dashboard.py
```
*(Displays ASCII banner, system status box, architecture comparison table, and direct test execution).*

---

### 5. WAV & MFCC Visualizer & Debugger (Desktop GUI)
A dedicated desktop application (Tkinter + Matplotlib) to inspect audio waveforms, frame boundaries, FFT spectra, Mel filterbanks, and $49 \times 10$ MFCC heatmaps:
```bash
py -m tools.wav_visualizer.app.main
```
*(Or launch directly from the CLI Dashboard via Option `8`).*

---

## 🔍 Verification Status & Confidence Boundaries

To avoid unverified claims, Swara maintains a strict boundary between what is tested/verified and what is pending full training data:

### ✅ CURRENTLY VERIFIED
- **Real Dataset Ingestion:** 11 real "hello swara" recordings successfully imported from Google Drive, SHA-256 deduplicated, and cataloged in `data/raw/swara/manifest.json`.
- **Physical Dataset Partitioning:** Materialized into `data/train/`, `data/val/`, `data/test/` via deterministic content hashing.
- **C Audio Front-End:** Circular ring buffer (`audio_buffer.c`), Radix-2 512-point FFT (`fft.c`), and 20 Mel / 10 MFCC extractor (`mfcc.c`) with zero dynamic memory allocation.
- **DSP Parity:** Numerical equivalence between C and Python feature extraction (Max absolute error $0.0172 < 0.05$).
- **WAV Ingestion Robustness:** Clean rejection of corrupt headers, non-PCM, stereo, and wrong sample rates.
- **TFLM Operator Compatibility:** Verified that DS-CNN compiles to 8 standard operators (`CONV_2D`, `DEPTHWISE_CONV_2D`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `CONV_2D`, `MEAN`, `FULLY_CONNECTED`, `SOFTMAX`), 100% supported by `AllOpsResolver`.
- **Automated Tests:** 100% pass across all 7 CTest suites and 10 infrastructure unit tests.

### ⏳ PENDING (Awaiting Silence & Unknown Audio Samples)
- **Model Training & Accuracy:** Training is held until negative class recordings (`silence`, `unknown`) are placed in `data/train` and `data/val` to prevent training on dummy/synthetic audio.
- **Speaker Generalization:** Speaker identities are not labeled; evaluation is strictly recording-level.
- **Final INT8 Calibration:** Calibration scales will be calculated from the full 3-class dataset during post-training quantization.
- **Measured Hardware Arena:** TFLM tensor arena (~36.1 KB) is an **ESTIMATE**; physical runtime measurement via `interpreter.arena_used_bytes()` will be recorded on target microcontroller boards.

---

## 🎯 Model Input / Output Contract

| Property | Value / Specification | Notes |
| :--- | :--- | :--- |
| **Input Shape** | `[1, 49, 10, 1]` | 49 time frames × 10 MFCC coefficients × 1 channel |
| **Input Dtype** | `int8` (post-quantization) | Full integer quantized (-128 to 127) |
| **Input Scale & Zero Point** | *Calculated at PTQ* | Determined from real representative audio features |
| **Output Shape** | `[1, 3]` | 3 target classes |
| **Output Dtype** | `int8` (post-quantization) | Quantized Softmax probabilities |
| **Class Index 0** | `silence` | Background noise / silence |
| **Class Index 1** | `unknown` | Non-wake speech / background utterances |
| **Class Index 2** | `swara` | Target wake phrase |

---

## 💾 System Memory Footprint & 256 KB Budget Breakdown

The architecture is constrained to a total system budget of **$\le 256\text{ KB}$ Total RAM**.
Memory accounting is categorized into **MEASURED** (front-end C DSP) versus **ESTIMATED** (neural network activations & RTOS):

| Subsystem Component | Memory Type | Allocation | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **1-sec PCM Audio Buffer** | Static RAM | `32,000 bytes` (31.25 KB) | **MEASURED** | `int16_t[16000]` circular sliding buffer |
| **MFCC Configuration & Tables** | Static RAM | `26,436 bytes` (25.82 KB) | **MEASURED** | 20 Mel filterbanks, FFT tables, DCT matrix |
| **49×10 Feature Buffer** | Static RAM | `1,960 bytes` (1.91 KB) | **MEASURED** | 49 frames × 10 MFCCs (`float32[490]`) |
| **VAD Engine State** | Static RAM | `16 bytes` (0.02 KB) | **MEASURED** | Energy thresholds and hangover state |
| **Front-End Call Stack Scratch**| Stack Memory| `6,084 bytes` (5.94 KB) | **MEASURED** | FFT scratch buffers & power spectrum |
| **TFLM Tensor Arena (64-ch)** | Heap / Arena| `~36,960 bytes` (~36.09 KB)| **ESTIMATED** | Activations working arena (estimated) |
| *(TFLM Arena Alt: 32-ch)* | Heap / Arena| `~20,600 bytes` (~20.12 KB)| **ESTIMATED** | Alternative low-memory candidate |
| **Firmware Stack & RTOS Overhead**| Stack/BSS  | `~15,360 bytes` (~15.00 KB)| **ESTIMATED** | FreeRTOS task stacks, interrupt stack, BSS |
| **Application State & Queues** | Static RAM | `~4,096 bytes` (~4.00 KB) | **ESTIMATED** | Classification event flags, IPC queues |
| **TOTAL SYSTEM RAM (64-ch)** | **Total RAM** | **~122,912 bytes (~120.0 KB)** | **ESTIMATED** | **$\le 256\text{ KB}$ Compliant (~136 KB Headroom)** |

*Model weights (~12.2 KB for INT8 64-channel, or ~4.8 KB for INT8 32-channel) reside in Flash ROM (`alignas(16) const unsigned char g_swara_model_data[]`) and do not consume system RAM.*

---

## 🏛️ Architecture & Governance

Detailed technical specifications, component data flows, memory budgets, and change management processes are maintained in [architecture.md](architecture.md).

For regular changes and updates to the architecture, use the `swara-architecture` skill.


