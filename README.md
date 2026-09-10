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
│   │   ├── vad.c           # [Upcoming] Lightweight energy/ZCR voice activity gate
│   │   └── vad.h           # [Upcoming] VAD header & state definitions
│   │
│   ├── features/
│   │   ├── fft.c           # 512-point Radix-2 Cooley-Tukey FFT & Hanning window
│   │   ├── fft.h           # FFT API, twiddle factors & power spectrum
│   │   ├── mfcc.c          # 20 Mel filterbanks & 10 DCT-II cepstral coefficients
│   │   └── mfcc.h          # MFCC extraction API & sparse filter structures
│   │
│   ├── model/
│   │   ├── classifier.c    # [Upcoming] INT8 TFLM model execution & thresholding
│   │   └── classifier.h    # [Upcoming] Classifier interface
│   │
│   └── main.c              # [Upcoming] Embedded application loop & wake event handler
│
├── tests/              # Native C verification & unit test suites
│   ├── test_audio_buffer.c # Ring buffer capacity, extraction & wrap-around tests
│   ├── test_fft.c          # FFT twiddle, bit-reversal & sine tone frequency tests
│   └── test_mfcc.c         # Mel filterbank, log-compression & DCT tests
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
│   └── quantize.py
│
├── data/               # Offline training & validation datasets
│   ├── raw/
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
| **Mel Filters** | `20` | Triangular Mel filterbank count |
| **MFCC Coefficients** | `10` | First 10 cepstral coefficients |
| **Feature Tensor Shape** | `(49, 10, 1)` | 49 time frames × 10 MFCCs × 1 channel |

---

## 🚀 Getting Started

### 1. Build & Run Native C Tests (CMake)

Swara's runtime engine and audio feature extraction pipeline are built in native C99 with zero dynamic memory allocation.

```bash
# Configure and build with CMake
cmake -B build
cmake --build build

# Run unit test suite
ctest --test-dir build --output-on-failure
```

Included unit tests:
- `test_audio_buffer`: Validates circular buffer wrap-around, frame extraction (480 samples), and window retrieval (16,000 samples).
- `test_fft`: Validates 512-point Radix-2 FFT, Hanning windowing, and 1000 Hz pure sine wave frequency bin detection.
- `test_mfcc`: Validates 20-channel Mel filterbank creation, log-compression, and 10-coefficient DCT-II calculation.

---

### 2. Optional Offline Python Tooling (Training & Quantization)

Install dependencies if training or quantizing new models:
```bash
pip install tensorflow numpy librosa scipy soundfile
```

### 2. Prepare Data

1. Place your raw audio samples in `data/raw/`.
2. Run audio preprocessing and feature extraction using `training/dataset.py`:
   ```bash
   python training/dataset.py
   ```

### 3. Model Training

Train the lightweight keyword spotting network:
```bash
python training/train.py --epochs 50 --batch_size 32 --lr 0.001
```

### 4. Evaluation & Benchmarking

Evaluate performance on test datasets:
```bash
python training/evaluate.py --model_path ../models/swara_float32.tflite
```

### 5. Quantization & Embedded Export

Convert the trained model to Float32 & INT8 TFLite, and generate C++ deployment arrays:
```bash
python training/quantize.py
```

Generated C arrays are saved to:
- `deployment/model_data.h`
- `deployment/model_data.cc`

---

## 🏛️ Architecture & Governance

Detailed technical specifications, component data flows, memory budgets, and change management processes are maintained in [architecture.md](architecture.md).

For regular changes and updates to the architecture, use the `swara-architecture` skill.

