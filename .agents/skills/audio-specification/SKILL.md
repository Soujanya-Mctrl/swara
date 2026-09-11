---
name: audio-specification
description: Frozen Audio Specification (V0) for Swara edge voice recognition. Use this skill whenever configuring audio recording, preprocessing, feature extraction (MFCC/Mel), DSP framing, neural network input tensors, or embedded microcontroller inference.
---

# Swara Frozen Audio Specification (V0) Skill

This skill documents and enforces the frozen audio front-end and feature extraction specifications for the **Swara** project.

---

## 1. Frozen Audio Parameters Table (V0)

| Parameter | Value | Details & Hardware Implementation |
| :--- | :--- | :--- |
| **Sample Rate** | `16,000 Hz` | Standard acoustic bandwidth (Nyquist frequency: 8 kHz) |
| **Channels** | `1` | Single-channel Mono |
| **Sample Format** | `signed 16-bit PCM` | Standard linear PCM (`int16_t`), values from $-32768$ to $+32767$ |
| **Window Duration** | `1,000 ms` | 1 second analysis window |
| **Samples per Window** | `16,000` | $16,000 \text{ samples/sec} \times 1.0 \text{ sec}$ |
| **Frame Length** | `30 ms` | 480 samples ($16 \times 30$) |
| **Frame Step (Hop)** | `20 ms` | 320 samples ($16 \times 20$); 10 ms (160 samples) frame overlap |
| **FFT Size** | `512` | Radix-2 FFT covering 480 samples (padded with 32 zeros) |
| **Mel Filters** | `20` | Triangular Mel-spaced filterbanks |
| **MFCC Coefficients** | `10` | First 10 discrete cosine transform (DCT-II) coefficients |
| **Feature Tensor Shape** | `(49, 10, 1)` | 49 time frames $\times$ 10 MFCC coefficients $\times$ 1 channel |

---

## 2. Derivation of Tensor Dimensions

- **Number of Frames:**
  $$\text{Frames} = 1 + \left\lfloor \frac{\text{Total Samples} - \text{Frame Length}}{\text{Frame Step}} \right\rfloor = 1 + \left\lfloor \frac{16000 - 480}{320} \right\rfloor = 1 + 48 = 49$$
- **Coefficients per Frame:** $10$
- **Input Shape to Neural Network:** `(49, 10, 1)` or `(batch_size, 49, 10, 1)`

---

## 3. Audio Pipeline & Preprocessing Steps

The runtime audio processing follows the hard constraint sequence:
$$\text{PCM} \longrightarrow \text{VAD} \longrightarrow \text{MFCC} \longrightarrow \text{DS-CNN} \longrightarrow \text{Wake Detection}$$

1. **Audio Ingestion:**
   - 16 kHz, 1-channel, signed 16-bit linear PCM (`int16_t`) stream.
   - Sliced into 30 ms frames (480 samples) advanced by 20 ms steps (320 samples).
2. **VAD Gating (Idle CPU $\approx 10\%$):**
   - Lightweight frame-energy and zero-crossing rate (ZCR) evaluation executed every 20 ms.
   - If energy is below dynamic noise floor, frame is marked as silence/noise:
     - **MFCC extraction and NN inference are bypassed.**
     - Processor enters low-power sleep / WFI, maintaining $\approx 10\%$ CPU idle utilization.
   - If speech energy is detected (with hangover smoothing), proceed to MFCC feature extraction.
3. **Pre-emphasis:**
   - Apply first-order high-pass pre-emphasis filter: $y[n] = x[n] - 0.97 \cdot x[n-1]$.
4. **Hamming Windowing:**
   - Apply a Hamming window of length 480 to the 30 ms active frame: $w[n] = 0.54 - 0.46 \cos(2\pi n / 479)$.
5. **Zero-Padding & FFT:**
   - Pad the 480-sample frame with 32 zeros to reach $N = 512$.
   - Execute 512-point Real FFT yielding 257 unique magnitude spectrum bins ($k = 0 \dots 256$).
5. **Mel Filterbank:**
   - Multiply power spectrum by 20 triangular Mel filters spanning 20 Hz to 8,000 Hz.
   - Accumulate energy in each filter band and compute $\log(\text{energy} + \epsilon)$.
6. **Discrete Cosine Transform (DCT-II):**
   - Apply DCT-II across the 20 log-energies to compress into 10 orthogonal cepstral coefficients.
7. **Feature Matrix Buffer:**
   - Collect 49 frames of 10 coefficients into an array of dimension `[49, 10, 1]` fed to INT8 DS-CNN.

---

## 4. Enforcement & Consistency Checklist

When writing or modifying code in Swara:

- [ ] `training/dataset.py`: Ensure `sample_rate=16000`, `clip_duration_ms=1000`, `n_mfcc=10`, `n_mels=20`, `n_fft=512`, `win_length=480`, `hop_length=320`, pre-emphasis=0.97.
- [ ] `training/model.py`: Ensure model `input_shape` default is `(49, 10, 1)`, classes=3 (`0: silence`, `1: unknown`, `2: swara`), filters=64.
- [ ] `training/quantize.py`: Ensure calibration generator inputs match shape `(1, 49, 10, 1)` and use REAL data, with full INT8 input and output.
- [ ] Model contract: Input `[1, 49, 10, 1]` INT8, output `[1, 3]` INT8.
- [ ] Microcontroller DSP code / C++ drivers: Verify I2S or PDM microphone is configured for 16 kHz, 16-bit Mono, and ring buffer feeds frames of 480 samples every 320 samples.

