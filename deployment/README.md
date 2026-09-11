# Swara TFLM Deployment Infrastructure & Memory Budget Accounting

This directory contains the TensorFlow Lite Micro (TFLM) deployment assets and contract specifications for the Swara wake word detection engine.

---

## 1. Deployment Workflow Pipeline

```text
[Offline Training (Python)]
       ↓
training/train.py                  → Trained Keras / SavedModel
       ↓
training/quantize.py               → Full Integer INT8 TFLite (calibrated with REAL dataset)
       ↓
training/validate_tflite.py        → TFLite flatbuffer & TFLM operator audit
       ↓
training/export_model_header.py    → deployment/model_data.cc & model_data.h
       ↓
[Embedded C++ Firmware (TFLM)]
src/model/classifier.cc            → tflite::MicroInterpreter on bare-metal / RTOS
```

---

## 2. Model Input / Output Contract

| Property | Value / Specification | Notes |
| :--- | :--- | :--- |
| **Input Tensor Shape** | `[1, 49, 10, 1]` | 1 batch × 49 time frames × 10 MFCC coefficients × 1 channel |
| **Input Data Type** | `int8` | Signed 8-bit integer (-128 to 127) |
| **Input Scale** | *Determined at quantization* | Calculated post-training from real representative training audio |
| **Input Zero Point** | *Determined at quantization* | Calculated post-training from real representative training audio |
| **Output Tensor Shape** | `[1, 3]` | 1 batch × 3 classes |
| **Output Data Type** | `int8` | Signed 8-bit integer (-128 to 127) |
| **Output Scale** | *Determined at quantization* | Quantized Softmax probability distribution |
| **Output Zero Point** | *Determined at quantization* | Quantized Softmax probability distribution |
| **Class 0** | `silence` | Background noise / silence |
| **Class 1** | `unknown` | Non-target speech / other words |
| **Class 2** | `swara` | Target wake phrase ("swara") |

> [!IMPORTANT]
> Scale and zero-point values cannot and must not be invented before the real model is quantized. They will be computed by TensorFlow Lite Post-Training Quantization (PTQ) using calibration data from the real training set.

---

## 3. TFLM Supported Operator Mapping

The Swara Depthwise Separable CNN (DS-CNN) architecture compiles to **8 standard inference operations**:

1. `CONV_2D` (Initial strided 2D convolution over time/frequency)
2. `DEPTHWISE_CONV_2D` (Block 1 spatial filtering)
3. `CONV_2D` (Block 1 1×1 pointwise projection)
4. `DEPTHWISE_CONV_2D` (Block 2 spatial filtering)
5. `CONV_2D` (Block 2 1×1 pointwise projection)
6. `MEAN` (Global average pooling across time and frequency)
7. `FULLY_CONNECTED` (Dense 3-class linear classifier)
8. `SOFTMAX` (3-class normalized probability output)

All 8 operators are natively supported by the standard TensorFlow Lite Micro `AllOpsResolver` or a minimal `MicroMutableOpResolver<5>` (`Conv2D`, `DepthwiseConv2D`, `Mean`, `FullyConnected`, `Softmax`).

---

## 4. System Memory Footprint & 256 KB Budget Breakdown

The target embedded hardware budget is **$\le 256\text{ KB}$ Total RAM**.

Below is the complete memory accounting, distinguishing between **MEASURED** front-end buffers and **ESTIMATED** neural network arena requirements:

| Subsystem Component | Memory Type | Allocation | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **1-second PCM Audio Buffer** | Static RAM | `32,000 bytes` (31.25 KB) | **MEASURED** | `int16_t[16000]` circular sliding buffer |
| **MFCC Configuration & Tables** | Static RAM | `26,436 bytes` (25.82 KB) | **MEASURED** | 20 Mel filterbanks, FFT tables, DCT matrix |
| **49×10 Output Feature Buffer** | Static RAM | `1,960 bytes` (1.91 KB) | **MEASURED** | 49 frames × 10 MFCCs (`float32[490]`) |
| **VAD Control State** | Static RAM | `16 bytes` (0.02 KB) | **MEASURED** | Energy threshold and hangover state |
| **Front-End Call Stack Scratch**| Stack Memory| `6,084 bytes` (5.94 KB) | **MEASURED** | FFT real/imag, power spectrum, pre-emphasis |
| **TFLM Tensor Arena (64-channel)**| Heap/Arena | `~36,960 bytes` (~36.09 KB)| **ESTIMATED** | Activations working arena (estimated) |
| *(TFLM Arena Alternative: 32-ch)*| Heap/Arena | `~20,600 bytes` (~20.12 KB)| **ESTIMATED** | Conservative candidate if tighter RAM needed |
| **Firmware Stack & RTOS Overhead**| Stack/BSS  | `~15,360 bytes` (~15.00 KB)| **ESTIMATED** | FreeRTOS task stacks, interrupt stack, BSS |
| **Application Logic & Buffers** | Static RAM | `~4,096 bytes` (~4.00 KB) | **ESTIMATED** | Classification event flags, IPC queues |
| **TOTAL SYSTEM RAM (64-ch)** | **Total RAM** | **~122,912 bytes (~120.0 KB)** | **ESTIMATED** | **$\le 256\text{ KB}$ Budget Compliant (~136 KB Headroom)** |

> [!NOTE]
> - Model weights (estimated at ~12.2 KB for INT8 64-ch, or ~4.8 KB for INT8 32-ch) are placed in **Flash memory (ROM)** via `alignas(16) const unsigned char g_swara_model_data[]` in `model_data.cc` and do not consume system RAM.
> - Tensor Arena size is currently an **ESTIMATE**. The exact memory allocation must be measured at runtime via `interpreter.arena_used_bytes()` once the real model is quantized and instantiated with `tflite::MicroInterpreter`.
> - Microcontroller execution requires no dynamic memory allocation (`malloc`/`free`).


---

## 5. Deployment Placeholder Notice

The current files:
- `deployment/model_data.h`
- `deployment/model_data.cc`

are **infrastructure placeholders** provided for compile-time header resolution and CTest link validation. They do not contain a trained wake word model. Real weights will be exported once the Google Drive dataset is imported and trained.
