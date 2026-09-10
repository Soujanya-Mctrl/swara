---
name: swara-architecture
description: Maintain, evolve, and document the Swara edge voice/speech recognition architecture. Use this skill whenever making changes to the Swara project structure, updating training, quantization, or deployment pipelines, modifying models, or logging architecture progress in architecture.md.
---

# Swara Architecture Maintenance & Evolution Skill

This skill defines the standardized workflow, non-negotiable hard constraints, and update guidelines for the **Swara** edge voice and wake detection architecture.

---

## 1. Non-Negotiable Hard Constraints

Every change to Swara must satisfy these hard constraints:

* **Implementation:** **C/C++ only** for the actual project and runtime. No runtime interpreter.
* **Target Memory Footprint:** **$\le 256\text{ KB}$** total system footprint (Flash code + INT8 weights + Tensor Arena + PCM ring buffer + MFCC scratch + stack).
* **CPU Target:** **$\approx 10\%$ CPU usage while idle** achieved via front-end Voice Activity Detection (VAD) gating.
* **Deployment Target:** Edge / TinyML (bare-metal, FreeRTOS, Zephyr on Cortex-M / RISC-V).
* **Inference Engine:** INT8 Depthwise Separable CNN (DS-CNN).
* **Audio Pipeline:** `PCM → VAD → MFCC → DS-CNN → wake detection`.
* **Role of Python:** **Not part of the deployed system.** Python is strictly optional external tooling for offline dataset preparation, model training, and quantization export.

---

## 2. Memory Footprint Allocation ($\le 256\text{ KB}$)

| Subsystem | Budget Limit | Description |
| :--- | :--- | :--- |
| **INT8 Model Weights** | `45 - 70 KB` | Quantized DS-CNN weights linked into Flash memory |
| **TFLM Tensor Arena** | `35 - 50 KB` | Working RAM for activations and intermediate tensor buffers |
| **PCM Audio Ring Buffer** | `32 KB` | 1,000 ms sliding window of 16-bit PCM samples ($16,000 \times 2\text{ bytes}$) |
| **MFCC & FFT Scratch** | `6 - 10 KB` | 512-point FFT twiddle factors, 20 Mel filter tables, and DCT scratchpad |
| **VAD State & Buffers** | `1 - 2 KB` | Energy thresholds, zero-crossing stats, and hangover counters |
| **Firmware Code & Stack** | `60 - 80 KB` | Compiled C/C++ binary text, RTOS task stack, and static BSS |
| **TOTAL FOOTPRINT** | **$\le 244\text{ KB}$** | **Headroom remaining within $\le 256\text{ KB}$ limit** |

---

## 3. Project Directory Reference

```text
D:\Projects\swara\
├── deployment\         # Core C/C++ runtime and deployment artifacts (PRIMARY RUNTIME)
│   ├── model_data.cc   # C++ byte array definition for TFLite Micro runtime
│   └── model_data.h    # C++ header declaring external model array & length
├── models\             # Frozen model binaries
│   ├── swara_float32.tflite  # Baseline Float32 model
│   └── swaral_int8.tflite    # Fully quantized INT8 model for microcontrollers
├── training\           # OPTIONAL external tooling (Python offline development)
│   ├── train.py        # Model training loop, callbacks, saved model export
│   ├── model.py        # Neural network architecture (DS-CNN, Edge Conv)
│   ├── dataset.py      # Audio loading, feature extraction & generator pipelines
│   ├── evaluate.py     # Evaluation metrics (accuracy, F1, latency, confusion matrix)
│   └── quantize.py     # TFLite conversion (Float32 & full INT8) + C array generator
├── data\               # Offline training & validation datasets
│   ├── raw\            # Original audio recordings (.wav)
│   ├── processed\      # Extracted spectrograms, MFCC features
│   └── augmented\      # Synthetic/augmented data (noise injection, time shift, pitch)
├── architecture.md     # Primary architecture specification & change log
└── README.md           # Project overview and developer guide
```

---

## 4. Regular Change Workflow

Whenever you make architectural or structural modifications to Swara, follow this 4-step procedure:

### Step 1: Hard Constraints & Pre-Change Impact Assessment
- **Memory Verification:** Verify that any buffer, model layer, or feature table does NOT violate the **$\le 256\text{ KB}$** total budget.
- **Idle CPU Verification:** Ensure that any algorithm added to the idle path can execute within the **$\approx 10\%$ CPU** idle budget (VAD-gated).
- **Audio Spec Consistency:** Verify against the frozen audio specification (`audio-specification` skill):
  - 16 kHz sample rate, mono 16-bit PCM, 1,000 ms window (16,000 samples), 30 ms frame length (480 samples), 20 ms frame step (320 samples), 512 FFT, 20 Mel filters, 10 MFCC coefficients $\rightarrow$ shape `(49, 10, 1)`.
- **C/C++ First:** Ensure runtime algorithms are designed for pure C/C++ embedded execution without dynamic memory allocation (`malloc`/`new`).

### Step 2: Implement Code / Structure Changes
- Maintain clear separation: runtime C/C++ code belongs in `deployment/` (or embedded project source); Python code in `training/` remains external tooling.

### Step 3: Update `architecture.md`
Whenever a change occurs, update [architecture.md](../../architecture.md):
1. **Milestones / Status:** Update the progress table if a milestone has progressed or completed.
2. **Component Details:** If parameters or budgets changed, update Sections 1 and 3.
3. **Architectural Decision Records (ADRs):** Record any new decisions in Section 5 (`ADR-XXX`).
4. **Change Log:** Append an entry to Section 7 with date, version tag, author, and description.

### Step 4: Verify Alignment with README.md
- Ensure `README.md` reflects current architecture, constraints, and relative file links.

---

## 5. Edge / Embedded Constraints Checklist

Before approving any model, DSP, or firmware change, verify against this checklist:

- [ ] **Total Memory:** Total footprint (Flash + RAM + Model + Arena + Buffers) $\le 256\text{ KB}$.
- [ ] **Idle CPU Budget:** Idle duty cycle remains $\approx 10\%$; heavy processing (MFCC, NN) is gated by VAD.
- [ ] **Zero Dynamic Allocation:** No runtime heap allocation (`malloc`/`free`) in the active audio loop.
- [ ] **INT8 Quantization:** All weights and activations are quantized to signed 8-bit integers (`int8`).
- [ ] **TFLM Operations Support:** Only use supported operators (Conv2D, DepthwiseConv2D, FullyConnected, Softmax, Add, Reshape, ReLU).
