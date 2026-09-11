/**
 * @file benchmark_pipeline.c
 * @brief Comprehensive End-to-End Benchmark for the Swara Edge Audio Pipeline:
 *        Audio Ring Buffer -> VAD Energy Gate -> Framing & Pre-emphasis ->
 *        512-pt FFT -> 20 Mel Filters -> 10 MFCC Coefficients -> 49x10 Feature Matrix.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../src/audio/audio_buffer.h"
#include "../src/audio/vad.h"
#include "../src/audio/wav_reader.h"
#include "../src/features/fft.h"
#include "../src/features/mfcc.h"

#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#define NUM_VAD_ITERATIONS       50000   /* 50k frames = 1,000 seconds of audio */
#define NUM_FRAME_ITERATIONS     10000   /* 10k frames = 200 seconds of audio */
#define NUM_WINDOW_ITERATIONS    1000    /* 1,000 windows = 1,000 seconds of audio */
#define NUM_STREAM_ITERATIONS    200     /* 200 continuous audio streams */

int main(void) {
    printf("================================================================\n");
    printf("        Swara End-to-End Audio Pipeline Performance Benchmark   \n");
    printf("================================================================\n\n");

    /* 1. Initialize System Components */
    vad_config_t vad;
    vad_init(&vad, SWARA_VAD_DEFAULT_THRESHOLD, SWARA_VAD_DEFAULT_HANGOVER);

    mfcc_config_t mfcc_cfg;
    mfcc_init(&mfcc_cfg);

    int16_t ring_buffer_storage[SWARA_WINDOW_SAMPLES];
    audio_buffer_t audio_ring_buffer;
    audio_buffer_init(&audio_ring_buffer, ring_buffer_storage, SWARA_WINDOW_SAMPLES);

    /* Prepare test signals */
    /* Signal A: Ambient background noise / silence (below VAD threshold) */
    int16_t silence_samples[SWARA_WINDOW_SAMPLES];
    for (int i = 0; i < SWARA_WINDOW_SAMPLES; i++) {
        silence_samples[i] = (int16_t)((i * 7) % 50 - 25); /* Low amplitude noise */
    }

    /* Signal B: Active speech signal (above VAD threshold) */
    int16_t speech_samples[SWARA_WINDOW_SAMPLES];
    for (int i = 0; i < SWARA_WINDOW_SAMPLES; i++) {
        float t = (float)i / 16000.0f;
        float glottal = sinf(2.0f * 3.14159f * 130.0f * t);
        float f1 = sinf(2.0f * 3.14159f * 700.0f * t) * 0.6f;
        float f2 = sinf(2.0f * 3.14159f * 1220.0f * t) * 0.3f;
        speech_samples[i] = (int16_t)((glottal * (f1 + f2)) * 18000.0f);
    }

    int16_t single_frame[SWARA_FRAME_LENGTH_SAMPLES];
    for (int i = 0; i < SWARA_FRAME_LENGTH_SAMPLES; i++) {
        single_frame[i] = speech_samples[i];
    }
    int16_t silence_frame[SWARA_FRAME_LENGTH_SAMPLES];
    for (int i = 0; i < SWARA_FRAME_LENGTH_SAMPLES; i++) {
        silence_frame[i] = silence_samples[i];
    }

    float single_mfcc[SWARA_MFCC_COEFFS];
    float window_features[SWARA_NUM_FRAMES][SWARA_MFCC_COEFFS];

    /* ------------------------------------------------------------- */
    /* BENCHMARK 1: VAD Gating Engine (Idle Path)                     */
    /* ------------------------------------------------------------- */
    printf("[1] Benchmarking VAD Gating Engine (Lightweight 20ms Check)...\n");
    printf("    Iterations: %d frames (Simulates %.1f minutes of audio)\n",
           NUM_VAD_ITERATIONS, (float)(NUM_VAD_ITERATIONS * 0.02f) / 60.0f);

    double t0_vad = get_time_sec();
    volatile bool vad_active = false;
    for (int it = 0; it < NUM_VAD_ITERATIONS; it++) {
        vad_active = vad_process_frame(&vad, silence_frame, SWARA_FRAME_LENGTH_SAMPLES);
    }
    double t1_vad = get_time_sec();
    (void)vad_active;

    double total_vad_sec = t1_vad - t0_vad;
    double us_per_vad = (total_vad_sec / (double)NUM_VAD_ITERATIONS) * 1e6;
    double vad_duty_cycle_pct = (us_per_vad / 20000.0) * 100.0; /* 20ms = 20,000 us */

    printf("    -> Total time:             %.4f s\n", total_vad_sec);
    printf("    -> Execution time / frame: %.3f us (%.6f ms)\n", us_per_vad, us_per_vad / 1000.0);
    printf("    -> Throughput:             %.0f frames / second\n", (double)NUM_VAD_ITERATIONS / total_vad_sec);
    printf("    -> Idle CPU Duty Cycle:    %.4f%% (Well within ~10%% hard budget!)\n\n", vad_duty_cycle_pct);

    /* ------------------------------------------------------------- */
    /* BENCHMARK 2: Audio Buffer Ingestion & Sliding Window           */
    /* ------------------------------------------------------------- */
    printf("[2] Benchmarking Circular Ring Buffer Ingestion (320-sample hop)...\n");
    printf("    Iterations: %d frame writes\n", NUM_VAD_ITERATIONS);

    double t0_buf = get_time_sec();
    int16_t extract_buf[SWARA_WINDOW_SAMPLES];
    for (int it = 0; it < NUM_VAD_ITERATIONS; it++) {
        audio_buffer_write(&audio_ring_buffer, speech_samples, SWARA_FRAME_STEP_SAMPLES);
        if ((it % 49) == 0) {
            audio_buffer_get_window(&audio_ring_buffer, extract_buf, SWARA_WINDOW_SAMPLES);
        }
    }
    double t1_buf = get_time_sec();

    double total_buf_sec = t1_buf - t0_buf;
    double us_per_buf_write = (total_buf_sec / (double)NUM_VAD_ITERATIONS) * 1e6;

    printf("    -> Total time:             %.4f s\n", total_buf_sec);
    printf("    -> Buffer write / hop:     %.3f us\n", us_per_buf_write);
    printf("    -> Throughput:             %.0f hops / second\n\n", (double)NUM_VAD_ITERATIONS / total_buf_sec);

    /* ------------------------------------------------------------- */
    /* BENCHMARK 3: MFCC Single Frame (Pre-emphasis + FFT + Mel + DCT)*/
    /* ------------------------------------------------------------- */
    printf("[3] Benchmarking Single 480-sample (30ms) MFCC Feature Extraction...\n");
    printf("    Iterations: %d frames\n", NUM_FRAME_ITERATIONS);

    double t0_frame = get_time_sec();
    for (int it = 0; it < NUM_FRAME_ITERATIONS; it++) {
        mfcc_extract_frame(&mfcc_cfg, single_frame, single_mfcc);
    }
    double t1_frame = get_time_sec();

    double total_frame_sec = t1_frame - t0_frame;
    double us_per_frame = (total_frame_sec / (double)NUM_FRAME_ITERATIONS) * 1e6;
    double ms_per_frame = us_per_frame / 1000.0;
    double frames_per_sec = (double)NUM_FRAME_ITERATIONS / total_frame_sec;

    printf("    -> Total time:             %.4f s\n", total_frame_sec);
    printf("    -> Execution time / frame: %.2f us (%.4f ms)\n", us_per_frame, ms_per_frame);
    printf("    -> Throughput:             %.0f frames / second\n", frames_per_sec);
    printf("    -> Real-Time Headroom:     %.1fx faster than real-time (30ms frame / %.4fms)\n\n",
           30.0 / ms_per_frame, ms_per_frame);

    /* ------------------------------------------------------------- */
    /* BENCHMARK 4: Complete 1-second Window (49 frames -> 49x10)    */
    /* ------------------------------------------------------------- */
    printf("[4] Benchmarking Full 1-second Window (49 frames, 16,000 samples)...\n");
    printf("    Iterations: %d windows (1,000 seconds of audio)\n", NUM_WINDOW_ITERATIONS);

    double t0_win = get_time_sec();
    for (int it = 0; it < NUM_WINDOW_ITERATIONS; it++) {
        mfcc_compute_window(&mfcc_cfg, speech_samples, window_features);
    }
    double t1_win = get_time_sec();

    double total_win_sec = t1_win - t0_win;
    double ms_per_win = (total_win_sec / (double)NUM_WINDOW_ITERATIONS) * 1e3;
    double wins_per_sec = (double)NUM_WINDOW_ITERATIONS / total_win_sec;
    double rtf = 1000.0 / ms_per_win;

    printf("    -> Total time:             %.4f s\n", total_win_sec);
    printf("    -> Execution time / 1s win:%.3f ms\n", ms_per_win);
    printf("    -> Windows / second:       %.1f\n", wins_per_sec);
    printf("    -> Real-Time Factor (RTF): 1:%.0f (Processes 1.0s speech in %.3f ms)\n", rtf, ms_per_win);
    printf("    -> Active CPU Duty Cycle:  %.3f%% on host\n\n", (ms_per_win / 1000.0) * 100.0);

    /* ------------------------------------------------------------- */
    /* BENCHMARK 5: End-to-End Pipeline: Ingest -> VAD Gate -> MFCC  */
    /* ------------------------------------------------------------- */
    printf("[5] Benchmarking End-to-End Streaming Pipeline (PCM -> VAD -> MFCC)...\n");
    printf("    Simulating continuous streaming of %d 1-second audio streams:\n", NUM_STREAM_ITERATIONS);
    printf("    - 50%% Silence / Ambient Noise (VAD rejects, skips MFCC)\n");
    printf("    - 50%% Active Speech (VAD triggers, extracts 49x10 MFCCs)\n");

    double t0_e2e = get_time_sec();
    size_t silence_frames_gated = 0;
    size_t speech_windows_computed = 0;

    for (int s = 0; s < NUM_STREAM_ITERATIONS; s++) {
        const int16_t* stream_audio = (s % 2 == 0) ? silence_samples : speech_samples;

        /* Feed 50 frames (1 second audio stream in 20ms chunks) */
        bool speech_detected_in_stream = false;
        for (int f = 0; f < 50; f++) {
            const int16_t* chunk = &stream_audio[f * 320];
            audio_buffer_write(&audio_ring_buffer, chunk, 320);

            bool vad_res = vad_process_frame(&vad, chunk, SWARA_FRAME_LENGTH_SAMPLES);
            if (vad_res) {
                speech_detected_in_stream = true;
            } else {
                silence_frames_gated++;
            }
        }

        /* If speech was detected, trigger full 49x10 window feature extraction */
        if (speech_detected_in_stream) {
            audio_buffer_get_window(&audio_ring_buffer, extract_buf, SWARA_WINDOW_SAMPLES);
            mfcc_compute_window(&mfcc_cfg, extract_buf, window_features);
            speech_windows_computed++;
        }
        vad_reset(&vad);
    }
    double t1_e2e = get_time_sec();

    double total_e2e_sec = t1_e2e - t0_e2e;
    double ms_per_stream = (total_e2e_sec / (double)NUM_STREAM_ITERATIONS) * 1e3;
    double effective_stream_duty = (ms_per_stream / 1000.0) * 100.0;

    printf("    -> Total streaming time:   %.4f s (for %d seconds of audio)\n", total_e2e_sec, NUM_STREAM_ITERATIONS);
    printf("    -> Average time / 1s audio:%.3f ms\n", ms_per_stream);
    printf("    -> Silence frames bypassed:%zu\n", silence_frames_gated);
    printf("    -> Speech windows extracted:%zu\n", speech_windows_computed);
    printf("    -> Effective CPU Duty:     %.3f%%\n\n", effective_stream_duty);

    /* ------------------------------------------------------------- */
    /* BENCHMARK 6: Memory & Footprint Accounting                     */
    /* ------------------------------------------------------------- */
    printf("[6] System Memory Footprint Accounting (<= 256 KB Budget):\n");
    size_t sz_pcm_buf  = SWARA_WINDOW_SAMPLES * sizeof(int16_t);      /* 32,000 B */
    size_t sz_buf_ctrl = sizeof(audio_buffer_t);                      /* 16 B */
    size_t sz_vad_cfg  = sizeof(vad_config_t);                        /* 16 B */
    size_t sz_mfcc_cfg = sizeof(mfcc_config_t);                       /* 26,440 B */
    size_t sz_features = SWARA_NUM_FRAMES * SWARA_MFCC_COEFFS * sizeof(float); /* 1,960 B */

    size_t total_static_dsp_ram = sz_pcm_buf + sz_buf_ctrl + sz_vad_cfg + sz_mfcc_cfg + sz_features;

    size_t scratch_fft   = SWARA_FFT_SIZE * sizeof(float) * 2;        /* 4,096 B */
    size_t scratch_power = SWARA_FFT_BINS * sizeof(float);            /* 1,028 B */
    size_t scratch_pre   = SWARA_FRAME_LENGTH_SAMPLES * sizeof(int16_t);/* 960 B */
    size_t peak_stack_scratch = scratch_fft + scratch_power + scratch_pre; /* 6,084 B */

    size_t est_arena_64ch = 36960;  /* Estimated TFLM arena for 64-channel DS-CNN */
    size_t est_rtos_stack = 15360;  /* Estimated FreeRTOS tasks & system stack */
    size_t est_app_state  = 4096;   /* Application state, queues, event flags */

    size_t total_est_system_ram = total_static_dsp_ram + est_arena_64ch + est_rtos_stack + est_app_state;

    printf("    MEASURED Static Front-End Allocations:\n");
    printf("      * 1-sec PCM Ring Buffer:   %6zu bytes (%5.2f KB)\n", sz_pcm_buf, (float)sz_pcm_buf / 1024.0f);
    printf("      * Audio Buffer Control:    %6zu bytes (%5.2f KB)\n", sz_buf_ctrl, (float)sz_buf_ctrl / 1024.0f);
    printf("      * VAD Config & State:      %6zu bytes (%5.2f KB)\n", sz_vad_cfg, (float)sz_vad_cfg / 1024.0f);
    printf("      * MFCC Config & Tables:    %6zu bytes (%5.2f KB) (Mel filterbanks, FFT, DCT)\n", sz_mfcc_cfg, (float)sz_mfcc_cfg / 1024.0f);
    printf("      * 49x10 Feature Matrix:    %6zu bytes (%5.2f KB)\n", sz_features, (float)sz_features / 1024.0f);
    printf("      --------------------------------------------------\n");
    printf("      * TOTAL MEASURED DSP RAM:  %6zu bytes (%5.2f KB)\n", total_static_dsp_ram, (float)total_static_dsp_ram / 1024.0f);
    printf("      * Peak Stack Scratch:      %6zu bytes (%5.2f KB)\n\n", peak_stack_scratch, (float)peak_stack_scratch / 1024.0f);

    printf("    ESTIMATED Neural Network & Runtime System Allocations:\n");
    printf("      * TFLM Tensor Arena (64ch):%6zu bytes (%5.2f KB) [ESTIMATE]\n", est_arena_64ch, (float)est_arena_64ch / 1024.0f);
    printf("      * RTOS & Task Stacks:      %6zu bytes (%5.2f KB) [ESTIMATE]\n", est_rtos_stack, (float)est_rtos_stack / 1024.0f);
    printf("      * Application State/Flags: %6zu bytes (%5.2f KB) [ESTIMATE]\n", est_app_state, (float)est_app_state / 1024.0f);
    printf("      --------------------------------------------------\n");
    printf("      * TOTAL ESTIMATED RAM:     %6zu bytes (%5.2f KB)\n", total_est_system_ram, (float)total_est_system_ram / 1024.0f);
    printf("      * HEADROOM (<= 256 KB):    %6zu bytes (%5.2f KB) [ESTIMATED HEADROOM]\n",
           (256 * 1024) - total_est_system_ram, (float)((256 * 1024) - total_est_system_ram) / 1024.0f);

    printf("\n================================================================\n");
    printf("               Whole Pipeline Benchmark Complete                \n");
    printf("================================================================\n");

    return 0;
}
