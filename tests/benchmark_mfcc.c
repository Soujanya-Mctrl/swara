#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
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

#define NUM_FRAME_ITERATIONS   5000
#define NUM_WINDOW_ITERATIONS  500

int main(void) {
    printf("================================================================\n");
    printf("           Swara MFCC Audio Front-End Performance Benchmark       \n");
    printf("================================================================\n\n");

    /* 1. Initialize MFCC engine */
    mfcc_config_t cfg;
    mfcc_init(&cfg);

    /* Synthetic test signal: 1-second (16,000 samples) */
    int16_t window_samples[16000];
    for (int i = 0; i < 16000; i++) {
        window_samples[i] = (int16_t)((i * 37) % 30000 - 15000);
    }

    int16_t single_frame[SWARA_FRAME_LEN];
    for (int i = 0; i < SWARA_FRAME_LEN; i++) {
        single_frame[i] = window_samples[i];
    }

    float single_mfcc[SWARA_MFCC_COEFFS_COUNT];
    float window_features[49][SWARA_MFCC_COEFFS_COUNT];

    /* Warm-up run */
    for (int i = 0; i < 50; i++) {
        mfcc_extract_frame(&cfg, single_frame, single_mfcc);
        mfcc_compute_window(&cfg, window_samples, window_features);
    }

    /* 2. Benchmark Single 480-sample MFCC Frame */
    printf("[1] Benchmarking Single 480-sample (30 ms) Frame Extraction...\n");
    printf("    Iterations: %d\n", NUM_FRAME_ITERATIONS);

    double start_frame = get_time_sec();
    for (int it = 0; it < NUM_FRAME_ITERATIONS; it++) {
        mfcc_extract_frame(&cfg, single_frame, single_mfcc);
    }
    double end_frame = get_time_sec();

    double total_frame_sec = end_frame - start_frame;
    double sec_per_frame = total_frame_sec / (double)NUM_FRAME_ITERATIONS;
    double us_per_frame = sec_per_frame * 1e6;
    double ms_per_frame = sec_per_frame * 1e3;
    double frames_per_sec = (double)NUM_FRAME_ITERATIONS / total_frame_sec;

    printf("    -> Total time:             %.4f s\n", total_frame_sec);
    printf("    -> Execution time / frame: %.2f us (%.4f ms)\n", us_per_frame, ms_per_frame);
    printf("    -> Throughput:             %.0f frames / second\n", frames_per_sec);
    printf("    -> Audio time factor:      %.1fx faster than real-time (30ms frame / %.4fms)\n\n",
           30.0 / ms_per_frame, ms_per_frame);

    /* 3. Benchmark Full 1-second Window (49 frames) */
    printf("[2] Benchmarking Full 1-second Window (49 frames, 16,000 samples)...\n");
    printf("    Iterations: %d\n", NUM_WINDOW_ITERATIONS);

    double start_window = get_time_sec();
    for (int it = 0; it < NUM_WINDOW_ITERATIONS; it++) {
        mfcc_compute_window(&cfg, window_samples, window_features);
    }
    double end_window = get_time_sec();

    double total_win_sec = end_window - start_window;
    double sec_per_win = total_win_sec / (double)NUM_WINDOW_ITERATIONS;
    double ms_per_win = sec_per_win * 1e3;
    double windows_per_sec = (double)NUM_WINDOW_ITERATIONS / total_win_sec;
    double real_time_duty_cycle_pct = (ms_per_win / 1000.0) * 100.0;

    printf("    -> Total time:             %.4f s\n", total_win_sec);
    printf("    -> Execution time / 1s win:%.3f ms\n", ms_per_win);
    printf("    -> Windows / second:       %.1f\n", windows_per_sec);
    printf("    -> Real-Time Factor (RTF): 1:%.0f (Processes 1s audio in %.3f ms)\n",
           1000.0 / ms_per_win, ms_per_win);
    printf("    -> Host CPU Duty Cycle:    %.3f%% (for active speech window)\n\n",
           real_time_duty_cycle_pct);

    /* 4. Memory Footprint Analysis */
    printf("[3] Front-End Static & Scratch Memory Accounting:\n");
    size_t sz_mfcc_cfg = sizeof(mfcc_config_t);
    size_t sz_fft_cfg  = sizeof(fft_config_t);
    size_t sz_filters  = sizeof(mel_filter_t) * SWARA_MEL_FILTERS_COUNT;
    size_t sz_dct      = sizeof(float) * SWARA_MFCC_COEFFS_COUNT * SWARA_MEL_FILTERS_COUNT;
    size_t sz_pcm_1s   = 16000 * sizeof(int16_t);
    size_t sz_features = 49 * SWARA_MFCC_COEFFS_COUNT * sizeof(float);

    /* Peak stack scratch in mfcc_extract_frame & fft_compute_power_spectrum */
    size_t scratch_fft_real   = SWARA_FFT_SIZE * sizeof(float);   /* 2048 B */
    size_t scratch_fft_imag   = SWARA_FFT_SIZE * sizeof(float);   /* 2048 B */
    size_t scratch_preemph    = SWARA_FRAME_LEN * sizeof(int16_t);/* 960 B */
    size_t scratch_power_spec = SWARA_FFT_BINS * sizeof(float);   /* 1028 B */
    size_t peak_stack_scratch = scratch_fft_real + scratch_fft_imag + scratch_preemph + scratch_power_spec;

    printf("    Static Front-End Structures:\n");
    printf("      * mfcc_config_t:         %zu bytes (%.2f KB)\n", sz_mfcc_cfg, (float)sz_mfcc_cfg / 1024.0f);
    printf("          - 20 Mel filterbanks:%zu bytes (%.2f KB)\n", sz_filters, (float)sz_filters / 1024.0f);
    printf("          - FFT tables (cfg):  %zu bytes (%.2f KB)\n", sz_fft_cfg, (float)sz_fft_cfg / 1024.0f);
    printf("          - DCT basis matrix:  %zu bytes (%.2f KB)\n", sz_dct, (float)sz_dct / 1024.0f);
    printf("      * 1-sec PCM Ring Buffer: %zu bytes (%.2f KB)\n", sz_pcm_1s, (float)sz_pcm_1s / 1024.0f);
    printf("      * 49x10 Feature Matrix:  %zu bytes (%.2f KB)\n", sz_features, (float)sz_features / 1024.0f);
    printf("      --------------------------------------------------\n");
    printf("      * TOTAL STATIC RAM:      %zu bytes (%.2f KB)\n\n",
           sz_mfcc_cfg + sz_pcm_1s + sz_features,
           (float)(sz_mfcc_cfg + sz_pcm_1s + sz_features) / 1024.0f);

    printf("    Peak Call Stack Scratch Memory:\n");
    printf("      * FFT real + imag (512x2): %zu bytes\n", scratch_fft_real + scratch_fft_imag);
    printf("      * Power spectrum (257):    %zu bytes\n", scratch_power_spec);
    printf("      * Pre-emphasis buffer:     %zu bytes\n", scratch_preemph);
    printf("      --------------------------------------------------\n");
    printf("      * PEAK CALL STACK SCRATCH: %zu bytes (%.2f KB)\n\n",
           peak_stack_scratch, (float)peak_stack_scratch / 1024.0f);

    printf("================================================================\n");
    printf("                     Benchmark Complete                         \n");
    printf("================================================================\n");

    return 0;
}
