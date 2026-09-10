#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "../src/audio/wav_reader.h"
#include "../src/audio/vad.h"
#include "../src/features/mfcc.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WINDOW_LEN 16000

/* Generate 1-second silence WAV */
void create_silence_wav(const char* filepath) {
    int16_t silence[WINDOW_LEN] = {0};
    wav_write_samples(filepath, silence, WINDOW_LEN, 16000);
}

/* Generate 1-second 1 kHz pure sine tone WAV */
void create_tone_wav(const char* filepath, float freq_hz) {
    int16_t tone[WINDOW_LEN];
    for (int i = 0; i < WINDOW_LEN; i++) {
        float val = sinf(2.0f * (float)M_PI * freq_hz * (float)i / 16000.0f);
        tone[i] = (int16_t)(val * 24000.0f);
    }
    wav_write_samples(filepath, tone, WINDOW_LEN, 16000);
}

/* Generate 1-second synthetic speech vowel /a/ with formant resonators (F1=700Hz, F2=1220Hz, F3=2600Hz) */
void create_speech_wav(const char* filepath) {
    int16_t speech[WINDOW_LEN];
    float f0 = 130.0f; /* Fundamental pitch 130 Hz */
    for (int i = 0; i < WINDOW_LEN; i++) {
        float t = (float)i / 16000.0f;
        /* Harmonic excitation modulated with formants */
        float glottal = sinf(2.0f * (float)M_PI * f0 * t) + 0.5f * sinf(4.0f * (float)M_PI * f0 * t);
        float f1 = sinf(2.0f * (float)M_PI * 700.0f * t) * 0.5f;
        float f2 = sinf(2.0f * (float)M_PI * 1220.0f * t) * 0.3f;
        float f3 = sinf(2.0f * (float)M_PI * 2600.0f * t) * 0.15f;
        float composite = glottal * (f1 + f2 + f3);
        speech[i] = (int16_t)(composite * 20000.0f);
    }
    wav_write_samples(filepath, speech, WINDOW_LEN, 16000);
}

void test_pipeline_silence(const mfcc_config_t* cfg) {
    printf("[TEST A] 1-sec Silence WAV -> 49 x 10 MFCC...\n");
    const char* file = "test_silence.wav";
    create_silence_wav(file);

    int16_t samples[WINDOW_LEN];
    size_t read_count = 0;
    wav_status_t status = wav_read_samples(file, samples, WINDOW_LEN, &read_count);
    assert(status == SWARA_WAV_OK);
    assert(read_count == WINDOW_LEN);

    float features[49][10];
    mfcc_compute_window(cfg, samples, features);

    /* Verify no NaN, no Inf, all finite across all 49 frames and 10 coefficients */
    for (int f = 0; f < 49; f++) {
        for (int c = 0; c < 10; c++) {
            assert(!isnan(features[f][c]));
            assert(!isinf(features[f][c]));
        }
    }
    printf("         -> PASSED: All 49x10 values are finite, no NaN/Inf detected.\n");
}

void test_pipeline_tone(const mfcc_config_t* cfg) {
    printf("[TEST B] 1-sec 1000 Hz Tone WAV -> 49 x 10 MFCC...\n");
    const char* file = "test_tone.wav";
    create_tone_wav(file, 1000.0f);

    int16_t samples[WINDOW_LEN];
    size_t read_count = 0;
    wav_status_t status = wav_read_samples(file, samples, WINDOW_LEN, &read_count);
    assert(status == SWARA_WAV_OK);
    assert(read_count == WINDOW_LEN);

    float features[49][10];
    mfcc_compute_window(cfg, samples, features);

    /* Verify consistency across frames (steady sine tone should produce stable features) */
    for (int f = 5; f < 45; f++) {
        for (int c = 0; c < 10; c++) {
            assert(!isnan(features[f][c]));
            assert(!isinf(features[f][c]));
            /* Frame-to-frame variance should be very small for steady-state tone */
            float diff = fabsf(features[f][c] - features[5][c]);
            assert(diff < 2.0f);
        }
    }
    printf("         -> PASSED: 1 kHz tone features are stable, repeatable and verified.\n");
}

void test_pipeline_speech(const mfcc_config_t* cfg) {
    printf("[TEST C] 1-sec Formant Speech WAV -> 49 x 10 MFCC...\n");
    const char* file = "test_speech.wav";
    create_speech_wav(file);

    int16_t samples[WINDOW_LEN];
    size_t read_count = 0;
    wav_status_t status = wav_read_samples(file, samples, WINDOW_LEN, &read_count);
    assert(status == SWARA_WAV_OK);
    assert(read_count == WINDOW_LEN);

    float features[49][10];
    mfcc_compute_window(cfg, samples, features);

    for (int f = 0; f < 49; f++) {
        for (int c = 0; c < 10; c++) {
            assert(!isnan(features[f][c]));
            assert(!isinf(features[f][c]));
        }
    }

    printf("         Sample Frame 20 MFCCs:\n         ");
    for (int c = 0; c < 10; c++) {
        printf("%.2f ", features[20][c]);
    }
    printf("\n         -> PASSED: Valid 49x10 feature matrix generated.\n");
}

void benchmark_cpu(const mfcc_config_t* cfg) {
    printf("\n[BENCHMARK] Measuring 1-second audio (49 frames) execution time...\n");
    int16_t samples[WINDOW_LEN];
    for (int i = 0; i < WINDOW_LEN; i++) {
        samples[i] = (int16_t)(i * 3);
    }

    float features[49][10];

    const int iterations = 100;
    clock_t start = clock();
    for (int it = 0; it < iterations; it++) {
        mfcc_compute_window(cfg, samples, features);
    }
    clock_t end = clock();

    double total_sec = (double)(end - start) / (double)CLOCKS_PER_SEC;
    double sec_per_window = total_sec / (double)iterations;
    double ms_per_window = sec_per_window * 1000.0;
    double ms_per_frame = ms_per_window / 49.0;
    /* 1-second audio processed in ms_per_window ms. CPU % of real-time = (ms_per_window / 1000ms) * 100% */
    double real_time_cpu_pct = (ms_per_window / 1000.0) * 100.0;

    printf("       Total time for %d windows: %.4f s\n", iterations, total_sec);
    printf("       Time per 1-second window (49 frames): %.3f ms\n", ms_per_window);
    printf("       Time per frame (480 samples / 30 ms): %.3f ms\n", ms_per_frame);
    printf("       Real-time CPU utilization equivalent: %.2f%% (host processor)\n", real_time_cpu_pct);
}

int main(void) {
    printf("=== Swara End-to-End WAV -> MFCC Pipeline Verification ===\n");
    mfcc_config_t cfg;
    mfcc_init(&cfg);

    test_pipeline_silence(&cfg);
    test_pipeline_tone(&cfg);
    test_pipeline_speech(&cfg);
    benchmark_cpu(&cfg);

    printf("\n=== All Pipeline Verification Tests Passed Successfully! ===\n");
    return 0;
}
