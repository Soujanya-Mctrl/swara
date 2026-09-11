#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "../src/audio/wav_reader.h"
#include "../src/features/mfcc.h"

#define SAMPLES_1SEC 16000
#define NUM_FRAMES   49
#define NUM_COEFFS   10
#define CANARY_INT16 ((int16_t)0x5A5A)
#define CANARY_FLOAT 12345.678f

/* Try primary path or parent build directory path */
static const char* resolve_wav_path(const char* rel_path) {
    FILE* fp = fopen(rel_path, "rb");
    if (fp) {
        fclose(fp);
        return rel_path;
    }
    static char alt_path[256];
    snprintf(alt_path, sizeof(alt_path), "../%s", rel_path);
    fp = fopen(alt_path, "rb");
    if (fp) {
        fclose(fp);
        return alt_path;
    }
    return rel_path;
}

int main(void) {
    printf("=== Running Swara WAV -> MFCC Pipeline Integration Test ===\n\n");

    const char* target_file = resolve_wav_path("tests/data/test_16k_1s.wav");
    printf("[1] Locating test WAV fixture: %s\n", target_file);

    /* --- Step 1: Read & validate WAV Header --- */
    wav_info_t info;
    wav_status_t status = wav_read_header(target_file, &info);
    assert(status == SWARA_WAV_OK);
    assert(info.sample_rate == 16000);
    assert(info.num_channels == 1);
    assert(info.bits_per_sample == 16);
    assert(info.num_samples == SAMPLES_1SEC);
    printf("    -> Header validated: 16000 Hz, 1 channel, 16-bit, %u samples.\n", info.num_samples);

    /* --- Step 2: Load 16,000 samples with canary guards to verify zero buffer overrun --- */
    printf("[2] Reading 16,000 samples with buffer overrun canary checks...\n");
    int16_t guard_before[32];
    int16_t samples[SAMPLES_1SEC];
    int16_t guard_after[32];

    for (int i = 0; i < 32; i++) {
        guard_before[i] = CANARY_INT16;
        guard_after[i] = CANARY_INT16;
    }

    size_t samples_read = 0;
    status = wav_read_samples(target_file, samples, SAMPLES_1SEC, &samples_read);
    assert(status == SWARA_WAV_OK);
    assert(samples_read == SAMPLES_1SEC);

    /* Verify guards were not modified */
    for (int i = 0; i < 32; i++) {
        assert(guard_before[i] == CANARY_INT16);
        assert(guard_after[i] == CANARY_INT16);
    }
    printf("    -> Successfully loaded exactly %zu samples. Canary guards intact.\n", samples_read);

    /* --- Step 3: Initialize MFCC configuration (zero dynamic allocation) --- */
    printf("[3] Initializing MFCC DSP engine...\n");
    mfcc_config_t cfg;
    mfcc_init(&cfg);
    assert(cfg.initialized == true);
    printf("    -> MFCC processor initialized (20 Mel filterbanks, 10 DCT coefficients, 512 FFT).\n");

    /* --- Step 4: Compute full 49 x 10 MFCC window with output canary guards --- */
    printf("[4] Computing full 49 x 10 MFCC feature window...\n");
    float feat_guard_before[16];
    float features[NUM_FRAMES][NUM_COEFFS];
    float feat_guard_after[16];

    for (int i = 0; i < 16; i++) {
        feat_guard_before[i] = CANARY_FLOAT;
        feat_guard_after[i] = CANARY_FLOAT;
    }

    mfcc_compute_window(&cfg, samples, features);

    /* Verify feature matrix bounds */
    for (int i = 0; i < 16; i++) {
        assert(feat_guard_before[i] == CANARY_FLOAT);
        assert(feat_guard_after[i] == CANARY_FLOAT);
    }
    printf("    -> Exactly 49 frames computed without matrix boundary overrun.\n");

    /* --- Step 5: Verify every single coefficient is finite (no NaN, no Inf) --- */
    printf("[5] Verifying numerical validity of 49 x 10 feature matrix...\n");
    for (int f = 0; f < NUM_FRAMES; f++) {
        for (int c = 0; c < NUM_COEFFS; c++) {
            float val = features[f][c];
            assert(!isnan(val));
            assert(!isinf(val));
        }
    }
    printf("    -> All 490 coefficients are finite numbers (no NaN or Inf).\n");

    /* Print sample coefficients from frame 0 and frame 24 */
    printf("    Sample Frame 0 MFCCs:  [");
    for (int c = 0; c < NUM_COEFFS; c++) printf("%.2f%s", features[0][c], c < 9 ? ", " : "]\n");
    printf("    Sample Frame 24 MFCCs: [");
    for (int c = 0; c < NUM_COEFFS; c++) printf("%.2f%s", features[24][c], c < 9 ? ", " : "]\n");

    /* --- Step 6: Verify deterministic extraction repeatability --- */
    printf("[6] Verifying deterministic repeatability across independent runs...\n");
    float features_repeat[NUM_FRAMES][NUM_COEFFS];
    mfcc_compute_window(&cfg, samples, features_repeat);

    for (int f = 0; f < NUM_FRAMES; f++) {
        for (int c = 0; c < NUM_COEFFS; c++) {
            assert(features[f][c] == features_repeat[f][c]);
        }
    }
    printf("    -> Extraction is 100%% deterministic (exact bit-for-bit match on repeated call).\n");

    /* --- Step 7: Verify clean rejection of invalid files --- */
    printf("[7] Verifying error handling on non-existent file...\n");
    wav_info_t bad_info;
    wav_status_t err_status = wav_read_header("non_existent_audio_file.wav", &bad_info);
    assert(err_status == SWARA_WAV_ERR_FILE_NOT_FOUND);
    printf("    -> Correctly returned SWARA_WAV_ERR_FILE_NOT_FOUND.\n");

    printf("\n=== All WAV -> MFCC Pipeline Integration Tests PASSED Successfully! ===\n");
    return 0;
}
