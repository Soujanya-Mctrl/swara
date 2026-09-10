#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../src/features/mfcc.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_mfcc_init(void) {
    printf("[TEST] mfcc_init...\n");
    mfcc_config_t cfg;
    mfcc_init(&cfg);
    assert(cfg.initialized == true);

    // Verify filterbank bins
    for (int m = 0; m < SWARA_MEL_FILTERS_COUNT; m++) {
        assert(cfg.mel_filters[m].num_bins > 0);
        assert(cfg.mel_filters[m].start_bin < SWARA_FFT_BINS);
    }
    printf("       -> PASSED (20 Mel filters initialized)\n");
}

void test_mfcc_silence(void) {
    printf("[TEST] mfcc silence handling...\n");
    mfcc_config_t cfg;
    mfcc_init(&cfg);

    int16_t silence[SWARA_FRAME_LEN] = {0};
    float mfcc_out[SWARA_MFCC_COEFFS_COUNT];

    mfcc_extract_frame(&cfg, silence, mfcc_out);

    // Verify no NaNs or Infs
    for (int i = 0; i < SWARA_MFCC_COEFFS_COUNT; i++) {
        assert(!isnan(mfcc_out[i]));
        assert(!isinf(mfcc_out[i]));
    }
    printf("       -> PASSED (Silence produced finite values without NaN/Inf)\n");
}

void test_mfcc_tone_extraction(void) {
    printf("[TEST] mfcc 1000 Hz tone feature extraction...\n");
    mfcc_config_t cfg;
    mfcc_init(&cfg);

    int16_t tone[SWARA_FRAME_LEN];
    for (int i = 0; i < SWARA_FRAME_LEN; i++) {
        float val = sinf(2.0f * (float)M_PI * 1000.0f * (float)i / 16000.0f);
        tone[i] = (int16_t)(val * 28000.0f);
    }

    float mfcc_out[SWARA_MFCC_COEFFS_COUNT];
    mfcc_extract_frame(&cfg, tone, mfcc_out);

    printf("       MFCC Coefficients [0..9]:\n       ");
    for (int i = 0; i < SWARA_MFCC_COEFFS_COUNT; i++) {
        assert(!isnan(mfcc_out[i]));
        assert(!isinf(mfcc_out[i]));
        printf("%.2f ", mfcc_out[i]);
    }
    printf("\n       -> PASSED\n");
}

int main(void) {
    printf("=== Running MFCC Unit Tests ===\n");
    test_mfcc_init();
    test_mfcc_silence();
    test_mfcc_tone_extraction();
    printf("=== All MFCC Tests Passed Successfully! ===\n");
    return 0;
}
