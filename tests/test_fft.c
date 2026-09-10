#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../src/features/fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void test_fft_init(void) {
    printf("[TEST] fft_init...\n");
    fft_config_t cfg;
    fft_init(&cfg);
    assert(cfg.initialized == true);
    // Verify Hamming window edges and center (0.54 - 0.46 = 0.08 at edges, 1.0 at center)
    assert(fabsf(cfg.hamming_window[0] - 0.08f) < 0.01f);
    assert(fabsf(cfg.hamming_window[SWARA_FRAME_LEN / 2] - 1.0f) < 0.02f);
    printf("       -> PASSED\n");
}

void test_fft_pure_sine_tone(void) {
    printf("[TEST] fft pure 1000 Hz sine wave detection...\n");
    fft_config_t cfg;
    fft_init(&cfg);

    int16_t tone[SWARA_FRAME_LEN];
    float freq_hz = 1000.0f;
    float sample_rate = 16000.0f;

    // Generate 1000 Hz tone
    for (int i = 0; i < SWARA_FRAME_LEN; i++) {
        float val = sinf(2.0f * (float)M_PI * freq_hz * (float)i / sample_rate);
        tone[i] = (int16_t)(val * 30000.0f);
    }

    float power_spectrum[SWARA_FFT_BINS];
    fft_compute_power_spectrum(&cfg, tone, power_spectrum);

    // Expected bin: k = 1000 * 512 / 16000 = 32
    int expected_bin = 32;
    int max_bin = 0;
    float max_power = 0.0f;

    for (int k = 1; k < SWARA_FFT_BINS; k++) { // Skip DC
        if (power_spectrum[k] > max_power) {
            max_power = power_spectrum[k];
            max_bin = k;
        }
    }

    printf("       Expected peak bin: %d, Detected peak bin: %d (Max Power: %f)\n", expected_bin, max_bin, max_power);
    assert(max_bin == expected_bin);
    printf("       -> PASSED\n");
}

int main(void) {
    printf("=== Running FFT Unit Tests ===\n");
    test_fft_init();
    test_fft_pure_sine_tone();
    printf("=== All FFT Tests Passed Successfully! ===\n");
    return 0;
}
