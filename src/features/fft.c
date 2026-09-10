#include "fft.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Reverse the bits of an integer (log2(512) = 9 bits) */
static uint16_t reverse_9_bits(uint16_t x) {
    uint16_t rev = 0;
    for (int i = 0; i < 9; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    return rev;
}

void fft_init(fft_config_t* config) {
    if (!config) return;

    /* 1. Precalculate Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1)) */
    for (int n = 0; n < SWARA_FRAME_LEN; n++) {
        config->hamming_window[n] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)n / (float)(SWARA_FRAME_LEN - 1));
    }

    /* 2. Precalculate Twiddle Factors: W_N^k = cos(2*pi*k / N) - j*sin(2*pi*k / N) */
    for (int k = 0; k < SWARA_FFT_SIZE / 2; k++) {
        float angle = -2.0f * (float)M_PI * (float)k / (float)SWARA_FFT_SIZE;
        config->twiddle_cos[k] = cosf(angle);
        config->twiddle_sin[k] = sinf(angle);
    }

    /* 3. Precalculate bit-reversal permutation */
    for (int i = 0; i < SWARA_FFT_SIZE; i++) {
        config->bit_reverse[i] = reverse_9_bits((uint16_t)i);
    }

    config->initialized = true;
}

void fft_transform_radix2(
    const fft_config_t* config,
    float* real,
    float* imag
) {
    if (!config || !real || !imag) return;

    /* Bit-reversal reordering */
    for (int i = 0; i < SWARA_FFT_SIZE; i++) {
        uint16_t target = config->bit_reverse[i];
        if (i < target) {
            float temp_r = real[i];
            float temp_i = imag[i];
            real[i] = real[target];
            imag[i] = imag[target];
            real[target] = temp_r;
            imag[target] = temp_i;
        }
    }

    /* Cooley-Tukey Radix-2 Butterfly Computation */
    for (int len = 2; len <= SWARA_FFT_SIZE; len <<= 1) {
        int half_len = len >> 1;
        int step = SWARA_FFT_SIZE / len;

        for (int i = 0; i < SWARA_FFT_SIZE; i += len) {
            for (int j = 0; j < half_len; j++) {
                int k = j * step;
                float c = config->twiddle_cos[k];
                float s = config->twiddle_sin[k];

                int u_idx = i + j;
                int v_idx = i + j + half_len;

                /* Complex multiplication: v * W */
                float tr = real[v_idx] * c - imag[v_idx] * s;
                float ti = real[v_idx] * s + imag[v_idx] * c;

                real[v_idx] = real[u_idx] - tr;
                imag[v_idx] = imag[u_idx] - ti;
                real[u_idx] = real[u_idx] + tr;
                imag[u_idx] = imag[u_idx] + ti;
            }
        }
    }
}

void fft_compute_power_spectrum(
    const fft_config_t* config,
    const int16_t* samples,
    float* power_spectrum
) {
    if (!config || !samples || !power_spectrum) return;

    /* Local stack work buffers: 512 floats each = 2 KB real, 2 KB imag */
    float real[SWARA_FFT_SIZE];
    float imag[SWARA_FFT_SIZE];

    /* 1. Apply Hamming window and normalize int16_t to float [-1.0, 1.0] */
    for (int i = 0; i < SWARA_FRAME_LEN; i++) {
        float norm_sample = (float)samples[i] / 32768.0f;
        real[i] = norm_sample * config->hamming_window[i];
        imag[i] = 0.0f;
    }

    /* 2. Zero-pad remaining 32 samples (480 to 511) */
    for (int i = SWARA_FRAME_LEN; i < SWARA_FFT_SIZE; i++) {
        real[i] = 0.0f;
        imag[i] = 0.0f;
    }

    /* 3. In-place FFT */
    fft_transform_radix2(config, real, imag);

    /* 4. Power spectrum: |X[k]|^2 for k = 0 to 256 */
    for (int k = 0; k < SWARA_FFT_BINS; k++) {
        power_spectrum[k] = (real[k] * real[k]) + (imag[k] * imag[k]);
    }
}
