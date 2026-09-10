#ifndef SWARA_FFT_H_
#define SWARA_FFT_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SWARA_FFT_SIZE       (512)
#define SWARA_FFT_BINS       ((SWARA_FFT_SIZE / 2) + 1) /* 257 unique frequency bins */
#define SWARA_FRAME_LEN      (480)                      /* 30 ms @ 16 kHz */

/**
 * @brief FFT context containing precalculated tables and work buffers.
 * All memory is statically allocated or embedded within this structure.
 */
typedef struct {
    float hamming_window[SWARA_FRAME_LEN];
    float twiddle_cos[SWARA_FFT_SIZE / 2];
    float twiddle_sin[SWARA_FFT_SIZE / 2];
    uint16_t bit_reverse[SWARA_FFT_SIZE];
    bool initialized;
} fft_config_t;

/**
 * @brief Initialize FFT tables (Hanning window, twiddle factors, bit-reversal).
 * Call once during startup.
 *
 * @param config Pointer to fft_config_t instance.
 */
void fft_init(fft_config_t* config);

/**
 * @brief Apply Hanning window, zero-pad to 512, execute FFT, and compute power spectrum.
 *
 * @param config Pointer to initialized fft_config_t.
 * @param samples Array of 480 signed 16-bit PCM samples.
 * @param power_spectrum Output array of size 257 (SWARA_FFT_BINS) for |X[k]|^2.
 */
void fft_compute_power_spectrum(
    const fft_config_t* config,
    const int16_t* samples,
    float* power_spectrum
);

/**
 * @brief Raw complex FFT for N=512 points in-place.
 *
 * @param config Pointer to initialized fft_config_t.
 * @param real Array of 512 real components (in/out).
 * @param imag Array of 512 imaginary components (in/out).
 */
void fft_transform_radix2(
    const fft_config_t* config,
    float* real,
    float* imag
);

#ifdef __cplusplus
}
#endif

#endif /* SWARA_FFT_H_ */
