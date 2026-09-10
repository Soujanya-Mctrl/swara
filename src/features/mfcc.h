#ifndef SWARA_MFCC_H_
#define SWARA_MFCC_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fft.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SWARA_MEL_FILTERS_COUNT   (20)
#define SWARA_MFCC_COEFFS_COUNT   (10)
#define SWARA_LOW_FREQ_HZ         (20.0f)
#define SWARA_HIGH_FREQ_HZ        (8000.0f)

/**
 * @brief Sparse Mel Filter Representation
 */
typedef struct {
    uint16_t start_bin;
    uint16_t num_bins;
    float weights[SWARA_FFT_BINS];
} mel_filter_t;

/**
 * @brief MFCC Context containing precalculated Mel filterbanks and DCT matrix.
 * Statically allocated with zero dynamic memory.
 */
typedef struct {
    mel_filter_t mel_filters[SWARA_MEL_FILTERS_COUNT];
    float dct_matrix[SWARA_MFCC_COEFFS_COUNT][SWARA_MEL_FILTERS_COUNT];
    fft_config_t fft_cfg;
    bool initialized;
} mfcc_config_t;

/**
 * @brief Initialize the MFCC processor (builds Mel filterbank and DCT-II basis matrix).
 *
 * @param config Pointer to mfcc_config_t instance.
 */
void mfcc_init(mfcc_config_t* config);

/**
 * @brief Compute 10 MFCC coefficients from 257-bin FFT power spectrum.
 *
 * @param config Pointer to initialized mfcc_config_t.
 * @param power_spectrum Array of 257 float power spectrum values.
 * @param out_mfcc Destination array for 10 MFCC coefficients.
 */
void mfcc_compute_from_power_spectrum(
    const mfcc_config_t* config,
    const float* power_spectrum,
    float* out_mfcc
);

#define SWARA_PREEMPHASIS_COEFF   (0.97f)

/**
 * @brief Apply pre-emphasis filter to a 480-sample frame: y[n] = x[n] - 0.97 * x[n-1]
 *
 * @param in_samples Input array of 480 signed 16-bit PCM samples.
 * @param out_samples Destination array of 480 signed 16-bit PCM samples.
 */
void mfcc_apply_pre_emphasis(const int16_t* in_samples, int16_t* out_samples);

/**
 * @brief Extract 10 MFCC coefficients for a single 480-sample frame:
 * PCM -> Pre-emphasis -> Hamming -> 512 FFT -> Power spectrum -> 20 Mel filters -> log() -> DCT-II -> 10 MFCCs.
 *
 * @param config Pointer to initialized mfcc_config_t.
 * @param samples Array of 480 signed 16-bit PCM samples.
 * @param out_mfcc Destination array for 10 MFCC coefficients.
 */
void mfcc_extract_frame(
    const mfcc_config_t* config,
    const int16_t* samples,
    float* out_mfcc
);

/**
 * @brief Extract full 49 x 10 MFCC feature matrix from a 1-second (16,000 samples) window.
 *
 * @param config Pointer to initialized mfcc_config_t.
 * @param window_samples Array of 16,000 signed 16-bit PCM samples.
 * @param out_features Destination 49x10 matrix [49 frames][10 coefficients].
 */
void mfcc_compute_window(
    const mfcc_config_t* config,
    const int16_t* window_samples,
    float out_features[49][SWARA_MFCC_COEFFS_COUNT]
);

#ifdef __cplusplus
}
#endif

#endif /* SWARA_MFCC_H_ */
