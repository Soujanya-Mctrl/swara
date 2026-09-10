#include "mfcc.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Hz to Mel conversion */
static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + (hz / 700.0f));
}

/* Mel to Hz conversion */
static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void mfcc_init(mfcc_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(mfcc_config_t));

    /* 1. Initialize underlying FFT module */
    fft_init(&config->fft_cfg);

    /* 2. Build 20 triangular Mel filterbanks */
    float mel_min = hz_to_mel(SWARA_LOW_FREQ_HZ);
    float mel_max = hz_to_mel(SWARA_HIGH_FREQ_HZ);
    float mel_step = (mel_max - mel_min) / (float)(SWARA_MEL_FILTERS_COUNT + 1);

    /* 22 center points in Mel scale */
    float filter_centers_hz[SWARA_MEL_FILTERS_COUNT + 2];
    int filter_bins[SWARA_MEL_FILTERS_COUNT + 2];

    for (int i = 0; i < SWARA_MEL_FILTERS_COUNT + 2; i++) {
        float mel = mel_min + ((float)i * mel_step);
        filter_centers_hz[i] = mel_to_hz(mel);
        /* Bin index: f * N / Fs */
        int bin = (int)floorf(filter_centers_hz[i] * (float)SWARA_FFT_SIZE / 16000.0f);
        if (bin < 0) bin = 0;
        if (bin >= SWARA_FFT_BINS) bin = SWARA_FFT_BINS - 1;
        filter_bins[i] = bin;
    }

    /* Build triangular weights for each of the 20 filters */
    for (int m = 0; m < SWARA_MEL_FILTERS_COUNT; m++) {
        int left_bin   = filter_bins[m];
        int center_bin = filter_bins[m + 1];
        int right_bin  = filter_bins[m + 2];

        /* Guard against adjacent duplicate bins at low frequencies */
        if (center_bin == left_bin) center_bin = left_bin + 1;
        if (right_bin <= center_bin) right_bin = center_bin + 1;
        if (right_bin >= SWARA_FFT_BINS) right_bin = SWARA_FFT_BINS - 1;

        config->mel_filters[m].start_bin = (uint16_t)left_bin;
        config->mel_filters[m].num_bins  = (uint16_t)(right_bin - left_bin + 1);

        for (int k = left_bin; k < center_bin; k++) {
            float weight = (float)(k - left_bin) / (float)(center_bin - left_bin);
            config->mel_filters[m].weights[k] = weight;
        }
        for (int k = center_bin; k <= right_bin; k++) {
            float weight = (float)(right_bin - k) / (float)(right_bin - center_bin);
            config->mel_filters[m].weights[k] = weight;
        }
    }

    /* 3. Precompute DCT-II basis matrix (10 coeffs x 20 filters) */
    for (int i = 0; i < SWARA_MFCC_COEFFS_COUNT; i++) {
        for (int m = 0; m < SWARA_MEL_FILTERS_COUNT; m++) {
            float angle = ((float)M_PI * (float)i * ((float)m + 0.5f)) / (float)SWARA_MEL_FILTERS_COUNT;
            config->dct_matrix[i][m] = cosf(angle);
        }
    }

    config->initialized = true;
}

void mfcc_compute_from_power_spectrum(
    const mfcc_config_t* config,
    const float* power_spectrum,
    float* out_mfcc
) {
    if (!config || !power_spectrum || !out_mfcc) return;

    /* 1. Filterbank energies */
    float mel_energies[SWARA_MEL_FILTERS_COUNT];

    for (int m = 0; m < SWARA_MEL_FILTERS_COUNT; m++) {
        float sum = 0.0f;
        uint16_t start = config->mel_filters[m].start_bin;
        uint16_t count = config->mel_filters[m].num_bins;

        for (uint16_t j = 0; j < count; j++) {
            uint16_t k = start + j;
            if (k < SWARA_FFT_BINS) {
                sum += power_spectrum[k] * config->mel_filters[m].weights[k];
            }
        }

        /* 2. Log compression with floor to prevent log(0) */
        mel_energies[m] = logf(sum + 1e-6f);
    }

    /* 3. Discrete Cosine Transform (DCT-II) */
    for (int i = 0; i < SWARA_MFCC_COEFFS_COUNT; i++) {
        float val = 0.0f;
        for (int m = 0; m < SWARA_MEL_FILTERS_COUNT; m++) {
            val += mel_energies[m] * config->dct_matrix[i][m];
        }
        out_mfcc[i] = val;
    }
}

void mfcc_apply_pre_emphasis(const int16_t* in_samples, int16_t* out_samples) {
    if (!in_samples || !out_samples) return;
    out_samples[0] = in_samples[0];
    for (size_t i = 1; i < SWARA_FRAME_LEN; i++) {
        float val = (float)in_samples[i] - (SWARA_PREEMPHASIS_COEFF * (float)in_samples[i - 1]);
        if (val > 32767.0f) val = 32767.0f;
        if (val < -32768.0f) val = -32768.0f;
        out_samples[i] = (int16_t)val;
    }
}

void mfcc_extract_frame(
    const mfcc_config_t* config,
    const int16_t* samples,
    float* out_mfcc
) {
    if (!config || !samples || !out_mfcc) return;

    /* 1. Pre-emphasis */
    int16_t pre_emphasized[SWARA_FRAME_LEN];
    mfcc_apply_pre_emphasis(samples, pre_emphasized);

    /* 2. FFT & Power spectrum (applies Hamming window & 512-point FFT) */
    float power_spectrum[SWARA_FFT_BINS];
    fft_compute_power_spectrum(&config->fft_cfg, pre_emphasized, power_spectrum);

    /* 3. Mel filterbank & DCT-II */
    mfcc_compute_from_power_spectrum(config, power_spectrum, out_mfcc);
}

void mfcc_compute_window(
    const mfcc_config_t* config,
    const int16_t* window_samples,
    float out_features[49][SWARA_MFCC_COEFFS_COUNT]
) {
    if (!config || !window_samples || !out_features) return;

    for (int frame = 0; frame < 49; frame++) {
        const int16_t* frame_ptr = &window_samples[frame * 320];
        mfcc_extract_frame(config, frame_ptr, out_features[frame]);
    }
}
