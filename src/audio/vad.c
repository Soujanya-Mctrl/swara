#include "vad.h"
#include <math.h>
#include <string.h>

void vad_init(vad_config_t* vad, uint32_t energy_threshold, uint16_t hangover_frames) {
    if (!vad) return;
    vad->energy_threshold = (energy_threshold > 0) ? energy_threshold : SWARA_VAD_DEFAULT_THRESHOLD;
    vad->hangover_frames = hangover_frames;
    vad->current_hangover = 0;
    vad->initialized = true;
}

void vad_reset(vad_config_t* vad) {
    if (!vad) return;
    vad->current_hangover = 0;
}

uint32_t vad_calculate_rms(const int16_t* samples, size_t count) {
    if (!samples || count == 0) return 0;

    uint64_t sum_squares = 0;
    for (size_t i = 0; i < count; i++) {
        int32_t val = (int32_t)samples[i];
        sum_squares += (uint64_t)(val * val);
    }

    uint64_t mean_square = sum_squares / count;
    return (uint32_t)sqrt((double)mean_square);
}

bool vad_process_frame(vad_config_t* vad, const int16_t* frame, size_t count) {
    if (!vad || !frame || count == 0) return false;

    uint32_t rms = vad_calculate_rms(frame, count);

    if (rms >= vad->energy_threshold) {
        /* Speech activity detected: refresh hangover window */
        vad->current_hangover = vad->hangover_frames;
        return true;
    }

    /* Silence / noise frame */
    if (vad->current_hangover > 0) {
        vad->current_hangover--;
        return true; /* Maintain speech state during hangover smoothing */
    }

    return false;
}
