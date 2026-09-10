#ifndef SWARA_VAD_H_
#define SWARA_VAD_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SWARA_VAD_DEFAULT_THRESHOLD  (350)  /* Default RMS threshold on 16-bit PCM */
#define SWARA_VAD_DEFAULT_HANGOVER   (5)    /* 5 frames (100 ms) hangover smoothing */

/**
 * @brief Energy-based Voice Activity Detection (VAD) state.
 * Uses zero dynamic memory.
 */
typedef struct {
    uint32_t energy_threshold;  /**< RMS threshold above which speech is detected */
    uint16_t hangover_frames;   /**< Number of frames to hold active state after speech stops */
    uint16_t current_hangover;  /**< Remaining hangover countdown */
    bool initialized;
} vad_config_t;

/**
 * @brief Initialize VAD with energy threshold and hangover frames.
 *
 * @param vad Pointer to vad_config_t instance.
 * @param energy_threshold RMS energy threshold (e.g., 350 for normal speech).
 * @param hangover_frames Hangover smoothing frame count (e.g., 5 frames = 100 ms).
 */
void vad_init(vad_config_t* vad, uint32_t energy_threshold, uint16_t hangover_frames);

/**
 * @brief Reset VAD hangover state without changing threshold parameters.
 */
void vad_reset(vad_config_t* vad);

/**
 * @brief Calculate RMS energy of a frame of 16-bit signed PCM samples.
 *
 * @param samples Array of 16-bit PCM audio samples.
 * @param count Number of samples (e.g. 480).
 * @return RMS energy value.
 */
uint32_t vad_calculate_rms(const int16_t* samples, size_t count);

/**
 * @brief Process an audio frame and determine whether speech/voice activity is present.
 *
 * @param vad Pointer to initialized vad_config_t.
 * @param frame Array of 16-bit PCM samples (e.g. 480 samples).
 * @param count Number of samples.
 * @return true if speech detected (or inside hangover window), false if silence/noise.
 */
bool vad_process_frame(vad_config_t* vad, const int16_t* frame, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* SWARA_VAD_H_ */
