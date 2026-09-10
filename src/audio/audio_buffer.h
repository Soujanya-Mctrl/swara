#ifndef SWARA_AUDIO_BUFFER_H_
#define SWARA_AUDIO_BUFFER_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Frozen Audio Specification (V0) Constants
 */
#define SWARA_SAMPLE_RATE_HZ       (16000)
#define SWARA_WINDOW_DURATION_MS   (1000)
#define SWARA_WINDOW_SAMPLES       (16000)   /* 1.0s @ 16 kHz = 16,000 samples */
#define SWARA_FRAME_LENGTH_MS      (30)
#define SWARA_FRAME_LENGTH_SAMPLES (480)     /* 30ms @ 16 kHz = 480 samples */
#define SWARA_FRAME_STEP_MS        (20)
#define SWARA_FRAME_STEP_SAMPLES   (320)     /* 20ms @ 16 kHz = 320 samples */
#define SWARA_FFT_SIZE             (512)     /* Radix-2 FFT size */
#define SWARA_MEL_FILTERS          (20)      /* 20 triangular Mel filters */
#define SWARA_MFCC_COEFFS          (10)      /* 10 MFCC coefficients */
#define SWARA_NUM_FRAMES           (49)      /* 1 + (16000 - 480) / 320 = 49 */

/**
 * @brief Circular Ring Buffer for 16-bit PCM audio samples.
 * Designed for embedded targets with zero dynamic memory allocation.
 */
typedef struct {
    int16_t* storage;              /**< Pointer to external or statically allocated buffer */
    size_t capacity;               /**< Maximum sample capacity (16,000 for 1-second window) */
    volatile uint32_t write_index; /**< Volatile write index for continuous microphone writes */
    size_t count;                  /**< Total number of valid samples currently stored */
} audio_buffer_t;

/**
 * @brief Initialize an audio buffer with external storage.
 *
 * @param buf Pointer to audio_buffer_t struct.
 * @param storage Memory buffer allocated by caller (int16_t array).
 * @param capacity Maximum capacity in samples.
 * @return true on success, false on invalid parameters.
 */
bool audio_buffer_init(audio_buffer_t* buf, int16_t* storage, size_t capacity);

/**
 * @brief Reset the audio buffer count and indices without freeing memory.
 *
 * @param buf Pointer to audio_buffer_t struct.
 */
void audio_buffer_reset(audio_buffer_t* buf);

/**
 * @brief Write new audio samples into the circular buffer.
 * If the buffer overflows, oldest samples are overwritten.
 *
 * @param buf Pointer to audio_buffer_t struct.
 * @param samples Array of 16-bit signed PCM samples.
 * @param count Number of samples to write.
 * @return Number of samples written.
 */
size_t audio_buffer_write(audio_buffer_t* buf, const int16_t* samples, size_t count);

/**
 * @brief Read the most recent N samples (e.g. 480-sample frame for VAD/FFT).
 *
 * @param buf Pointer to audio_buffer_t struct.
 * @param out_frame Destination buffer of size at least frame_length.
 * @param frame_length Number of recent samples to extract.
 * @return true if requested frame_length samples were extracted, false if insufficient samples.
 */
bool audio_buffer_get_latest_frame(const audio_buffer_t* buf, int16_t* out_frame, size_t frame_length);

/**
 * @brief Read the entire sliding window (e.g. 16,000 samples) in chronological order.
 *
 * @param buf Pointer to audio_buffer_t struct.
 * @param out_window Destination buffer of size at least window_length.
 * @param window_length Number of samples to extract (up to buf->count).
 * @return Number of samples copied.
 */
size_t audio_buffer_get_window(const audio_buffer_t* buf, int16_t* out_window, size_t window_length);

/**
 * @brief Get the current number of valid samples stored in the buffer.
 */
size_t audio_buffer_get_count(const audio_buffer_t* buf);

/**
 * @brief Check if the buffer holds at least one full analysis window (16,000 samples).
 */
bool audio_buffer_is_window_ready(const audio_buffer_t* buf);

#ifdef __cplusplus
}
#endif

#endif /* SWARA_AUDIO_BUFFER_H_ */
