#ifndef SWARA_WAV_READER_H_
#define SWARA_WAV_READER_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SWARA_WAV_OK = 0,
    SWARA_WAV_ERR_FILE_NOT_FOUND,
    SWARA_WAV_ERR_INVALID_HEADER,
    SWARA_WAV_ERR_UNSUPPORTED_FORMAT,   /* Non-PCM */
    SWARA_WAV_ERR_UNSUPPORTED_CHANNELS, /* Non-mono */
    SWARA_WAV_ERR_UNSUPPORTED_RATE,     /* Not 16,000 Hz */
    SWARA_WAV_ERR_UNSUPPORTED_BITS,     /* Not 16-bit */
    SWARA_WAV_ERR_READ_FAILED
} wav_status_t;

/**
 * @brief WAV file metadata header
 */
typedef struct {
    uint16_t num_channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    uint32_t num_samples;
    uint32_t data_bytes;
} wav_info_t;

/**
 * @brief Read and validate header of a WAV file.
 *
 * @param filepath Path to the WAV file.
 * @param info Output structure to receive metadata.
 * @return SWARA_WAV_OK if compliant 16kHz 16-bit mono PCM, error code otherwise.
 */
wav_status_t wav_read_header(const char* filepath, wav_info_t* info);

/**
 * @brief Read 16-bit PCM samples from a 16kHz mono WAV file.
 *
 * @param filepath Path to the WAV file.
 * @param out_samples Destination buffer for int16_t samples.
 * @param max_samples Maximum capacity of destination buffer.
 * @param out_read_samples Output pointer receiving the actual number of samples read.
 * @return SWARA_WAV_OK on success, error code on validation or I/O failure.
 */
wav_status_t wav_read_samples(
    const char* filepath,
    int16_t* out_samples,
    size_t max_samples,
    size_t* out_read_samples
);

/**
 * @brief Helper to write standard 16kHz 16-bit mono PCM WAV for test fixtures.
 *
 * @param filepath Destination path.
 * @param samples Array of 16-bit PCM audio samples.
 * @param count Number of samples.
 * @param sample_rate Audio sample rate (e.g. 16000).
 * @return true on success, false on write failure.
 */
bool wav_write_samples(
    const char* filepath,
    const int16_t* samples,
    size_t count,
    uint32_t sample_rate
);

#ifdef __cplusplus
}
#endif

#endif /* SWARA_WAV_READER_H_ */
