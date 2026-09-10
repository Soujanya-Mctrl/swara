#include "audio_buffer.h"
#include <string.h>

bool audio_buffer_init(audio_buffer_t* buf, int16_t* storage, size_t capacity) {
    if (!buf || !storage || capacity == 0) {
        return false;
    }
    buf->storage = storage;
    buf->capacity = capacity;
    buf->write_index = 0;
    buf->count = 0;
    return true;
}

void audio_buffer_reset(audio_buffer_t* buf) {
    if (!buf) return;
    buf->write_index = 0;
    buf->count = 0;
}

size_t audio_buffer_write(audio_buffer_t* buf, const int16_t* samples, size_t count) {
    if (!buf || !buf->storage || !samples || count == 0) {
        return 0;
    }

    /* In microcontrollers, samples may arrive continuously from I2S DMA interrupts */
    for (size_t i = 0; i < count; i++) {
        buf->storage[buf->write_index] = samples[i];
        buf->write_index = (buf->write_index + 1) % (uint32_t)buf->capacity;
    }

    buf->count += count;
    if (buf->count > buf->capacity) {
        buf->count = buf->capacity; /* Oldest samples overwritten */
    }

    return count;
}

bool audio_buffer_get_latest_frame(const audio_buffer_t* buf, int16_t* out_frame, size_t frame_length) {
    if (!buf || !buf->storage || !out_frame || frame_length == 0) {
        return false;
    }

    if (buf->count < frame_length) {
        return false; /* Not enough samples accumulated yet */
    }

    /* The latest samples end at (buf->write_index - 1).
     * Start index = (buf->write_index - frame_length + buf->capacity) % buf->capacity
     */
    size_t start_idx = (buf->write_index + buf->capacity - (frame_length % buf->capacity)) % buf->capacity;

    for (size_t i = 0; i < frame_length; i++) {
        out_frame[i] = buf->storage[(start_idx + i) % buf->capacity];
    }

    return true;
}

size_t audio_buffer_get_window(const audio_buffer_t* buf, int16_t* out_window, size_t window_length) {
    if (!buf || !buf->storage || !out_window || window_length == 0) {
        return 0;
    }

    size_t to_copy = (buf->count < window_length) ? buf->count : window_length;

    /* Start from oldest sample among the requested window_length:
     * end is buf->write_index. Start is (buf->write_index - to_copy + buf->capacity) % buf->capacity
     */
    size_t start_idx = (buf->write_index + buf->capacity - (to_copy % buf->capacity)) % buf->capacity;

    for (size_t i = 0; i < to_copy; i++) {
        out_window[i] = buf->storage[(start_idx + i) % buf->capacity];
    }

    return to_copy;
}

size_t audio_buffer_get_count(const audio_buffer_t* buf) {
    return buf ? buf->count : 0;
}

bool audio_buffer_is_window_ready(const audio_buffer_t* buf) {
    return buf ? (buf->count >= SWARA_WINDOW_SAMPLES) : false;
}
