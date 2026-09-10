#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../src/audio/audio_buffer.h"

#define TEST_CAPACITY 16000

static int16_t s_storage[TEST_CAPACITY];
static int16_t s_frame_buf[SWARA_FRAME_LENGTH_SAMPLES];
static int16_t s_window_buf[TEST_CAPACITY];

void test_initialization(void) {
    printf("[TEST] audio_buffer initialization...\n");
    audio_buffer_t buf;
    bool ok = audio_buffer_init(&buf, s_storage, TEST_CAPACITY);
    assert(ok == true);
    assert(audio_buffer_get_count(&buf) == 0);
    assert(audio_buffer_is_window_ready(&buf) == false);
    printf("       -> PASSED\n");
}

void test_write_and_frame_retrieval(void) {
    printf("[TEST] audio_buffer write and latest frame extraction...\n");
    audio_buffer_t buf;
    audio_buffer_init(&buf, s_storage, TEST_CAPACITY);

    int16_t test_chunk[320];
    for (int i = 0; i < 320; i++) {
        test_chunk[i] = (int16_t)(i + 1);
    }

    // Write 320 samples
    size_t written = audio_buffer_write(&buf, test_chunk, 320);
    assert(written == 320);
    assert(audio_buffer_get_count(&buf) == 320);

    // Cannot get 480 samples yet
    bool ok = audio_buffer_get_latest_frame(&buf, s_frame_buf, 480);
    assert(ok == false);

    // Write another 320 samples (values 321 to 640)
    for (int i = 0; i < 320; i++) {
        test_chunk[i] = (int16_t)(321 + i);
    }
    written = audio_buffer_write(&buf, test_chunk, 320);
    assert(written == 320);
    assert(audio_buffer_get_count(&buf) == 640);

    // Now we can get 480 samples
    ok = audio_buffer_get_latest_frame(&buf, s_frame_buf, 480);
    assert(ok == true);

    // The 480 latest samples out of 640 are samples from 161 to 640
    assert(s_frame_buf[0] == 161);
    assert(s_frame_buf[479] == 640);
    printf("       -> PASSED\n");
}

void test_wrap_around(void) {
    printf("[TEST] audio_buffer circular wrap-around...\n");
    audio_buffer_t buf;
    audio_buffer_init(&buf, s_storage, 100); // Small buffer of size 100

    int16_t chunk[80];
    for (int i = 0; i < 80; i++) {
        chunk[i] = (int16_t)i;
    }
    audio_buffer_write(&buf, chunk, 80);
    assert(audio_buffer_get_count(&buf) == 80);

    // Write 50 more (overwriting 30 samples)
    int16_t chunk2[50];
    for (int i = 0; i < 50; i++) {
        chunk2[i] = (int16_t)(100 + i);
    }
    audio_buffer_write(&buf, chunk2, 50);
    assert(audio_buffer_get_count(&buf) == 100);

    // Latest 10 samples should be 140 to 149
    int16_t out[10];
    bool ok = audio_buffer_get_latest_frame(&buf, out, 10);
    assert(ok == true);
    for (int i = 0; i < 10; i++) {
        assert(out[i] == (int16_t)(140 + i));
    }
    printf("       -> PASSED\n");
}

void test_window_retrieval(void) {
    printf("[TEST] audio_buffer window retrieval...\n");
    audio_buffer_t buf;
    audio_buffer_init(&buf, s_storage, TEST_CAPACITY);

    int16_t chunk[500];
    for (int i = 0; i < 500; i++) {
        chunk[i] = (int16_t)(i * 2);
    }
    audio_buffer_write(&buf, chunk, 500);

    size_t copied = audio_buffer_get_window(&buf, s_window_buf, 500);
    assert(copied == 500);
    for (int i = 0; i < 500; i++) {
        assert(s_window_buf[i] == (int16_t)(i * 2));
    }
    printf("       -> PASSED\n");
}

int main(void) {
    printf("=== Running Audio Buffer Unit Tests ===\n");
    test_initialization();
    test_write_and_frame_retrieval();
    test_wrap_around();
    test_window_retrieval();
    printf("=== All Audio Buffer Tests Passed Successfully! ===\n");
    return 0;
}
