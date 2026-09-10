#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../src/audio/wav_reader.h"

#define TEST_WAV_FILE "test_sample_16k.wav"

void test_wav_write_and_read(void) {
    printf("[TEST] wav write and read compliant 16kHz mono...\n");

    int16_t original[16000];
    for (int i = 0; i < 16000; i++) {
        original[i] = (int16_t)(i % 1000);
    }

    bool write_ok = wav_write_samples(TEST_WAV_FILE, original, 16000, 16000);
    assert(write_ok == true);

    wav_info_t info;
    wav_status_t status = wav_read_header(TEST_WAV_FILE, &info);
    assert(status == SWARA_WAV_OK);
    assert(info.sample_rate == 16000);
    assert(info.num_channels == 1);
    assert(info.bits_per_sample == 16);
    assert(info.num_samples == 16000);

    int16_t read_back[16000];
    size_t samples_read = 0;
    status = wav_read_samples(TEST_WAV_FILE, read_back, 16000, &samples_read);
    assert(status == SWARA_WAV_OK);
    assert(samples_read == 16000);

    for (int i = 0; i < 16000; i++) {
        assert(read_back[i] == original[i]);
    }
    printf("       -> PASSED\n");
}

void test_wav_rejection(void) {
    printf("[TEST] wav reject non-16kHz rate...\n");
    const char* bad_rate_file = "test_bad_rate_44k.wav";
    int16_t dummy[100] = {0};

    // Write a 44,100 Hz file
    wav_write_samples(bad_rate_file, dummy, 100, 44100);

    wav_info_t info;
    wav_status_t status = wav_read_header(bad_rate_file, &info);
    assert(status == SWARA_WAV_ERR_UNSUPPORTED_RATE);
    printf("       -> PASSED (Correctly rejected 44.1 kHz)\n");
}

int main(void) {
    printf("=== Running WAV Reader Unit Tests ===\n");
    test_wav_write_and_read();
    test_wav_rejection();
    printf("=== All WAV Tests Passed Successfully! ===\n");
    return 0;
}
