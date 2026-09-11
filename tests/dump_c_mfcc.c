#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "../src/audio/wav_reader.h"
#include "../src/features/mfcc.h"

int main(void) {
    const char* wav_path = "tests/data/test_16k_1s.wav";
    const char* out_bin  = "tests/data/test_c_mfcc_reference.bin";

    int16_t samples[16000];
    size_t read_count = 0;
    wav_status_t status = wav_read_samples(wav_path, samples, 16000, &read_count);
    if (status != SWARA_WAV_OK || read_count != 16000) {
        fprintf(stderr, "Failed to read %s\n", wav_path);
        return 1;
    }

    mfcc_config_t cfg;
    mfcc_init(&cfg);

    float features[49][10];
    mfcc_compute_window(&cfg, samples, features);

    FILE* fp = fopen(out_bin, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", out_bin);
        return 1;
    }

    fwrite(features, sizeof(float), 49 * 10, fp);
    fclose(fp);

    printf("Dumped 49x10 C MFCC reference features to %s\n", out_bin);
    return 0;
}
