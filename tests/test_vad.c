#include <stdio.h>
#include <assert.h>
#include "../src/audio/vad.h"

void test_vad_silence(void) {
    printf("[TEST] vad silence detection...\n");
    vad_config_t vad;
    vad_init(&vad, 300, 3);

    int16_t silence[480] = {0};
    uint32_t rms = vad_calculate_rms(silence, 480);
    assert(rms == 0);

    bool speech = vad_process_frame(&vad, silence, 480);
    assert(speech == false);
    printf("       -> PASSED\n");
}

void test_vad_speech_and_hangover(void) {
    printf("[TEST] vad speech detection and hangover smoothing...\n");
    vad_config_t vad;
    vad_init(&vad, 300, 2); // 2 frames hangover

    int16_t loud[480];
    for (int i = 0; i < 480; i++) {
        loud[i] = (int16_t)(i % 2 == 0 ? 1500 : -1500); // RMS ~1500
    }

    uint32_t rms = vad_calculate_rms(loud, 480);
    assert(rms > 1000);

    // Frame 1: Loud speech -> detected
    bool speech = vad_process_frame(&vad, loud, 480);
    assert(speech == true);

    // Frame 2: Silence -> still true due to hangover 1
    int16_t silence[480] = {0};
    speech = vad_process_frame(&vad, silence, 480);
    assert(speech == true);

    // Frame 3: Silence -> still true due to hangover 2
    speech = vad_process_frame(&vad, silence, 480);
    assert(speech == true);

    // Frame 4: Silence -> hangover expired, now false
    speech = vad_process_frame(&vad, silence, 480);
    assert(speech == false);

    printf("       -> PASSED\n");
}

int main(void) {
    printf("=== Running VAD Unit Tests ===\n");
    test_vad_silence();
    test_vad_speech_and_hangover();
    printf("=== All VAD Tests Passed Successfully! ===\n");
    return 0;
}
