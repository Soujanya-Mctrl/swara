cmake_minimum_required(VERSION 3.15)
project(swara C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags: optimize for size and embedded compatibility
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

# Include directories
include_directories(
    src/audio
    src/features
)

# Core Audio & Feature Extraction Library
add_library(swara_core STATIC
    src/audio/audio_buffer.c
    src/audio/vad.c
    src/audio/wav_reader.c
    src/features/fft.c
    src/features/mfcc.c
)

# Math library linking for Unix / MinGW
if(NOT MSVC)
    target_link_libraries(swara_core m)
endif()

# Tests
enable_testing()

add_executable(test_audio_buffer tests/test_audio_buffer.c)
target_link_libraries(test_audio_buffer swara_core)
add_test(NAME test_audio_buffer COMMAND test_audio_buffer)

add_executable(test_vad tests/test_vad.c)
target_link_libraries(test_vad swara_core)
if(NOT MSVC)
    target_link_libraries(test_vad m)
endif()
add_test(NAME test_vad COMMAND test_vad)

add_executable(test_wav tests/test_wav.c)
target_link_libraries(test_wav swara_core)
add_test(NAME test_wav COMMAND test_wav)

add_executable(test_fft tests/test_fft.c)
target_link_libraries(test_fft swara_core)
if(NOT MSVC)
    target_link_libraries(test_fft m)
endif()
add_test(NAME test_fft COMMAND test_fft)

add_executable(test_mfcc tests/test_mfcc.c)
target_link_libraries(test_mfcc swara_core)
if(NOT MSVC)
    target_link_libraries(test_mfcc m)
endif()
add_test(NAME test_mfcc COMMAND test_mfcc)

add_executable(test_pipeline tests/test_pipeline.c)
target_link_libraries(test_pipeline swara_core)
if(NOT MSVC)
    target_link_libraries(test_pipeline m)
endif()
add_test(NAME test_pipeline COMMAND test_pipeline)

add_executable(test_wav_mfcc tests/test_wav_mfcc.c)
target_link_libraries(test_wav_mfcc swara_core)
if(NOT MSVC)
    target_link_libraries(test_wav_mfcc m)
endif()
add_test(NAME test_wav_mfcc COMMAND test_wav_mfcc)

add_executable(benchmark_mfcc tests/benchmark_mfcc.c)
target_link_libraries(benchmark_mfcc swara_core)
if(NOT MSVC)
    target_link_libraries(benchmark_mfcc m)
endif()

add_executable(benchmark_pipeline tests/benchmark_pipeline.c)
target_link_libraries(benchmark_pipeline swara_core)
if(NOT MSVC)
    target_link_libraries(benchmark_pipeline m)
endif()

