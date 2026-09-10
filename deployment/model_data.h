#ifndef SWARA_MODEL_DATA_H_
#define SWARA_MODEL_DATA_H_

#include <cstdint>

// Swara Keyword Spotting / Audio Recognition Model
// Quantized INT8 model data formatted as C byte array for TensorFlow Lite Micro
extern const unsigned char g_swara_model_data[];
extern const unsigned int g_swara_model_data_len;

#endif  // SWARA_MODEL_DATA_H_
