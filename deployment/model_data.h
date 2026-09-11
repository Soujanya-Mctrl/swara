#ifndef SWARA_MODEL_DATA_H_
#define SWARA_MODEL_DATA_H_

#include <cstdint>

// Swara Keyword Spotting / Audio Recognition Model
// [INFRASTRUCTURE PLACEHOLDER]
// NOTE: Real Swara INT8 model has not yet been trained or quantized.
// The actual model bytes will be generated via training/export_model_header.py
// once the real Google Drive training dataset is imported and trained.
extern const unsigned char g_swara_model_data[];
extern const unsigned int g_swara_model_data_len;

#endif  // SWARA_MODEL_DATA_H_
