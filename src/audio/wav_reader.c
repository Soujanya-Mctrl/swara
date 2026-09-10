#include "wav_reader.h"
#include <stdio.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    char chunk_id[4];        /* "RIFF" */
    uint32_t chunk_size;     /* 36 + subchunk2_size */
    char format[4];          /* "WAVE" */
    char subchunk1_id[4];    /* "fmt " */
    uint32_t subchunk1_size; /* 16 for PCM */
    uint16_t audio_format;   /* 1 for PCM */
    uint16_t num_channels;   /* 1 for Mono */
    uint32_t sample_rate;    /* 16000 */
    uint32_t byte_rate;      /* sample_rate * num_channels * bits_per_sample / 8 */
    uint16_t block_align;    /* num_channels * bits_per_sample / 8 */
    uint16_t bits_per_sample;/* 16 */
} wav_header_fmt_t;

typedef struct {
    char subchunk2_id[4];    /* "data" */
    uint32_t subchunk2_size; /* num_samples * num_channels * bits_per_sample / 8 */
} wav_header_data_t;
#pragma pack(pop)

wav_status_t wav_read_header(const char* filepath, wav_info_t* info) {
    if (!filepath || !info) return SWARA_WAV_ERR_READ_FAILED;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return SWARA_WAV_ERR_FILE_NOT_FOUND;

    wav_header_fmt_t fmt;
    if (fread(&fmt, sizeof(wav_header_fmt_t), 1, fp) != 1) {
        fclose(fp);
        return SWARA_WAV_ERR_INVALID_HEADER;
    }

    if (memcmp(fmt.chunk_id, "RIFF", 4) != 0 || memcmp(fmt.format, "WAVE", 4) != 0) {
        fclose(fp);
        return SWARA_WAV_ERR_INVALID_HEADER;
    }

    if (fmt.audio_format != 1) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_FORMAT;
    }

    if (fmt.num_channels != 1) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_CHANNELS;
    }

    if (fmt.sample_rate != 16000) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_RATE;
    }

    if (fmt.bits_per_sample != 16) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_BITS;
    }

    /* Find "data" chunk (handle any metadata chunks like LIST or JUNK) */
    char chunk_tag[4];
    uint32_t chunk_len = 0;
    bool found_data = false;

    while (fread(chunk_tag, 4, 1, fp) == 1 && fread(&chunk_len, 4, 1, fp) == 1) {
        if (memcmp(chunk_tag, "data", 4) == 0) {
            found_data = true;
            break;
        }
        /* Skip non-data chunk */
        if (fseek(fp, (long)chunk_len, SEEK_CUR) != 0) {
            break;
        }
    }

    fclose(fp);

    if (!found_data) {
        return SWARA_WAV_ERR_INVALID_HEADER;
    }

    info->num_channels = fmt.num_channels;
    info->sample_rate = fmt.sample_rate;
    info->bits_per_sample = fmt.bits_per_sample;
    info->data_bytes = chunk_len;
    info->num_samples = chunk_len / sizeof(int16_t);

    return SWARA_WAV_OK;
}

wav_status_t wav_read_samples(
    const char* filepath,
    int16_t* out_samples,
    size_t max_samples,
    size_t* out_read_samples
) {
    if (!filepath || !out_samples || max_samples == 0 || !out_read_samples) {
        return SWARA_WAV_ERR_READ_FAILED;
    }

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return SWARA_WAV_ERR_FILE_NOT_FOUND;

    wav_header_fmt_t fmt;
    if (fread(&fmt, sizeof(wav_header_fmt_t), 1, fp) != 1) {
        fclose(fp);
        return SWARA_WAV_ERR_INVALID_HEADER;
    }

    if (memcmp(fmt.chunk_id, "RIFF", 4) != 0 || memcmp(fmt.format, "WAVE", 4) != 0) {
        fclose(fp);
        return SWARA_WAV_ERR_INVALID_HEADER;
    }
    if (fmt.audio_format != 1) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_FORMAT;
    }
    if (fmt.num_channels != 1) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_CHANNELS;
    }
    if (fmt.sample_rate != 16000) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_RATE;
    }
    if (fmt.bits_per_sample != 16) {
        fclose(fp);
        return SWARA_WAV_ERR_UNSUPPORTED_BITS;
    }

    /* Find data chunk */
    char chunk_tag[4];
    uint32_t data_bytes = 0;
    bool found_data = false;

    while (fread(chunk_tag, 4, 1, fp) == 1 && fread(&data_bytes, 4, 1, fp) == 1) {
        if (memcmp(chunk_tag, "data", 4) == 0) {
            found_data = true;
            break;
        }
        if (fseek(fp, (long)data_bytes, SEEK_CUR) != 0) break;
    }

    if (!found_data) {
        fclose(fp);
        return SWARA_WAV_ERR_INVALID_HEADER;
    }

    size_t samples_in_file = data_bytes / sizeof(int16_t);
    size_t samples_to_read = (samples_in_file < max_samples) ? samples_in_file : max_samples;

    size_t read_count = fread(out_samples, sizeof(int16_t), samples_to_read, fp);
    fclose(fp);

    *out_read_samples = read_count;
    return SWARA_WAV_OK;
}

bool wav_write_samples(
    const char* filepath,
    const int16_t* samples,
    size_t count,
    uint32_t sample_rate
) {
    if (!filepath || !samples) return false;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return false;

    uint32_t data_bytes = (uint32_t)(count * sizeof(int16_t));
    uint32_t byte_rate = sample_rate * 1 * sizeof(int16_t);

    wav_header_fmt_t fmt;
    memcpy(fmt.chunk_id, "RIFF", 4);
    fmt.chunk_size = 36 + data_bytes;
    memcpy(fmt.format, "WAVE", 4);
    memcpy(fmt.subchunk1_id, "fmt ", 4);
    fmt.subchunk1_size = 16;
    fmt.audio_format = 1; /* PCM */
    fmt.num_channels = 1; /* Mono */
    fmt.sample_rate = sample_rate;
    fmt.byte_rate = byte_rate;
    fmt.block_align = (uint16_t)sizeof(int16_t);
    fmt.bits_per_sample = 16;

    wav_header_data_t data_hdr;
    memcpy(data_hdr.subchunk2_id, "data", 4);
    data_hdr.subchunk2_size = data_bytes;

    if (fwrite(&fmt, sizeof(fmt), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    if (fwrite(&data_hdr, sizeof(data_hdr), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    if (fwrite(samples, sizeof(int16_t), count, fp) != count) {
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}
