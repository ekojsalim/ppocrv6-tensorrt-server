#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppocrv6_detector ppocrv6_detector;

int ppocrv6_detector_create(const char *config_json,
                            ppocrv6_detector **out_handle, char **out_error);

void ppocrv6_detector_destroy(ppocrv6_detector *handle);

int ppocrv6_detector_info_json(ppocrv6_detector *handle, char **out_json,
                               char **out_error);

int ppocrv6_detector_detect_f32(ppocrv6_detector *handle, const float *nchw,
                                int batch, int height, int width,
                                float *out_prob, size_t out_prob_count,
                                char **out_json, char **out_error);

void ppocrv6_detector_free_string(char *value);

#ifdef __cplusplus
}
#endif
