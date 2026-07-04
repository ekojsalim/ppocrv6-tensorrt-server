#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppocrv6_recognizer ppocrv6_recognizer;

int ppocrv6_recognizer_create(const char *config_json,
                              ppocrv6_recognizer **out_handle,
                              char **out_error);

void ppocrv6_recognizer_destroy(ppocrv6_recognizer *handle);

int ppocrv6_recognizer_info_json(ppocrv6_recognizer *handle, char **out_json,
                                 char **out_error);

int ppocrv6_recognizer_recognize_f32(ppocrv6_recognizer *handle,
                                     const float *nchw, int count, int width,
                                     int batch_size, int return_timesteps,
                                     char **out_json, char **out_error);

void ppocrv6_recognizer_free_string(char *value);

#ifdef __cplusplus
}
#endif
