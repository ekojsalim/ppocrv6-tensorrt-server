#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppocrv6_full_page ppocrv6_full_page;

int ppocrv6_full_page_create(const char *config_json,
                             ppocrv6_full_page **out_handle,
                             char **out_error);

void ppocrv6_full_page_destroy(ppocrv6_full_page *handle);

int ppocrv6_full_page_info_json(ppocrv6_full_page *handle, char **out_json,
                                char **out_error);

int ppocrv6_full_page_recognize(ppocrv6_full_page *handle,
                                const unsigned char *image, int image_height,
                                int image_width, int image_stride,
                                int source_color_order,
                                const float *detector_nchw,
                                int detector_height, int detector_width,
                                char **out_json, char **out_error);

int ppocrv6_full_page_recognize_image(ppocrv6_full_page *handle,
                                      const unsigned char *image,
                                      int image_height, int image_width,
                                      int image_stride,
                                      int source_color_order,
                                      char **out_json, char **out_error);

void ppocrv6_full_page_free_string(char *value);

#ifdef __cplusplus
}
#endif
