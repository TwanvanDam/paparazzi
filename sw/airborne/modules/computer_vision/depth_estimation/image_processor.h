#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "modules/computer_vision/lib/vision/image.h"
#include "inference.h"
#include <stdbool.h>

// Basic image processing
bool process_image(struct image_t *input, struct image_t *output, int target_width, int target_height);

// Preprocess image for inference
bool preprocess_image(struct image_t *input, struct image_t *output, float** preprocessed_input, int target_width, int target_height);

// Process image and run inference
bool process_image_and_infer(struct image_t *input, struct image_t *output, ModelContext* model_ctx);

void rgb_to_float_array(uint8_t* rgb_data, float* float_data, int width, int height);
#endif