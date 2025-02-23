#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "modules/computer_vision/lib/vision/image.h"
#include "inference.h"
#include <stdbool.h>

// Basic image processing
bool process_image(struct image_t *input, struct image_t *output);

// Process image and run inference
bool process_image_and_infer(struct image_t *input, struct image_t *output, ModelContext* model_ctx);

#endif