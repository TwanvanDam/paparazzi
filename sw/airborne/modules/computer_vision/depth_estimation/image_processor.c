#include "image_processor.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "inference.h"
#include <libyuv.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void rgb_to_float_array(uint8_t* rgb_data, float* float_data, int width, int height) {
    const float scale = 1.0f/255.0f;
    int total_pixels = width * height * 3;
    
    // #pragma omp simd
    for (int i = 0; i < total_pixels; i++) {
        float_data[i] = rgb_data[i] * scale;
    }
}

bool process_image(struct image_t *input, struct image_t *output, int target_width, int target_height) {
  int result;  
  
  // printf("\nProcessing - input type:%d size:%dx%d\n", input->type, input->w, input->h);

  if (input->type != IMAGE_YUV422) {
      printf("Failed: Input image is not YUV422\n");
      return false;
  }

  if (!input->buf) {
      printf("Failed: Input buffer is NULL\n");
      return false;
  }

  // Static intermediate buffers
  static uint8_t *y_buf = NULL;
  static uint8_t *u_buf = NULL;
  static uint8_t *v_buf = NULL;
  static uint8_t *rotated_y = NULL;
  static uint8_t *rotated_u = NULL;
  static uint8_t *rotated_v = NULL;
  static uint8_t *scaled_y = NULL;
  static uint8_t *scaled_u = NULL;
  static uint8_t *scaled_v = NULL;

  // Calculate buffer sizes
  size_t y_size = input->w * input->h;
  size_t uv_size = (input->w * input->h) / 2;
  size_t rgb_size = target_height * target_width * 3;

  // Allocate output buffer
  if (output->buf == NULL || output->w != target_height || output->h != target_width) {
      free(output->buf);
      output->buf = malloc(rgb_size);
      output->w = target_height;
      output->h = target_width;
      output->buf_size = rgb_size;
      output->type = IMAGE_YUV422;
  }

  if (!output->buf) {
      printf("Failed: Output buffer allocation failed\n");
      return false;
  }

  // One-time allocation of intermediate buffers
  if (!y_buf) y_buf = malloc(y_size);
  if (!u_buf) u_buf = malloc(uv_size);
  if (!v_buf) v_buf = malloc(uv_size);
  if (!rotated_y) rotated_y = malloc(y_size);
  if (!rotated_u) rotated_u = malloc(uv_size);
  if (!rotated_v) rotated_v = malloc(uv_size);
  if (!scaled_y) scaled_y = malloc(target_height * target_width);
  if (!scaled_u) scaled_u = malloc(target_height * target_width / 2);
  if (!scaled_v) scaled_v = malloc(target_height * target_width / 2);

  if (!y_buf || !u_buf || !v_buf || !rotated_y || !rotated_u || !rotated_v || 
      !scaled_y || !scaled_u || !scaled_v) {
      printf("Failed: Intermediate buffer allocation failed\n");
      return false;
  }

    // YUY2 to I422 conversion
    uint8_t* input_buf = (uint8_t*)input->buf;
    result = YUY2ToI422(
        input_buf, input->w * 2,
        y_buf, input->w,
        u_buf, input->w/2,
        v_buf, input->w/2,
        input->w, input->h
    );
    if (result != 0) return false;

    // Rotation
    result = I422Rotate(
        y_buf, input->w,
        u_buf, input->w/2,
        v_buf, input->w/2,
        rotated_y, input->h,
        rotated_u, input->h/2,
        rotated_v, input->h/2,
        input->w, input->h,
        kRotate270
    );
    if (result != 0) return false;

    // Scale to model input size (target_heightxtarget_width)
    result = I422Scale(
        rotated_y, input->h,
        rotated_u, input->h/2,
        rotated_v, input->h/2,
        input->h, input->w,
        scaled_y, target_height,
        scaled_u, target_height/2,
        scaled_v, target_height/2,
        target_height, target_width,
        kFilterBilinear
    );
    if (result != 0) return false;

    // Convert to RGB24
    result = I422ToRGB24(
        scaled_y, target_height,
        scaled_u, target_height/2,
        scaled_v, target_height/2,
        output->buf, target_height * 3,
        target_height, target_width
    );
    if (result != 0) return false;

    return true;
}

// Combine processing steps into a single function
bool preprocess_image(struct image_t *input, struct image_t *rgb_output, float **normalized_output, 
                        int target_width, int target_height) {
    // First process image to RGB
    if (!process_image(input, rgb_output, target_width, target_height)) {
        return false;
    }

    // Allocate normalized float array
    *normalized_output = malloc(target_height * target_width * 3 * sizeof(float));
    if (!*normalized_output) {
        return false;
    }

    // Convert RGB to normalized float array
    rgb_to_float_array((uint8_t*)rgb_output->buf, *normalized_output, target_height, target_width);

    return true;
}

// Combine processing with inference
bool process_image_and_infer(struct image_t *input, struct image_t *output, ModelContext* model_ctx) {
    if (!model_ctx) {
        return false;
    }

    float* preprocessed_input = NULL;
    if (!preprocess_image(input, output, &preprocessed_input, model_ctx->input_width, model_ctx->input_height)) {
        return false;
    }

    // Run inference
    DepthMapResult* depth_result = run_inference(model_ctx, preprocessed_input);
    free(preprocessed_input);

    if (!depth_result) {
        return false;
    }
    
    // Cleanup
    free_depth_map_result(depth_result);

    return true;
}