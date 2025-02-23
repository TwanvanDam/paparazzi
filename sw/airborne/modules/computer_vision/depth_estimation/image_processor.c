#include "image_processor.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "inference.h"
#include <libyuv.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Implementation of rgb_to_float_array
void rgb_to_float_array(uint8_t* rgb_data, float* float_data, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int rgb_idx = (i * width + j) * 3;
            int float_idx = (i * width + j) * 3;
            
            float_data[float_idx] = rgb_data[rgb_idx] / 255.0f;     // R
            float_data[float_idx + 1] = rgb_data[rgb_idx + 1] / 255.0f; // G
            float_data[float_idx + 2] = rgb_data[rgb_idx + 2] / 255.0f; // B
        }
    }
}


bool process_image(struct image_t *input, struct image_t *output) {
  struct timespec step_start, step_end;
  long step_time_ns;
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
  size_t rgb_size = 96 * 128 * 3;

  // Allocate output buffer
  if (output->buf == NULL || output->w != 96 || output->h != 128) {
      free(output->buf);
      output->buf = malloc(rgb_size);
      output->w = 96;
      output->h = 128;
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
  if (!scaled_y) scaled_y = malloc(96 * 128);
  if (!scaled_u) scaled_u = malloc(96 * 128 / 2);
  if (!scaled_v) scaled_v = malloc(96 * 128 / 2);

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

    // Scale to model input size (96x128)
    result = I422Scale(
        rotated_y, input->h,
        rotated_u, input->h/2,
        rotated_v, input->h/2,
        input->h, input->w,
        scaled_y, 96,
        scaled_u, 96/2,
        scaled_v, 96/2,
        96, 128,
        kFilterBilinear
    );
    if (result != 0) return false;

    // Convert to RGB24
    result = I422ToRGB24(
        scaled_y, 96,
        scaled_u, 96/2,
        scaled_v, 96/2,
        output->buf, 96 * 3,
        96, 128
    );
    if (result != 0) return false;

    // // Convert RGB to float array for inference
    // static float* inference_input = NULL;
    // if (!inference_input) {
    //     inference_input = malloc(96 * 128 * 3 * sizeof(float));
    //     if (!inference_input) return false;
    // }

    // // // Time the inference preparation and execution
    // // clock_gettime(CLOCK_MONOTONIC, &step_start);
    
    // // // rgb_to_float_array((uint8_t*)output->buf, inference_input, 96, 128);

    // // // // Run inference
    // // // DepthMapResult* depth_result = run_inference(model_ctx, inference_input);
    
    // clock_gettime(CLOCK_MONOTONIC, &step_end);
    // step_time_ns = (step_end.tv_sec - step_start.tv_sec) * 1000000000L + 
    //                (step_end.tv_nsec - step_start.tv_nsec);
    
    // if (!depth_result) {
    //     printf("Inference failed\n");
    //     return false;
    // }

    // printf("Inference took %.3f ms\n", step_time_ns / 1000000.0);
    
    // // Print depth map statistics
    // save_depth_map("depth_stats", depth_result);
    
    // // Cleanup
    // free_depth_map_result(depth_result);

    return true;
}

bool process_image_and_infer(struct image_t *input, struct image_t *output, ModelContext* model_ctx) {
  if (!process_image(input, output)) {
      return false;
  }

  if (!model_ctx) {
      return false;
  }

  // Convert RGB to float array for inference
  float* inference_input = malloc(96 * 128 * 3 * sizeof(float));
  if (!inference_input) {
      return false;
  }

  // Convert RGB data to normalized float array
  uint8_t* rgb_data = (uint8_t*)output->buf;
  for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 96; j++) {
          int rgb_idx = (i * 96 + j) * 3;
          int float_idx = (i * 96 + j) * 3;
          inference_input[float_idx] = rgb_data[rgb_idx] / 255.0f;     // R
          inference_input[float_idx + 1] = rgb_data[rgb_idx + 1] / 255.0f; // G
          inference_input[float_idx + 2] = rgb_data[rgb_idx + 2] / 255.0f; // B
      }
  }

  // Run inference
  DepthMapResult* depth_result = run_inference(model_ctx, inference_input);
  free(inference_input);

  if (!depth_result) {
      return false;
  }

  // Print depth map statistics
  save_depth_map("depth_stats", depth_result);
  
  // Cleanup
  free_depth_map_result(depth_result);

  return true;
}
