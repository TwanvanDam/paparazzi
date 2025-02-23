#include "depth_estimation.h"
#include "inference.h"
#include <stdio.h>
#include <stdlib.h>

static ModelContext* model_ctx = NULL;
static int input_width = 0;
static int input_height = 0;
static float* test_image = NULL;

bool depth_estimation_init(void) {
  // Initialize the model
  model_ctx = init_inference(MODEL_PATH, &input_width, &input_height);
  if (!model_ctx) {
      printf("[Depth Estimation] Failed to initialize model\n");
      return false;
  }

  printf("[Depth Estimation] Model initialized. Input dimensions: %dx%d\n", 
         input_width, input_height);

  // Create test image (all zeros)
  int input_size = input_width * input_height * 3; // Assuming 3 channels (RGB)
  test_image = (float*)calloc(input_size, sizeof(float));
  if (!test_image) {
      printf("[Depth Estimation] Failed to allocate test image\n");
      cleanup_inference(model_ctx);
      return false;
  }

  return true;
}

void depth_estimation_periodic(void) {
  if (!model_ctx || !test_image) {
      printf("[Depth Estimation] Model or test image not initialized\n");
      return;
  }

  // Run inference on test image
  DepthMapResult* result = run_inference(model_ctx, test_image);
  if (!result) {
      printf("[Depth Estimation] Inference failed\n");
      return;
  }

  // Print depth map statistics
  save_depth_map("test", result);

  // Cleanup
  free_depth_map_result(result);
}

void depth_estimation_cleanup(void) {
  if (test_image) {
      free(test_image);
      test_image = NULL;
  }
  if (model_ctx) {
      cleanup_inference(model_ctx);
      model_ctx = NULL;
  }
}