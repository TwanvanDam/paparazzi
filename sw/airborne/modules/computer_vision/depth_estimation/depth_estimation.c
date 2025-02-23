#include "depth_estimation.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "inference.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"

// Define camera device
#ifndef DEPTH_ESTIMATION_CAMERA
#define DEPTH_ESTIMATION_CAMERA "front_camera"
#endif

// Shared data structure
struct depth_data_t {
    int width;
    int height;
    bool frame_ready;
};

// Global variables
static struct video_listener* listener = NULL;
static pthread_mutex_t mutex;
static struct depth_data_t shared_data = {0, 0, false};
static ModelContext* model_ctx = NULL;
static int input_width = 0;
static int input_height = 0;
static float* test_image = NULL;

// Video callback function (runs in video thread)
static struct image_t* depth_estimation_callback(struct image_t* img) {
    // Store frame info in shared data structure
    pthread_mutex_lock(&mutex);
    shared_data.width = img->w;
    shared_data.height = img->h;
    shared_data.frame_ready = true;
    pthread_mutex_unlock(&mutex);
    
    return img; // Return original image for further processing
}

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
    // Initialize mutex
    pthread_mutex_init(&mutex, NULL);
    
    // Register video callback with FPS and filter arguments
    listener = cv_add_to_device(&DEPTH_ESTIMATION_CAMERA, depth_estimation_callback, 10, 0);
    
    if (listener == NULL) {
        printf("[Depth Estimation] Failed to register video callback\n");
        return false;
    }
    
    printf("[Depth Estimation] Initialized video capture\n");
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

    // Local copy of shared data
    struct depth_data_t local_data;
    
    // Safely copy shared data
    pthread_mutex_lock(&mutex);
    local_data = shared_data;
    shared_data.frame_ready = false; // Reset frame ready flag
    pthread_mutex_unlock(&mutex);

    // Process the data
    if (local_data.frame_ready) {
        printf("[Depth Estimation] New frame received: %dx%d\n", 
               local_data.width, local_data.height);
    }
}

void depth_estimation_cleanup(void) {
  pthread_mutex_destroy(&mutex);
  if (test_image) {
      free(test_image);
      test_image = NULL;
  }
  if (model_ctx) {
      cleanup_inference(model_ctx);
      model_ctx = NULL;
  }
}