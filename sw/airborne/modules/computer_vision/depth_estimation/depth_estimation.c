// Standard includes
#include <stdio.h>
#include <stdlib.h>

// External includes
#include <pthread.h>

// Project includes
#include "depth_estimation.h"
#include "inference.h"
#include "image_processor.h"

// Paparazzi includes
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"

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
static struct image_t* current_frame = NULL;
static struct image_t current_frame_copy = {.buf=NULL};
static struct image_t processed_img = {.buf=NULL};

// Video callback function (runs in video thread)
static struct image_t* depth_estimation_callback(struct image_t* img, uint8_t camera_id) {
    pthread_mutex_lock(&mutex);
    
    // Make a proper copy of the image
    if (current_frame_copy.buf == NULL) {
        image_create(&current_frame_copy, img->w, img->h, img->type);
    }
    image_copy(img, &current_frame_copy);
    current_frame = &current_frame_copy;
    
    shared_data.width = img->w;
    shared_data.height = img->h;
    shared_data.frame_ready = true;
    
    pthread_mutex_unlock(&mutex);
    return img;
}

bool depth_estimation_init(void) {
    // Initialize mutex
    pthread_mutex_init(&mutex, NULL);
    
    // Initialize the model
    int input_width, input_height;
    model_ctx = init_inference(MODEL_PATH, &input_width, &input_height);
    if (!model_ctx) {
        printf("[Depth Estimation] Failed to initialize model\n");
        return false;
    }
    printf("[Depth Estimation] Model initialized. Input dimensions: %dx%d\n", 
           input_width, input_height);
    
    // Register video callback
    listener = cv_add_to_device(&DEPTH_ESTIMATION_CAMERA, depth_estimation_callback, 10, 0);
    if (listener == NULL) {
        printf("[Depth Estimation] Failed to register video callback\n");
        cleanup_inference(model_ctx);
        return false;
    }
    
    printf("[Depth Estimation] Initialized successfully\n");
    return true;
}

void depth_estimation_periodic(void) {
  
  pthread_mutex_lock(&mutex);
  bool frame_ready = shared_data.frame_ready;
  struct image_t* frame_to_process = current_frame;
  shared_data.frame_ready = false;
  pthread_mutex_unlock(&mutex);

  if (frame_ready && frame_to_process) {

      // Process image and run inference
      if (!process_image_and_infer(frame_to_process, &processed_img, model_ctx)) {
          printf("[Depth Estimation] Processing or inference failed\n");
          return;
      }
  }
}

void depth_estimation_cleanup(void) {
    pthread_mutex_destroy(&mutex);
    
    if (current_frame_copy.buf) {
        image_free(&current_frame_copy);
    }
    
    if (processed_img.buf) {
        image_free(&processed_img);
    }
    
    if (model_ctx) {
        cleanup_inference(model_ctx);
        model_ctx = NULL;
    }
}