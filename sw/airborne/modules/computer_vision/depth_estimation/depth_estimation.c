// Standard includes
#include <stdio.h>
#include <stdlib.h>

// External includes
#include <pthread.h>
#include "udp_socket.h"
#include "mcu_periph/udp.h"

// Project includes
#include "depth_estimation.h"
#include "inference.h"
#include "image_processor.h"
#include "stream_funcs.h"

// Paparazzi includes
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/computer_vision/lib/encoding/rtp.h"
#include "modules/computer_vision/lib/encoding/jpeg.h"

#include "image_processor.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "inference.h"
#include <libyuv.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef DEPTH_ESTIMATION_CAMERA
#define DEPTH_ESTIMATION_CAMERA "front_camera"
#endif

#ifndef DEPTH_RTP_PORT_OUT
#define DEPTH_RTP_PORT_OUT 5100
#endif

#ifndef VIEWVIDEO_QUALITY_FACTOR
#define VIEWVIDEO_QUALITY_FACTOR 50
#endif

// Shared data structure
struct depth_data_t {
    int width;
    int height;
    bool frame_ready;
};

// Video globals
static struct video_listener* listener = NULL;
static pthread_mutex_t mutex;
static struct image_t* current_frame = NULL;
static struct image_t current_frame_copy = {.buf=NULL};
static struct image_t processed_img = {.buf=NULL};

// Model context and shared data
static struct depth_data_t shared_data = {0, 0, false};
static ModelContext* model_ctx = NULL;

// Stream  globals
static struct stream_context_t stream_ctx;
// static struct UdpSocket video_sock;
// static uint16_t rtp_packet_nr = 0;
// static uint32_t rtp_frame_time = 0;
// static struct image_t img_jpeg = {.buf=NULL, .buf_size=0};
// static struct stream_context_t stream_ctx;

// Video callback function (runs in video thread)
static struct image_t* depth_estimation_callback(struct image_t* img, uint8_t camera_id __attribute__((unused))) {
    printf("[Depth Estimation] Video callback\n");
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
    model_ctx = init_inference(MODEL_PATH);
    if (!model_ctx) {
        printf("[Depth Estimation] Failed to initialize model\n");
        return false;
    }

    printf("[Depth Estimation] Model initialized. Input dimensions: %dx%d, Output dimensions: %dx%d\n", 
        model_ctx->input_width, model_ctx->input_height, model_ctx->output_width, model_ctx->output_height);
    
    // // Initialize UDP socket
    // if (udp_socket_create(&video_sock, "127.0.0.1", DEPTH_RTP_PORT_OUT, -1, FALSE)) {
    //     printf("[Depth Estimation] Failed to open UDP socket for video stream\n");
    //     return false;
    // }
    // init_stream(&stream_ctx, "127.0.0.1", DEPTH_RTP_PORT_OUT);

    // Register video callback
    listener = cv_add_to_device(&DEPTH_ESTIMATION_CAMERA, depth_estimation_callback, 10, 0);
    if (listener == NULL) {
        printf("[Depth Estimation] Failed to register video callback\n");
        cleanup_inference(model_ctx);
        return false;
    }

    struct stream_context_t test_ctx;
    init_stream(&test_ctx, "127.0.0.1", 5100);
    test_full_streaming(&test_ctx);
    cleanup_stream(&test_ctx);

    // Initialize the stream context
    init_stream(&stream_ctx, "127.0.0.1", DEPTH_RTP_PORT_OUT);

    printf("[Depth Estimation] Initialized successfully\n");
    return true;
}

void depth_estimation_periodic(void) {
  printf("[Depth Estimation] Periodic function\n");
  
  pthread_mutex_lock(&mutex);
  bool frame_ready = shared_data.frame_ready;
  struct image_t* frame_to_process = current_frame;
  shared_data.frame_ready = false;
  pthread_mutex_unlock(&mutex);

  if (frame_ready && frame_to_process) {
    
    int width = frame_to_process->w;
    int height = frame_to_process->h;
    printf("Frame dimensions: %dx%d\n", width, height);
    
    // Allocate buffers for YUV separation and rotation
    size_t y_size = width * height;
    size_t uv_size = (width * height) / 2;
    
    uint8_t *y_buf = malloc(y_size);
    uint8_t *u_buf = malloc(uv_size);
    uint8_t *v_buf = malloc(uv_size);
    
    uint8_t *rotated_y = malloc(y_size);
    uint8_t *rotated_u = malloc(uv_size);
    uint8_t *rotated_v = malloc(uv_size);
    
    // Separate UYVY into planar YUV
    uint8_t* buf = (uint8_t*)frame_to_process->buf;
    for(int i = 0; i < width * height / 2; i++) {
        // UYVY: [U Y0 V Y1]
        u_buf[i] = buf[i * 4];        // U
        y_buf[i * 2] = buf[i * 4 + 1];    // Y0
        v_buf[i] = buf[i * 4 + 2];        // V
        y_buf[i * 2 + 1] = buf[i * 4 + 3];// Y1
    }
    
    // Rotate the components
    I422Rotate(
        y_buf, width,
        u_buf, width/2,
        v_buf, width/2,
        rotated_y, height,
        rotated_u, height/2,
        rotated_v, height/2,
        width, height,
        kRotate270
    );
    
    // Now convert rotated YUV to RGB
    // Note: After rotation, width and height are swapped
    float* rgb_output = malloc(width * height * 3 * sizeof(float));
    
    for(int i = 0; i < width * height / 2; i++) {
        float y0f = rotated_y[i * 2];
        float y1f = rotated_y[i * 2 + 1];
        float uf = rotated_u[i] - 128;
        float vf = rotated_v[i] - 128;
        
        // YUV to RGB conversion
        int idx0 = i * 6;
        rgb_output[idx0]     = (y0f + 1.402f * vf) / 255.0f;                    // R
        rgb_output[idx0 + 1] = (y0f - 0.344f * uf - 0.714f * vf) / 255.0f;     // G
        rgb_output[idx0 + 2] = (y0f + 1.772f * uf) / 255.0f;                    // B
        
        int idx1 = idx0 + 3;
        rgb_output[idx1]     = (y1f + 1.402f * vf) / 255.0f;                    // R
        rgb_output[idx1 + 1] = (y1f - 0.344f * uf - 0.714f * vf) / 255.0f;     // G
        rgb_output[idx1 + 2] = (y1f + 1.772f * uf) / 255.0f;                    // B
    }

    // Stream the frame with swapped dimensions due to rotation
    stream_frame(&stream_ctx, rgb_output, height, width);

    // Cleanup
    free(y_buf);
    free(u_buf);
    free(v_buf);
    free(rotated_y);
    free(rotated_u);
    free(rotated_v);
    free(rgb_output);
    // Process image 
    float* preprocessed_input = NULL;
    if (!preprocess_image(frame_to_process, &processed_img, &preprocessed_input, model_ctx->input_width, model_ctx->input_height)) {
        printf("[Depth Estimation] Preprocessing failed\n");
        return;
    }
    // Stream the preprocessed input
    // stream_frame(&stream_ctx, preprocessed_input, model_ctx->input_width, model_ctx->input_height);
    // test_color_pattern(&stream_ctx, model_ctx->input_width, model_ctx->input_height);

    // Run inference
    DepthMapResult* depth_result = run_inference(model_ctx, preprocessed_input);
    free(preprocessed_input);
    if (!depth_result) {
        printf("[Depth Estimation] Inference failed\n");
        return;
    }
    // Cleanup
    free_depth_map_result(depth_result);
    return;
  }
}

// Then in depth_estimation_cleanup():
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

    cleanup_stream(&stream_ctx);
}