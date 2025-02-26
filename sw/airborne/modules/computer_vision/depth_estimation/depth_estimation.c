// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    // printf("[Depth Estimation] Video callback\n");
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

static uint8_t *y_buf = NULL;
static uint8_t *u_buf = NULL;
static uint8_t *v_buf = NULL;
static uint8_t *scaled_y = NULL;
static uint8_t *scaled_u = NULL;
static uint8_t *scaled_v = NULL;
static uint8_t *rotated_y = NULL;
static uint8_t *rotated_u = NULL;
static uint8_t *rotated_v = NULL;
static float *rgb_output = NULL;

// Add initialization function
static bool init_buffers(int max_width, int max_height, int max_target_width, int max_target_height) {
    size_t max_y_size = max_width * max_height;
    size_t max_uv_size = (max_width * max_height) / 2;
    
    y_buf = malloc(max_y_size);
    u_buf = malloc(max_uv_size);
    v_buf = malloc(max_uv_size);
    
    scaled_y = malloc(max_target_width * max_target_height);
    scaled_u = malloc(max_target_width * max_target_height / 2);
    scaled_v = malloc(max_target_width * max_target_height / 2);
    
    rotated_y = malloc(max_target_width * max_target_height);
    rotated_u = malloc(max_target_width * max_target_height / 2);
    rotated_v = malloc(max_target_width * max_target_height / 2);
    
    rgb_output = malloc(max_target_width * max_target_height * 3 * sizeof(float));
    
    return (y_buf && u_buf && v_buf && scaled_y && scaled_u && scaled_v && 
            rotated_y && rotated_u && rotated_v && rgb_output);
}

static float yuv_to_r[256][256]; // y, v -> r
static float yuv_to_g[256][256][256]; // y, u, v -> g
static float yuv_to_b[256][256]; // y, u -> b
static bool tables_initialized = false;

static void init_yuv_tables() {
    for (int y = 0; y < 256; y++) {
        for (int u = 0; u < 256; u++) {
            float uf = u - 128;
            for (int v = 0; v < 256; v++) {
                float vf = v - 128;
                yuv_to_r[y][v] = (y + 1.402f * vf) / 255.0f;
                yuv_to_g[y][u][v] = (y - 0.344f * uf - 0.714f * vf) / 255.0f;
                yuv_to_b[y][u] = (y + 1.772f * uf) / 255.0f;
            }
        }
    }
}

static inline double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// Process 4 UYVY pixels (8 bytes) at once
static inline void convert_uyvy_to_yuv_4pixels(const uint8_t* uyvy, 
    uint8_t* y_out, 
    uint8_t* u_out, 
    uint8_t* v_out) {
u_out[0] = uyvy[0];    // U0
y_out[0] = uyvy[1];    // Y0
v_out[0] = uyvy[2];    // V0
y_out[1] = uyvy[3];    // Y1

u_out[1] = uyvy[4];    // U1
y_out[2] = uyvy[5];    // Y2
v_out[1] = uyvy[6];    // V1
y_out[3] = uyvy[7];    // Y3
}

void depth_estimation_periodic(void) {
    static double t_old_total = 0, t_new_total = 0;
    static int old_count = 0, new_count = 0;
    static int frame_count = 0;
    
    printf("[Depth Estimation] Periodic function\n");
    
    pthread_mutex_lock(&mutex);
    bool frame_ready = shared_data.frame_ready;
    struct image_t* frame_to_process = current_frame;
    shared_data.frame_ready = false;
    pthread_mutex_unlock(&mutex);

    if (!frame_ready || !frame_to_process) {
        return;
    }

    int width = frame_to_process->w;
    int height = frame_to_process->h;
    int target_width = model_ctx->input_width;
    int target_height = model_ctx->input_height;
    
    if (y_buf == NULL) {
        if (!init_buffers(width, height, target_width, target_height)) {
            printf("Buffer allocation failed\n");
            return;
        }
    }

    if (!tables_initialized) {
        init_yuv_tables();
        tables_initialized = true;
    }

    // First measure original 2-pixel version
    double t1 = get_time_ms();
    
    uint8_t* buf = (uint8_t*)frame_to_process->buf;
    int pixels = width * height;
    int pixel_idx;
    
    for(pixel_idx = 0; pixel_idx < pixels - 3; pixel_idx += 4) {
        convert_uyvy_to_yuv_4pixels(
            &buf[pixel_idx * 2],
            &y_buf[pixel_idx],
            &u_buf[pixel_idx/2],
            &v_buf[pixel_idx/2]
        );
    }
    
    for(; pixel_idx < pixels; pixel_idx += 2) {
        int idx = pixel_idx * 2;
        u_buf[pixel_idx/2] = buf[idx];
        y_buf[pixel_idx] = buf[idx + 1];
        v_buf[pixel_idx/2] = buf[idx + 2];
        y_buf[pixel_idx + 1] = buf[idx + 3];
    }

    I422Scale(
        y_buf, width,
        u_buf, width/2,
        v_buf, width/2,
        width, height,
        scaled_y, target_width,
        scaled_u, target_width/2,
        scaled_v, target_width/2,
        target_width, target_height,
        kFilterBilinear
    );

    I422Rotate(
        scaled_y, target_width,
        scaled_u, target_width/2,
        scaled_v, target_width/2,
        rotated_y, target_height,
        rotated_u, target_height/2,
        rotated_v, target_height/2,
        target_width, target_height,
        kRotate270
    );

    // Original 2-pixel version
    int total_pixels = target_width * target_height / 2;
    int i;
    for(i = 0; i < total_pixels - 1; i += 2) {
        uint8_t y0 = rotated_y[i * 2];
        uint8_t y1 = rotated_y[i * 2 + 1];
        uint8_t u = rotated_u[i];
        uint8_t v = rotated_v[i];
        
        int idx0 = i * 6;
        rgb_output[idx0]     = yuv_to_r[y0][v];
        rgb_output[idx0 + 1] = yuv_to_g[y0][u][v];
        rgb_output[idx0 + 2] = yuv_to_b[y0][u];
        
        int idx1 = idx0 + 3;
        rgb_output[idx1]     = yuv_to_r[y1][v];
        rgb_output[idx1 + 1] = yuv_to_g[y1][u][v];
        rgb_output[idx1 + 2] = yuv_to_b[y1][u];

        y0 = rotated_y[i * 2 + 2];
        y1 = rotated_y[i * 2 + 3];
        u = rotated_u[i + 1];
        v = rotated_v[i + 1];
        
        idx0 = (i + 1) * 6;
        rgb_output[idx0]     = yuv_to_r[y0][v];
        rgb_output[idx0 + 1] = yuv_to_g[y0][u][v];
        rgb_output[idx0 + 2] = yuv_to_b[y0][u];
        
        idx1 = idx0 + 3;
        rgb_output[idx1]     = yuv_to_r[y1][v];
        rgb_output[idx1 + 1] = yuv_to_g[y1][u][v];
        rgb_output[idx1 + 2] = yuv_to_b[y1][u];
    }
    
    double old_time = get_time_ms() - t1;
    t_old_total += old_time;
    old_count++;

    // Now measure optimized 4-pixel version
    t1 = get_time_ms();
    
    // Optimized 4-pixel version with remainder handling
    for(i = 0; i < total_pixels - 1; i += 2) {
        // First pair of pixels
        uint8_t y0 = rotated_y[i * 2];
        uint8_t y1 = rotated_y[i * 2 + 1];
        uint8_t u = rotated_u[i];
        uint8_t v = rotated_v[i];
        
        int idx0 = i * 6;
        rgb_output[idx0]     = yuv_to_r[y0][v];
        rgb_output[idx0 + 1] = yuv_to_g[y0][u][v];
        rgb_output[idx0 + 2] = yuv_to_b[y0][u];
        
        int idx1 = idx0 + 3;
        rgb_output[idx1]     = yuv_to_r[y1][v];
        rgb_output[idx1 + 1] = yuv_to_g[y1][u][v];
        rgb_output[idx1 + 2] = yuv_to_b[y1][u];

        // Second pair of pixels
        y0 = rotated_y[i * 2 + 2];
        y1 = rotated_y[i * 2 + 3];
        u = rotated_u[i + 1];
        v = rotated_v[i + 1];
        
        idx0 = (i + 1) * 6;
        rgb_output[idx0]     = yuv_to_r[y0][v];
        rgb_output[idx0 + 1] = yuv_to_g[y0][u][v];
        rgb_output[idx0 + 2] = yuv_to_b[y0][u];
        
        idx1 = idx0 + 3;
        rgb_output[idx1]     = yuv_to_r[y1][v];
        rgb_output[idx1 + 1] = yuv_to_g[y1][u][v];
        rgb_output[idx1 + 2] = yuv_to_b[y1][u];
    }

    // Handle remaining pixels
    for(; i < total_pixels; i++) {
        uint8_t y0 = rotated_y[i * 2];
        uint8_t y1 = rotated_y[i * 2 + 1];
        uint8_t u = rotated_u[i];
        uint8_t v = rotated_v[i];
        
        int idx0 = i * 6;
        rgb_output[idx0]     = yuv_to_r[y0][v];
        rgb_output[idx0 + 1] = yuv_to_g[y0][u][v];
        rgb_output[idx0 + 2] = yuv_to_b[y0][u];
        
        int idx1 = idx0 + 3;
        rgb_output[idx1]     = yuv_to_r[y1][v];
        rgb_output[idx1 + 1] = yuv_to_g[y1][u][v];
        rgb_output[idx1 + 2] = yuv_to_b[y1][u];
    }

    double new_time = get_time_ms() - t1;
    t_new_total += new_time;
    new_count++;

    // Print stats every 30 frames
    if (++frame_count % 30 == 0) {
        double old_avg = t_old_total / old_count;
        double new_avg = t_new_total / new_count;
        
        printf("\nPerformance Comparison after %d frames:\n", frame_count);
        printf("Original version (2-pixel): %.2f ms (%d samples)\n", old_avg, old_count);
        printf("Optimized version (4-pixel): %.2f ms (%d samples)\n", new_avg, new_count);
        printf("Speedup: %.2fx\n\n", old_avg / new_avg);
    }

    stream_frame(&stream_ctx, rgb_output, target_height, target_width);

    // Process image for inference
    float* preprocessed_input = NULL;
    if (!preprocess_image(frame_to_process, &processed_img, &preprocessed_input, 
                         model_ctx->input_width, model_ctx->input_height)) {
        printf("[Depth Estimation] Preprocessing failed\n");
        return;
    }

    DepthMapResult* depth_result = run_inference(model_ctx, preprocessed_input);
    free(preprocessed_input);
    if (!depth_result) {
        printf("[Depth Estimation] Inference failed\n");
        return;
    }
    free_depth_map_result(depth_result);
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

    cleanup_stream(&stream_ctx);
}