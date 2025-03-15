// obstacle_detection.h
#ifndef OBSTACLE_DETECTION_H
#define OBSTACLE_DETECTION_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include "modules/computer_vision/lib/vision/image.h"
#include "video_stream.h"

#define MODEL_TYPE_OBSTACLE 0
#define MODEL_TYPE_BORDER 1

// Camera dimensions
#define FRONT_CAMERA_WIDTH 240
#define FRONT_CAMERA_HEIGHT 520
#define BOTTOM_CAMERA_WIDTH 240
#define BOTTOM_CAMERA_HEIGHT 240

#define DEBUG_LEVEL_NONE 0
#define DEBUG_LEVEL_BASIC 1
#define DEBUG_LEVEL_VERBOSE 2

extern int debug_level;  // Global debug level

// Configuration and state
struct obstacle_detection_t {
    bool front_enabled;
    bool bottom_enabled;
    bool stream_enabled;
};

// Shared data for each camera
struct camera_data_t {
    // Existing members
    struct image_t *frame;
    bool frame_ready;
    float *rgb_buffer;
    size_t rgb_buffer_size;
    pthread_mutex_t frame_mutex;
    struct stream_context_t stream_ctx;
    bool initialized;
    
    // Thread tracking
    pthread_t last_callback_thread;
    pthread_t last_inference_thread;
    uint32_t callback_count;
    uint32_t inference_count;
    
    // Memory tracking
    void* last_frame_addr;
    void* last_rgb_buffer_addr;
    size_t last_frame_size;
    uint32_t rgb_buffer_checksum;
    uint32_t error_count;
    size_t rgb_buffer_actual_size;  // Track actual allocated size
    void* rgb_buffer_last_write;    // Track last write position
    
    // Timing tracking
    struct timespec last_callback_time;
    struct timespec last_inference_time;
    double avg_processing_time;
    double max_processing_time;
    
    // Statistics
    uint32_t frames_processed;
    uint32_t frames_received;
    uint32_t last_successful_inference;
    float last_inference_result;
    float min_inference_result;
    float max_inference_result;
    float sum_inference_results;
    uint32_t significant_changes;
    uint32_t summary_counter;
    
    // Frame timing
    struct timeval last_frame_received_time;
    float received_fps;
    float input_tensor[1][3][240][240];
};

// External declarations
extern struct obstacle_detection_t obstacle_detection;
extern struct camera_data_t front_camera_data;
extern struct camera_data_t bottom_camera_data;

// Debug functions
void calculate_checksum(const void* data, size_t size, uint32_t* checksum);
void validate_buffers(struct camera_data_t* data, const char* location);
void log_state(struct camera_data_t* data, const char* location);

// Main interface
bool obstacle_detection_init(void);
void obstacle_detection_periodic(void);
void obstacle_detection_cleanup(void);

#endif // OBSTACLE_DETECTION_H