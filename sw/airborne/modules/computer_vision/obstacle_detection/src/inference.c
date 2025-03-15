// inference.c
#include "inference.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "model.h"
#include "border_model_wrapper.h"
#include <float.h>     // For FLT_MAX
#include <math.h>      // For fmin, fmax

#include "debug_print.h"
DEFINE_DEBUG_PRINT("INFERENCE")




// Forward declarations for both models
extern void entry_obstacle(const float 
    tensor_input[MODEL_INPUT_BATCH][MODEL_INPUT_CHANNELS][MODEL_INPUT_HEIGHT][MODEL_INPUT_WIDTH], 
    float tensor_output[1][1][MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE]);

extern void entry_border(const float 
    tensor_input[1][3][240][240], 
    float tensor_output[1][1]);
    
typedef float obstacle_input_tensor_t[MODEL_INPUT_BATCH][MODEL_INPUT_CHANNELS][MODEL_INPUT_HEIGHT][MODEL_INPUT_WIDTH];
typedef float obstacle_output_tensor_t[1][1][MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE];
typedef float border_input_tensor_t[1][3][240][240];  
typedef float border_output_tensor_t[1][1];  

static obstacle_input_tensor_t* obstacle_input = NULL;
static obstacle_output_tensor_t* obstacle_output = NULL;
static border_input_tensor_t* border_input = NULL;
static border_output_tensor_t* border_output = NULL;

// void clip_tensor_values(float* tensor, size_t size, float min_val, float max_val) {
//     for(size_t i = 0; i < size; i++) {
//         if(tensor[i] < min_val) tensor[i] = min_val;
//         if(tensor[i] > max_val) tensor[i] = max_val;
//     }
// }

// void normalize_input_tensor(float* tensor, int width, int height, int channels) {
//     // First pass - get mean and std
//     float sum = 0.0f;
//     float sq_sum = 0.0f;
//     int size = width * height * channels;
    
//     for(int i = 0; i < size; i++) {
//         sum += tensor[i];
//         sq_sum += tensor[i] * tensor[i];
//     }
    
//     float mean = sum / size;
//     float std = sqrtf(sq_sum/size - mean*mean);
    
//     // Second pass - normalize
//     for(int i = 0; i < size; i++) {
//         tensor[i] = (tensor[i] - mean) / (std + 1e-5f);
//         // Clip to reasonable range
//         if(tensor[i] < -3.0f) tensor[i] = -3.0f;
//         if(tensor[i] > 3.0f) tensor[i] = 3.0f;
//     }
// }

// void normalize_intermediate_tensor(float* tensor, int size) {
//     float sum = 0.0f;
//     float sq_sum = 0.0f;
    
//     for(int i = 0; i < size; i++) {
//         sum += tensor[i];
//         sq_sum += tensor[i] * tensor[i];
//     }
    
//     float mean = sum / size;
//     float std = sqrtf(sq_sum/size - mean*mean);
    
//     for(int i = 0; i < size; i++) {
//         tensor[i] = (tensor[i] - mean) / (std + 1e-5f);
//         // Clip to reasonable range
//         if(tensor[i] < -3.0f) tensor[i] = -3.0f;
//         if(tensor[i] > 3.0f) tensor[i] = 3.0f;
//     }
// }

bool init_inference(void) {
    // Allocate tensors for obstacle detection
    obstacle_input = (obstacle_input_tensor_t*)malloc(sizeof(obstacle_input_tensor_t));
    obstacle_output = (obstacle_output_tensor_t*)malloc(sizeof(obstacle_output_tensor_t));
    
    // Allocate tensors for border detection
    border_input = (border_input_tensor_t*)malloc(sizeof(border_input_tensor_t));
    border_output = (border_output_tensor_t*)malloc(sizeof(border_output_tensor_t));
    
    if (!obstacle_input || !obstacle_output || !border_input || !border_output) {
        printf("[Inference] Failed to allocate tensors\n");
        cleanup_inference();
        return false;
    }

    return true;
}

bool run_obstacle_inference(const float* rgb_data, 
    int width,
    int height,
    struct model_output_t* output) {
    if (!obstacle_input || !obstacle_output || !rgb_data || !output) {
        return false;
    }

    if (width != MODEL_INPUT_WIDTH || height != MODEL_INPUT_HEIGHT) {
        printf("[Inference] Invalid obstacle input dimensions: %dx%d (expected %dx%d)\n", 
        width, height, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
        return false;
    }

    // Get pointers to the RGB channels
    const float* r_data = rgb_data;
    const float* g_data = rgb_data + (width * height);
    const float* b_data = rgb_data + (2 * width * height);

    // Scale to [0,255] range and copy to input tensor
    // Assuming input tensor format is [batch][channel][height][width]
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            int src_idx = h * width + w;
            // Red channel
            (*obstacle_input)[0][0][h][w] = r_data[src_idx] * 255.0f;
            // Green channel
            (*obstacle_input)[0][1][h][w] = g_data[src_idx] * 255.0f;
            // Blue channel
            (*obstacle_input)[0][2][h][w] = b_data[src_idx] * 255.0f;
        }
    }

    memset(obstacle_output, 0, sizeof(obstacle_output_tensor_t));

    entry_obstacle(*obstacle_input, *obstacle_output);

    // Copy results to output structure 
    for (int i = 0; i < MODEL_OUTPUT_ROW_SIZE; i++) {
        for (int j = 0; j < MODEL_OUTPUT_COL_SIZE; j++) {
            output->values[i][j] = (*obstacle_output)[0][0][i][j];
        }
    }

    return true;
}

bool run_border_inference(const float* rgb_data, int width, int height, struct border_output_t* output) {
    if (!rgb_data || !output) {
        debug_print("Null pointer in run_border_inference");
        return false;
    }

    if (width != 240 || height != 240) {
        debug_print("Invalid dimensions: %dx%d", width, height);
        return false;
    }

    static int local_frame_counter = 0;
    static float period_min = FLT_MAX;
    static float period_max = -FLT_MAX;
    static float period_sum = 0;
    static int period_frames = 0;
    static float last_output = 0;
    static int stable_frames = 0;
    static int significant_changes = 0;

    local_frame_counter++;
    frame_counter = local_frame_counter;  // Update global counter used by model

    // Track input statistics
    const float* r_data = rgb_data;
    const float* g_data = rgb_data + (width * height);
    const float* b_data = rgb_data + (2 * width * height);

    float input_mean[3] = {0, 0, 0};
    float input_min[3] = {1.0f, 1.0f, 1.0f};
    float input_max[3] = {0.0f, 0.0f, 0.0f};

    // Calculate statistics for each channel
    for(int c = 0; c < 3; c++) {
        const float* channel_data = rgb_data + (c * width * height);
        for(int i = 0; i < width * height; i++) {
            input_mean[c] += channel_data[i];
            input_min[c] = fmin(input_min[c], channel_data[i]);
            input_max[c] = fmax(input_max[c], channel_data[i]);
        }
        input_mean[c] /= (width * height);
    }

    // Create input tensor and normalize
    float input_tensor[1][3][240][240] = {0};
    const float means[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float stds[3] = {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f};
    
    // Process input data with normalization
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            int src_idx = h * width + w;
            float r = r_data[src_idx] * 255.0f;
            float g = g_data[src_idx] * 255.0f;
            float b = b_data[src_idx] * 255.0f;
    
            input_tensor[0][0][h][w] = (r - means[0]) / stds[0];
            input_tensor[0][1][h][w] = (g - means[1]) / stds[1];
            input_tensor[0][2][h][w] = (b - means[2]) / stds[2];
        }
    }

    // Set debug level for critical period
    if (local_frame_counter >= 300 && local_frame_counter <= 400) {
        current_debug_level = DEBUG_MODERATE;
        // current_debug_level = DEBUG_NONE;
    } else {
        current_debug_level = DEBUG_NONE;
    }

    // Run inference
    float output_tensor[1][1] = {0};
    entry_border(input_tensor, output_tensor);

    float model_output = output_tensor[0][0];
    bool is_unstable = false;

    // Update period statistics
    period_min = fmin(period_min, model_output);
    period_max = fmax(period_max, model_output);
    period_sum += model_output;
    period_frames++;

    // Detect instability
    if (isnan(model_output)) {
        debug_print("Frame %d: WARNING - NaN output", local_frame_counter);
        model_output = 0.0f;
        is_unstable = true;
        current_debug_level = DEBUG_VERBOSE;
    } else if (model_output > 1.0f || model_output < 0.0f) {
        debug_print("Frame %d: WARNING - Output out of range: %.3f", 
            local_frame_counter, model_output);
        model_output = fmax(0.0f, fmin(1.0f, model_output));
        is_unstable = true;
        current_debug_level = DEBUG_VERBOSE;
    } else if (fabs(model_output - last_output) > 0.1) {
        is_unstable = true;
        current_debug_level = DEBUG_VERBOSE;
        significant_changes++;

        debug_print("\nFrame %d: Significant change detected:", local_frame_counter);
        debug_print("  Previous: %.6f", last_output);
        debug_print("  Current:  %.6f", model_output);
        debug_print("  Delta:    %.6f", fabs(model_output - last_output));
        
        if (stable_frames > 50) {
            debug_print("  First instability after %d stable frames", stable_frames);
        }
        
        stable_frames = 0;
    } else {
        stable_frames++;
    }

    // Print period summary
    // if (period_frames == 100) {
    //     float avg = period_sum / period_frames;
    //     float range = period_max - period_min;
        
    //     debug_print("\nFrames %d-%d Summary:", 
    //         local_frame_counter - 99, local_frame_counter);
    //     debug_print("Avg: %.6f, Range: %.6f [%.6f, %.6f]", 
    //         avg, range, period_min, period_max);
    //     debug_print("Changes: %d (%.2f%% of frames)", 
    //         significant_changes, 
    //         (float)significant_changes / period_frames * 100.0f);

    //     // Reset for next period
    //     period_min = FLT_MAX;
    //     period_max = -FLT_MAX;
    //     period_sum = 0;
    //     period_frames = 0;
    //     significant_changes = 0;
    // }

    // During critical period (300-400), print intermediate statistics every 20 frames
    if (local_frame_counter >= 300 && local_frame_counter <= 400 && 
        local_frame_counter % 20 == 0) {
        debug_print("\nFrame %d Critical Period Stats:", local_frame_counter);
        debug_print("  Current stability: %d frames", stable_frames);
        debug_print("  Recent range: [%.6f, %.6f]", period_min, model_output);
        debug_print("  Recent average: %.6f", period_sum / period_frames);
    }

    last_output = model_output;
    output->value = model_output;
    return true;
}

void cleanup_inference(void) {
    free(obstacle_input);
    free(obstacle_output);
    free(border_input);
    free(border_output);
    obstacle_input = NULL;
    obstacle_output = NULL;
    border_input = NULL;
    border_output = NULL;
}