#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/lite/c/c_api.h"
#include "inference.h"

ModelContext* init_inference(const char* model_path) {
    // Load model
    ModelContext* model_ctx = load_model(model_path);
    if (!model_ctx) {
        printf("Failed to load model\n");
        return NULL;
    }

    // Get input tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(model_ctx->interpreter, 0);
    if (!input_tensor) {
        printf("Failed to get input tensor\n");
        cleanup_inference(model_ctx);
        return NULL;
    }

    TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(model_ctx->interpreter, 0);
    if (!output_tensor) {
        printf("[Depth Estimation] Failed to get output tensor\n");
        cleanup_inference(model_ctx);
        return false;
    }

    // Get dimensions from tensor
    int num_dims = TfLiteTensorNumDims(input_tensor);
    if (num_dims != 4) {  // [batch, height, width, channels]
        printf("Expected input tensor with 4 dimensions, got %d\n", num_dims);
        cleanup_inference(model_ctx);
        return NULL;
    }

    // Store input and output dimensions for later use
    model_ctx->input_width = TfLiteTensorDim(input_tensor, 1);
    model_ctx->input_height = TfLiteTensorDim(input_tensor, 2);
    model_ctx->output_height = TfLiteTensorDim(output_tensor, 1);
    model_ctx->output_width = TfLiteTensorDim(output_tensor, 2);

    return model_ctx;
}

DepthMapResult* run_inference(ModelContext* model_ctx, float* preprocessed_data) {
    if (!model_ctx || !model_ctx->interpreter) {
        printf("Invalid model context\n");
        return NULL;
    }

    // Get input tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(model_ctx->interpreter, 0);
    if (!input_tensor) {
        printf("Failed to get input tensor\n");
        return NULL;
    }

    // Calculate input size
    int height = TfLiteTensorDim(input_tensor, 1);
    int width = TfLiteTensorDim(input_tensor, 2);
    int channels = TfLiteTensorDim(input_tensor, 3);
    size_t input_size = height * width * channels * sizeof(float);

    // Copy data to input tensor
    TfLiteStatus status = TfLiteTensorCopyFromBuffer(
        input_tensor,
        preprocessed_data,
        input_size
    );

    if (status != kTfLiteOk) {
        printf("Failed to copy data to input tensor\n");
        return NULL;
    }

    // Perform inference
    status = TfLiteInterpreterInvoke(model_ctx->interpreter);
    if (status != kTfLiteOk) {
        printf("Failed to invoke interpreter\n");
        return NULL;
    }

    // Get output tensor
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(model_ctx->interpreter, 0);
    if (!output_tensor) {
        printf("Failed to get output tensor\n");
        return NULL;
    }

    // Allocate result structure
    DepthMapResult* result = malloc(sizeof(DepthMapResult));
    if (!result) {
        printf("Failed to allocate memory for depth map result\n");
        return NULL;
    }

    // Get output dimensions
    result->height = TfLiteTensorDim(output_tensor, 1);
    result->width = TfLiteTensorDim(output_tensor, 2);
    size_t output_size = result->width * result->height * sizeof(float);

    // Allocate memory for depth map
    result->depth_map = malloc(output_size);
    if (!result->depth_map) {
        printf("Failed to allocate memory for depth map\n");
        free(result);
        return NULL;
    }

    // Copy output data
    status = TfLiteTensorCopyToBuffer(
        output_tensor,
        result->depth_map,
        output_size
    );

    if (status != kTfLiteOk) {
        printf("Failed to get output data\n");
        free(result->depth_map);
        free(result);
        return NULL;
    }

    return result;
}

void free_depth_map_result(DepthMapResult* result) {
    if (result) {
        if (result->depth_map) {
            free(result->depth_map);
        }
        free(result);
    }
}

void cleanup_inference(ModelContext* model_ctx) {
    if (model_ctx) {
        free_model_context(model_ctx); 
    }
}

void save_depth_map(const DepthMapResult* depth_map) {
    if (!depth_map || !depth_map->depth_map) return;

    float min_depth = depth_map->depth_map[0];
    float max_depth = depth_map->depth_map[0];
    float avg_depth = 0;

    int total_pixels = depth_map->width * depth_map->height;
    for (int i = 0; i < total_pixels; i++) {
        float d = depth_map->depth_map[i];
        if (d < min_depth) min_depth = d;
        if (d > max_depth) max_depth = d;
        avg_depth += d;
    }
    avg_depth /= total_pixels;

    printf("Depth Map Statistics:\n");
    printf("Dimensions: %dx%d\n", depth_map->width, depth_map->height);
    printf("Min depth: %.2f\n", min_depth);
    printf("Max depth: %.2f\n", max_depth);
    printf("Average depth: %.2f\n", avg_depth);

    return;
}